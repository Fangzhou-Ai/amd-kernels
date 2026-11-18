# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""K1 ``kda_pre``: one Kimi-K3 KDA layer from the AttnRes seam to the gated core
output, for a spec-verify step (R requests x L tokens, S = R L rows).

One launch a layer, ``BLOCKS`` CTAs x ``THREADS`` threads, in three roles so that
no CTA that polls a mailbox has weight loads in flight (a poll's load retires
behind every earlier load):

    attn CTAs 0 .. 55   slice (128 hidden columns each): prefix (+ delta) ->
                        bf16 prefix out, per-source partial sum of squares and
                        qk dot (PART); gate (token t on CTA t): softmax weights
                        (WGT); mix: the softmax mix, its partial sum of squares
                        (MSQ), every slice's -> output RMSNorm -> bf16 x rows
                        (plain device-scope stores, an XRDY flag a slice)
    gemv CTAs 56 ..     in_proj, two 16-row groups each: the first group's
                        weights issued at kernel start, the second's as x lands
                        -> q|k|v|g|f_a|beta (PROJ)
    kda tasks           (R x 12 heads x 8 v-chunks) on CTAs 0 ..: inputs that
                        need no hand-off loaded first (metadata, conv window,
                        state rows, f_b rows), then conv update of the owned
                        channels (q / k out through QK), f_b MFMA + lower-bound
                        gate, l2norm, the recurrence on 16 state rows (a state a
                        token written), partial norms (ONRM), gated RMSNorm ->
                        core_out

Rounding follows the original path where it is elementwise (the AttnRes prefix
add, every bf16 hand-off between the original kernels); split-K reductions
differ only in their order.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from amd_kernels.common import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from amd_kernels.common.mono.debug import (
    region_ids,
    stamp,
    stamp_begin,
    stamp_flush,
)
from amd_kernels.common.mono.ops import (
    CM_DEV,
    CM_NT,
    bf_hi,
    bf_lo,
    butterfly,
    hw_exp2,
    hw_rsq,
    ld_i32,
    lds_dma,
    dma_wait,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_max,
    wave_sum,
)
from amd_kernels.common.mono.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    first_task,
    key_tuple,
    pair_layout,
    source_digest,
)
from amd_kernels.common.mono.sync import (
    Mailbox,
    publish,
    sreg,
)

HIDDEN = 7168
HD = 128
NH = 12  # local heads at TP8
PROJ = NH * HD  # 1536
QKV = 3 * PROJ
G2_0 = QKV
FA_0 = G2_0 + PROJ  # 6144
BETA_0 = FA_0 + HD  # 6272
NPROJ = 6288  # in_proj rows a rank holds (6284 + 4 pad)
ROWS = 16
KCH = 128  # K a chunk: 4 MFMAs of 32
NKCH = HIDDEN // KCH  # 56
PER_WAVE = NKCH // WAVES  # 7
VCH = 16  # state rows a KDA task
NVCH = HD // VCH  # 8
CONV_W = 4
LOG2E = 1.0 / math.log(2.0)
LDS_PAD = 8  # bf16 a padded x row
MAX_S = 16  # rows a launch
XL_BYTES = 163_200 - 192  # LDS of the x rows: all of it but the timeline stamps  # LDS of the x rows
WGT0 = 256  # red: the source weights
MSQ0 = 512  # red: the mix's per-wave sums of squares
# timeline points: start, slice, gate, mix, load_x, gemv0, gemv1, k.stage,
# k.conv, k.gates, k.recur, k.norm
TL_POINTS = 24


_STREAM = fx.Stream(None)
REGIONS = ("xrdy", "proj", "onrm")
# every mono source file: in every build's JIT cache key
_SOURCES = source_digest("common/mono", "attention/kda_mono_decode_k1_h12_d128_ranks8")


@dataclass(frozen=True)
class KdaPreBuild:
    tokens: int  # S
    qlen: int  # L, tokens a request
    nblocks: int  # AttnRes block sources (the prefix is one more)
    delta: bool
    state_len: int  # conv state entries: CONV_W - 1 + num_spec
    eps: float = 1e-5  # AttnRes norm
    out_eps: float = 1e-5  # input_layernorm
    onorm_eps: float = 1e-5
    lower_bound: float = -5.0
    timeline: bool = False
    write_idx: int = -1  # a block-write layer's block: the updated prefix stored there
    # Per-token recurrent-state snapshots (S*NH*64 KB a layer) stored
    # non-temporal + device scope: they stream out instead of sitting dirty in
    # L2 and draining at kernel end (~1.3 us a launch). Bit-exact.
    stream_states: bool = False


ABI = KernelAbi(
    (
        "prefix",
        "delta",
        "blocks",
        "blk_sm",
        "blk_sr",
        "ares_nw",
        "ares_qk",
        "in_nw",
        "w_in",
        "w_fb",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "on_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "st_stride",
        "num_acc",
        "core_out",
        "scratch",
        "layer",
        "epoch",
        "health",
        "alarm",
        "tl",
    )
)


def scratch_layout(key: KdaPreBuild) -> dict:
    s, ns = key.tokens, key.nblocks + 1
    assert s % key.qlen == 0
    pairs = {
        "xrdy": s,
        "proj": s * NPROJ,
        "onrm": s * NH * NVCH,
    }
    lay = pair_layout(pairs.items())
    end = max(o + n for o, n in lay.values())
    # the plain bf16 x rows (not pairs), 16 B aligned
    lay["x"] = ((end + 15) // 16 * 16, s * HIDDEN * 2)
    return lay


def scratch_bytes(key: KdaPreBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


TAG_SLOTS = 256  # a step's mailbox tags: epoch x 256 + a launch's slot


def step_tag(epoch, slot):
    """This launch's mailbox tag: the step's epoch (a device counter the runner
    bumps once a step, in the graph) and the launch's slot, so no step's or
    other launch's pair ever passes for this one's -- nothing is zeroed."""
    e = uniform(ld_i32(epoch, 0))
    return (e % (1 << 22)) * TAG_SLOTS + slot + 1


def ld_bf(ptr, i, cm=0):
    return fx.Float32(
        fx.BFloat16(
            bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16, cache_modifier=cm)
        )
    )


def ld_bf4(ptr, i):
    """bf16 elements i .. i + 4 (i a multiple of 4) as four f32."""
    w = fx.Vector(bo.buffer_load(rsrc(ptr), i // 2, vec_width=2, dtype=T.i32))
    return [bf_lo(w[0]), bf_hi(w[0]), bf_lo(w[1]), bf_hi(w[1])]


def bf4_words(v):
    """Four f32 -> two words of packed bf16."""
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def bf16r(v):
    return fx.Float32(fx.Float32(v).to(fx.BFloat16))


def sigmoid(z):
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def f32_of(w):
    return w.bitcast(fx.Float32)


def sum_polled(vals):
    acc = f32_of(vals[0][0])
    for k in range(1, len(vals)):
        acc = acc + f32_of(vals[k][0])
    return acc


def token_tiles(s, n=16):
    """(first token, rows) of each tile of ``n`` rows over S = ``s``."""
    return [(t0, min(n, s - t0)) for t0 in range(0, s, n)]


# ---------------------------------------------------------------- attn CTAs
def ld16(ptr, i, cm=0):
    """bf16 elements i .. i + 8 (i a multiple of 8) as four words."""
    return fx.Vector(
        bo.buffer_load(
            rsrc(ptr), i // 2, vec_width=4, dtype=T.i32, cache_modifier=cm
        )
    )


def unpack8(w):
    return [f for j in range(4) for f in (bf_lo(w[j]), bf_hi(w[j]))]


def pack8(v):
    return fx.Vector.from_elements(v, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)


@traced
def stage_attn(c, t):
    """Token t, whole: the prefix (+ delta) rounded to bf16 and stored back, the
    softmax weights of the sources, the mix, its RMSNorm -> the bf16 x row, and
    the token's XRDY flag."""
    nb, ns, tid, lane, wave, red = (
        c["NB"], c["NS"], c["tid"], c["lane"], c["wave"], c["red"]
    )  # fmt: skip
    a = c["args"]
    chunks = HIDDEN // 8
    nj = (chunks + THREADS - 1) // THREADS
    live = [(tid + THREADS * j) < chunks for j in range(nj)]
    col = [fx.min(tid + THREADS * j, chunks - 1) * 8 for j in range(nj)]
    blk = [
        [ld16(a["blocks"], t * a["blk_sm"] + s * a["blk_sr"] + col[j]) for s in range(nb)]
        for j in range(nj)
    ]
    pre = [ld16(a["prefix"], t * HIDDEN + col[j]) for j in range(nj)]
    dl = [ld16(a["delta"], t * HIDDEN + col[j]) for j in range(nj)] if c["delta"] else None
    wq = []
    for j in range_constexpr(nj):
        w = unpack8(ld16(a["ares_nw"], col[j]))
        q = unpack8(ld16(a["ares_qk"], col[j]))
        wq.append([w[e] * q[e] for e in range(8)])
    wn = [unpack8(ld16(a["in_nw"], col[j])) for j in range(nj)]
    vs = []  # vs[j][s]: 8 f32 of (token t, source s)
    for j in range_constexpr(nj):
        pf = unpack8(pre[j])
        if const_expr(c["delta"]):
            df = unpack8(dl[j])
            pf = [bf16r(pf[e] + df[e]) for e in range(8)]
            if live[j]:
                bo.buffer_store(pack8(pf), rsrc(a["prefix"]), (t * HIDDEN + col[j]) // 2)
        if const_expr(c["write_idx"] >= 0):
            if live[j]:
                bo.buffer_store(
                    pack8(pf),
                    rsrc(a["blocks"]),
                    (t * a["blk_sm"] + c["write_idx"] * a["blk_sr"] + col[j]) // 2,
                )
        zero = fx.Float32(0.0)
        row = [[live[j].select(f, zero) for f in unpack8(b)] for b in blk[j]]
        row.append([live[j].select(f, zero) for f in pf])
        vs.append(row)
    stamp(c["on"], c["tls"], tid, 1)
    # per source: sum of squares and the logit's dot, every thread's chunks
    parts = []
    for s in range_constexpr(ns):
        sq = fx.Float32(0.0)
        dt = fx.Float32(0.0)
        for j in range_constexpr(nj):
            sj = fx.Float32(0.0)
            dj = fx.Float32(0.0)
            for e in range_constexpr(8):
                sj = sj + vs[j][s][e] * vs[j][s][e]
                dj = dj + vs[j][s][e] * wq[j][e]
            sq = sq + sj
            dt = dt + dj
        parts += [wave_sum(sq), wave_sum(dt)]
    stamp(c["on"], c["tls"], tid, 12)
    if lane == 0:
        for k in range_constexpr(2 * ns):
            fx.ptr_store(parts[k], red + (wave * 2 * ns + k))
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 13)
    if wave == 0:
        src = fx.min(lane, ns - 1)
        sq = fx.ptr_load(red + src * 2)
        dt = fx.ptr_load(red + src * 2 + 1)
        for w in range_constexpr(1, WAVES):
            sq = sq + fx.ptr_load(red + (w * 2 * ns + src * 2))
            dt = dt + fx.ptr_load(red + (w * 2 * ns + src * 2 + 1))
        rstd = hw_rsq(sq * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["eps"]))
        logit = dt * rstd * fx.Float32(LOG2E)
        logit = (lane < ns).select(logit, fx.Float32(-3.0e38))
        m = wave_max(logit)
        ex = (lane < ns).select(hw_exp2(logit - m), fx.Float32(0.0))
        den = butterfly(ex, (32, 16, 8, 4, 2, 1))
        if lane < ns:
            fx.ptr_store(ex, red + (WGT0 + lane))
        if lane == 0:
            fx.ptr_store(den, red + (WGT0 + ns))
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 14)
    wt = [fx.ptr_load(red + (WGT0 + s)) for s in range(ns)]
    den = fx.ptr_load(red + (WGT0 + ns))
    mixed = []
    msq = fx.Float32(0.0)
    for j in range_constexpr(nj):
        mj = [fx.Float32(0.0) for _ in range(8)]
        for s in range_constexpr(ns):
            mj = [mj[e] + wt[s] * vs[j][s][e] for e in range(8)]
        mj = [mj[e] / den for e in range(8)]
        mixed.append(mj)
        sj = fx.Float32(0.0)
        for e in range_constexpr(8):
            sj = sj + mj[e] * mj[e]
        msq = msq + live[j].select(sj, fx.Float32(0.0))
    msq = wave_sum(msq)
    if lane == 0:
        fx.ptr_store(msq, red + (MSQ0 + wave))
    gpu.barrier()
    stamp(c["on"], c["tls"], tid, 22)
    tot = fx.ptr_load(red + MSQ0)
    for w in range_constexpr(1, WAVES):
        tot = tot + fx.ptr_load(red + (MSQ0 + w))
    r = hw_rsq(tot * fx.Float32(1.0 / HIDDEN) + fx.Float32(c["out_eps"]))
    for j in range_constexpr(nj):
        xw = pack8([mixed[j][e] * r * wn[j][e] for e in range(8)])
        if live[j]:
            bo.buffer_store(
                xw, rsrc(c["xbuf"]), (t * HIDDEN + col[j]) // 2, cache_modifier=CM_DEV
            )
    stamp(c["on"], c["tls"], tid, 2)
    publish(c["put"], c["xrdy"], t, fx.Int32(1), tid == 0)


# ---------------------------------------------------------------- gemv CTAs
def proj_row(lr):
    """The in_proj row of logical row ``lr``: f_a / beta first, then q, k, v, g."""
    return lr + FA_0 - (lr >= NPROJ - FA_0).select(fx.Int32(NPROJ), fx.Int32(0))


def gemv_loads(c, lr0, nrows, chunks):
    """This wave's weight operands of logical rows ``lr0`` .. + ``nrows`` for its
    K chunks ``chunks`` (indices into wave, wave + 8, ..): a lane 64 contiguous
    bytes of its row a chunk (4 dwordx4). A row past ``nrows`` or the last
    loads nothing (an out-of-range buffer offset)."""
    lane, wave = c["lane"], c["wave"]
    r_w = rsrc(c["args"]["w_in"], NPROJ * HIDDEN * 2)
    lr = lr0 + lane % ROWS
    valid = (lane % ROWS < nrows) & (lr < NPROJ)
    row = valid.select(proj_row(lr), fx.Int32(NPROJ))
    g = lane // ROWS
    out = []
    for i in chunks:
        k = (wave + WAVES * i) * KCH + g * 8
        base = (row * HIDDEN + k) // 2  # dwords
        out.append(
            [
                fx.Vector(
                    bo.buffer_load(
                        r_w,
                        base + 16 * q,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=0,
                    )
                )
                for q in range(4)
            ]
        )
    return out


@traced
def wait_x(c):
    """Every token's x row ready."""
    if c["tid"] < c["S"]:
        c["poll"]([(c["xrdy"], c["tid"], 1)])
    gpu.barrier()


@traced
def load_x(c):
    """The bf16 x rows into LDS in block order (``xblock``): a 1 KB DMA a (row,
    piece of 16 blocks of 32 columns), every wave's in flight together. The
    last blocks past ``c["XB"]`` stay out (``x_direct``)."""
    lane, wave = c["lane"], c["wave"]
    nb = c["XB"]
    pieces = (nb + 15) // 16
    total = c["S"] * pieces
    for i in range_constexpr((total + WAVES - 1) // WAVES):
        idx = wave + WAVES * i
        t = idx // pieces
        p = idx % pieces
        blk = p * 16 + lane // 4
        col = xblock_col(blk) + (lane % 4) * 8
        src = c["xbuf"] + fx.Int64((t * HIDDEN + col) * 2)
        dst = c["xl_b"] + t * c["XROW"] + p * 1024
        if const_expr(total % WAVES == 0 and nb % 16 == 0):
            lds_dma(src, dst, "sc1")
        else:
            if (idx < total) & (blk < nb):
                lds_dma(src, dst, "sc1")


def xblock_col(blk):
    """First column of x block ``blk``: the 32 columns of MFMA step q of K chunk i
    of wave w, blk = (4 i + q) 8 + w."""
    i = blk // 32
    q = (blk // 8) % 4
    w = blk % 8
    return (w + WAVES * i) * KCH + 32 * q


def x_direct(c):
    """The B operands of the x blocks past the LDS ones, straight from the x
    rows: {(i, q): 4 words a lane}."""
    lane, wave = c["lane"], c["wave"]
    t = fx.min(lane % ROWS, c["S"] - 1)
    g = lane // ROWS
    out = {}
    for i in range(PER_WAVE):
        for q in range(4):
            if (4 * i + q) * 8 + 8 > c["XB"]:
                k = (wave + WAVES * i) * KCH + g * 8
                out[(i, q)] = fx.Vector(
                    bo.buffer_load(
                        rsrc(c["xbuf"]),
                        (t * HIDDEN + k + 32 * q) // 2,
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
    return out


def vm_wait(n):
    llvm.inline_asm(None, [], f"s_waitcnt vmcnt({n})", "", has_side_effects=True)


@traced
def stage_gemv(c, lr_a, ops_a, lr_b, n_b, ops_b, xd):
    """Logical rows ``lr_a`` .. + 16 and ``lr_b`` .. + ``n_b`` of in_proj against
    the S x rows (each B operand read once for both) -> bf16 PROJ."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    g = lane // ROWS
    n = c["S"]
    t = fx.min(lane % ROWS, n - 1)
    nb = c["XB"]
    acc_a = fx.Vector.filled(4, 0.0, fx.Float32)
    acc_b = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(PER_WAVE):
        for q in range_constexpr(4):
            base = (4 * i + q) * 8
            if const_expr(base >= nb):
                xb = xd[(i, q)]
            else:
                xb = fx.Vector(
                    fx.ptr_load(
                        c["xl"] + (t * c["XROW"] + (base + wave) * 64 + g * 16) // 4,
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                )
                if const_expr(base + 8 > nb):
                    xb = fx.Vector.from_elements(
                        [(wave >= nb - base).select(xd[(i, q)][e], xb[e]) for e in range(4)],
                        fx.Int32,
                    )
            xb = xb.bitcast(fx.BFloat16)
            acc_a = mfma_bf16(ops_a[i][q].bitcast(fx.BFloat16), xb, acc_a)
            acc_b = mfma_bf16(ops_b[i][q].bitcast(fx.BFloat16), xb, acc_b)
    gpu.barrier()  # every wave done reading x: ``red`` shares its LDS
    fx.ptr_store(acc_a, red + (wave * 64 + lane) * 4)
    fx.ptr_store(acc_b, c["red2"] + (wave * 64 + lane) * 4)
    gpu.barrier()
    r = tid % ROWS
    tt = tid // ROWS
    if tid < ROWS * n:
        v = bf16r(row_sum(red, r, tt))
        c["put"](c["proj"], tt * NPROJ + proj_row(lr_a + r), v)
        if (r < n_b) & (lr_b + r < NPROJ):
            vb = bf16r(row_sum(c["red2"], r, tt))
            c["put"](c["proj"], tt * NPROJ + proj_row(lr_b + r), vb)
    gpu.barrier()


# ---------------------------------------------------------------- kda
NCONV = 2 * HD + VCH  # conv channels a task computes: its head's q and k, its v chunk


def kda_stage(c, req, h, ch):
    """The task's inputs that need no hand-off, issued before its first poll:
    metadata, the conv channels' window / weights, the initial state rows, the
    f_b rows of this wave and the per-head constants."""
    L, sl, tid, lane, wave = c["L"], c["SL"], c["tid"], c["lane"], c["wave"]
    a = c["args"]
    k = {}
    # every value a buffer resource is built on wave-uniform (else a waterfall)
    k["acc_idx"] = uniform(ld_i32(a["num_acc"], req)) - 1
    k["slot_c"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"]))
    k["slot0"] = uniform(ld_i32(a["st_idx"], req * a["st_stride"] + k["acc_idx"]))
    k["slots"] = [
        uniform(ld_i32(a["st_idx"], req * a["st_stride"] + j)) for j in range(L)
    ]
    # conv: thread < NCONV owns a channel: q / k channel d of the head, or v
    # channel d of this task's chunk
    tc = fx.min(tid, NCONV - 1)
    isqk = tc < 2 * HD
    part = isqk.select(tc // HD, fx.Int32(2))
    d = isqk.select(tc % HD, ch * VCH + (tc - 2 * HD))
    chan = part * PROJ + h * HD + d
    k["own"] = (part == 2) | ((d // VCH) == ch)
    k["part"], k["d"], k["chan"] = part, d, chan
    cs = a["conv_st"] + fx.Int64(fx.max(k["slot_c"], 0)) * fx.Int64(a["cs_seq"]) * 2
    co = chan * a["cs_dim"]  # the channel's first entry (bf16 elements)
    k["cs"], k["co"] = cs, co
    k["cw"] = fx.Vector(
        bo.buffer_load(rsrc(a["conv_w"]), chan * CONV_W, vec_width=4, dtype=T.f32)
    )
    k["win"] = [
        ld_bf(cs, co + (k["acc_idx"] + j) * a["cs_tok"]) for j in range(CONV_W - 1)
    ]
    k["old"] = [
        ld_bf(cs, co + (k["acc_idx"] + 1 + j) * a["cs_tok"]) for j in range(sl - L)
    ]
    # recurrence: thread -> state row v = tid / 32, k columns (tid % 32) 4 ..
    vr = ch * VCH + tid // 32
    k0 = (tid % 32) * 4
    k["head_off"] = (h * HD + vr) * HD + k0
    base0 = a["rstate"] + fx.Int64(fx.max(k["slot0"], 0)) * fx.Int64(a["rs_seq"]) * 4
    k["st"] = fx.Vector(
        bo.buffer_load(rsrc(base0), k["head_off"], vec_width=4, dtype=T.f32)
    )
    # f_b: wave w's rows 16 w .. of head h, a lane 64 B of its row
    fr = h * HD + wave * 16 + lane % 16
    fb = (fr * HD + (lane // 16) * 32) // 2
    k["fb"] = [
        fx.Vector(bo.buffer_load(rsrc(a["w_fb"]), fb + 4 * q, vec_width=4, dtype=T.i32))
        for q in range(4)
    ]
    k["aa"] = hw_exp2(ld_f32(a["a_log"], h) * fx.Float32(LOG2E))
    k["dtb"] = [
        ld_f32(a["dt_bias"], h * HD + wave * 16 + 4 * (lane // 16) + i)
        for i in range(4)
    ]
    return k


@traced
def kda_inputs(c, k, req, h):
    """One poll batch for everything the task reads from PROJ: its conv channels'
    L inputs, f_a and beta. The spec conv update of request ``req``'s L tokens
    -> new conv state of the owned channels; q / k (raw conv outputs) and v to
    LDS; f_a and sigmoid(beta) to LDS."""
    L, sl, tid = c["L"], c["SL"], c["tid"]
    a = c["args"]
    n_fa = L * HD
    nfa = (n_fa + THREADS - 1) // THREADS
    specs = [
        (c["proj"], (req * L + j) * NPROJ + k["chan"], 1) for j in range(L)
    ]
    fa_at = []
    for i in range_constexpr(nfa):
        e = fx.min(tid + THREADS * i, n_fa - 1)
        fa_at.append(e)
        specs.append((c["proj"], (req * L + e // HD) * NPROJ + FA_0 + e % HD, 1))
    bt = fx.min(tid, L - 1)
    specs.append((c["proj"], (req * L + bt) * NPROJ + BETA_0 + h, 1))
    got = c["poll"](specs)
    for i in range_constexpr(nfa):
        if tid + THREADS * i < n_fa:
            fx.ptr_store(f32_of(got[L + i][0]), c["fa"] + fa_at[i])
    if tid < L:
        fx.ptr_store(sigmoid(f32_of(got[L + nfa][0])), c["beta"] + tid)
    if tid < NCONV:
        xv = [f32_of(got[j][0]) for j in range(L)]
        rs = rsrc(k["cs"])
        if k["own"] & (k["slot_c"] > 0):
            for j in range_constexpr(sl):
                nv = k["old"][j] if j < sl - L else xv[j - (sl - L)]
                bo.buffer_store(
                    fx.Float32(nv).to(fx.BFloat16), rs, k["co"] + j * a["cs_tok"]
                )
        seq = k["win"] + xv
        for j in range_constexpr(L):
            acc = fx.Float32(0.0)
            for q in range_constexpr(CONV_W):
                acc = acc + k["cw"][q] * seq[j + q]
            o = bf16r(acc * sigmoid(acc))
            if k["part"] < 2:
                fx.ptr_store(o, c["qkl"] + ((2 * j + k["part"]) * HD + k["d"]))
            if k["part"] == 2:
                fx.ptr_store(o, c["vloc"] + (j * VCH + (tid - 2 * HD)))
    gpu.barrier()


@traced
def kda_gates(c, k, req, h):
    """g1 = f_b f_a by MFMA (a wave 16 rows) -> lower-bound gate -> decay in LDS;
    q / k l2-normed (q scaled)."""
    L, tid, lane, wave = c["L"], c["tid"], c["lane"], c["wave"]
    t = fx.min(lane % 16, L - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for q in range_constexpr(4):
        kb = (lane // 16) * 32 + 8 * q
        xs = [fx.ptr_load(c["fa"] + (t * HD + kb + e)) for e in range(8)]
        xb = fx.Vector.from_elements(xs, fx.Float32).to(fx.BFloat16)
        acc = mfma_bf16(k["fb"][q].bitcast(fx.BFloat16), xb, acc)
    for i in range_constexpr(4):
        kk = wave * 16 + 4 * (lane // 16) + i
        g = bf16r(acc[i]) + k["dtb"][i]
        gate = fx.Float32(c["lb"]) * sigmoid(k["aa"] * g)
        if lane % 16 < L:
            fx.ptr_store(hw_exp2(gate * fx.Float32(LOG2E)), c["decay"] + (t * HD + kk))
    # l2norm: q / k row r (2 L rows of 128) on wave r % 8
    for rr in range_constexpr((2 * L + WAVES - 1) // WAVES):
        row = fx.min(wave + WAVES * rr, 2 * L - 1)
        x0 = fx.ptr_load(c["qkl"] + (row * HD + lane))
        x1 = fx.ptr_load(c["qkl"] + (row * HD + 64 + lane))
        ss = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
        sc = hw_rsq(ss + fx.Float32(1e-6))
        sc = (row % 2 == 0).select(sc * fx.Float32(HD**-0.5), sc)
        if wave + WAVES * rr < 2 * L:
            fx.ptr_store(x0 * sc, c["qkl"] + (row * HD + lane))
            fx.ptr_store(x1 * sc, c["qkl"] + (row * HD + 64 + lane))
    gpu.barrier()


@traced
def kda_recur(c, k, req, h, ch):
    """State rows 16 ch .. of head h: the recurrence over the L tokens, a state
    written a token, bf16 outputs to LDS ``ol`` and their partial norms (ONRM)."""
    L, tid, lane = c["L"], c["tid"], c["lane"]
    a = c["args"]
    v = tid // 32
    k0 = (tid % 32) * 4
    live0 = k["slot0"] > 0
    stv = [live0.select(k["st"][e], fx.Float32(0.0)) for e in range(4)]
    for j in range_constexpr(L):
        qv = [fx.ptr_load(c["qkl"] + ((2 * j) * HD + k0 + e)) for e in range(4)]
        kv = [fx.ptr_load(c["qkl"] + ((2 * j + 1) * HD + k0 + e)) for e in range(4)]
        dk = [fx.ptr_load(c["decay"] + (j * HD + k0 + e)) for e in range(4)]
        stv = [stv[e] * dk[e] for e in range(4)]
        p = stv[0] * kv[0] + stv[1] * kv[1] + stv[2] * kv[2] + stv[3] * kv[3]
        p = butterfly(p, (16, 8, 4, 2, 1))
        vn = (fx.ptr_load(c["vloc"] + (j * VCH + v)) - p) * fx.ptr_load(c["beta"] + j)
        stv = [stv[e] + vn * kv[e] for e in range(4)]
        o = stv[0] * qv[0] + stv[1] * qv[1] + stv[2] * qv[2] + stv[3] * qv[3]
        o = bf16r(butterfly(o, (16, 8, 4, 2, 1)))
        o = live0.select(o, fx.Float32(0.0))
        slot = k["slots"][j]
        if (slot > 0) & live0:
            dst = rsrc(a["rstate"] + fx.Int64(slot) * fx.Int64(a["rs_seq"]) * 4)
            bo.buffer_store(
                fx.Vector.from_elements(stv, fx.Float32),
                dst,
                k["head_off"],
                cache_modifier=c["st_cm"],
            )
        if lane % 32 == 0:
            fx.ptr_store(o, c["ol"] + (j * VCH + v))
    gpu.barrier()
    if tid < L:
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + (tid * VCH + e))
            ss = ss + x * x
        c["put"](c["onrm"], ((req * NH + h) * L + tid) * NVCH + ch, ss)
    gpu.barrier()


@traced
def kda_norm(c, req, h, ch):
    """Rows 16 ch ..: gated RMSNorm over the head's 128 (every chunk's partial)
    -> core_out."""
    L, tid = c["L"], c["tid"]
    a = c["args"]
    if tid < L * VCH:
        j = tid // VCH
        e = tid % VCH
        vr = ch * VCH + e
        got = c["poll"](
            [(c["onrm"], ((req * NH + h) * L + j) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], (req * L + j) * NPROJ + G2_0 + h * HD + vr, 1)]
        )
        ss = sum_polled(got[:NVCH])
        r = hw_rsq(ss * fx.Float32(1.0 / HD) + fx.Float32(c["onorm_eps"]))
        y = (
            fx.ptr_load(c["ol"] + (j * VCH + e))
            * r
            * ld_bf(a["on_w"], vr)
            * sigmoid(f32_of(got[NVCH][0]))
        )
        bo.buffer_store(
            y.to(fx.BFloat16), rsrc(a["core_out"]), (req * L + j) * PROJ + h * HD + vr
        )
    gpu.barrier()


@traced
def kda_tasks(c, on, tls):
    s, L, tid, bid = c["S"], c["L"], c["tid"], c["bid"]
    for task in range(first_task(bid, s), (s // L) * NH * NVCH, BLOCKS):
        req = task // (NH * NVCH)
        h = (task // NVCH) % NH
        ch = task % NVCH
        k = kda_stage(c, req, h, ch)
        stamp(on, tls, tid, 7)
        kda_inputs(c, k, req, h)
        stamp(on, tls, tid, 8)
        kda_gates(c, k, req, h)
        stamp(on, tls, tid, 9)
        kda_recur(c, k, req, h, ch)
        stamp(on, tls, tid, 10)
        kda_norm(c, req, h, ch)
        stamp(on, tls, tid, 11)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: KdaPreBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s, L = key.tokens, key.qlen
    ns = key.nblocks + 1
    lay = scratch_layout(key)
    assert 1 <= s <= MAX_S and L <= 16 and key.nblocks >= 1
    # x blocks in LDS: all 224, or what a row of ``XL_BYTES / s`` holds
    xb_lds = min(PER_WAVE * 4 * WAVES, (XL_BYTES // s - 16) // 64)
    xrow = xb_lds * 64 + 16

    @fx.struct
    class GemvLds:
        xl: fx.Array[fx.Int32, s * xrow // 4, 16]

    @fx.struct
    class KdaLds:
        fa: fx.Array[fx.Float32, L * HD, 16]
        qkl: fx.Array[fx.Float32, L * 2 * HD, 16]
        decay: fx.Array[fx.Float32, L * HD, 16]
        vloc: fx.Array[fx.Float32, L * VCH, 16]
        ol: fx.Array[fx.Float32, L * VCH, 16]
        beta: fx.Array[fx.Float32, 16, 16]

    @fx.struct
    class GemvAcc:
        v: fx.Array[fx.Float32, 2 * WAVES * 64 * 4, 16]

    @fx.union
    class RoleLds:
        gemv: GemvLds
        acc: GemvAcc
        kda: KdaLds

    @fx.struct
    class Smem:
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    keyed = key_tuple(key, _SOURCES)
    name = (
        f"k3_mono_kda_pre_s{s}_l{L}_nb{key.nblocks}_d{int(key.delta)}"
        f"_t{int(key.timeline)}" + ("_ns" if key.stream_states else "")
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kda_pre(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        role = alloc.allocate(RoleLds)
        gl, kl, ga = role.gemv.peek(), role.kda.peek(), role.acc.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1, health, alarm, region_ids(REGIONS))
        c = {
            "S": s,
            "L": L,
            "NS": ns,
            "NB": key.nblocks,
            "SL": key.state_len,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "delta": key.delta,
            "eps": key.eps,
            "out_eps": key.out_eps,
            "onorm_eps": key.onorm_eps,
            "lb": key.lower_bound,
            "write_idx": key.write_idx,
            "st_cm": (CM_NT | CM_DEV) if key.stream_states else 0,
            "XB": xb_lds,
            "XROW": xrow,
            "put": mb.put,
            "poll": mb.poll,
            "red": ga.v.ptr,
            "red2": ga.v.ptr + WAVES * 64 * 4,
            "xl": gl.xl.ptr,
            "xl_b": fx.Int32(fx.ptrtoint(gl.xl.ptr)),
            "fa": kl.fa.ptr,
            "qkl": kl.qkl.ptr,
            "decay": kl.decay.ptr,
            "vloc": kl.vloc.ptr,
            "ol": kl.ol.ptr,
            "beta": kl.beta.ptr,
            "xbuf": scratch + fx.Int64(lay["x"][0]),
            "args": {
                "prefix": prefix,
                "delta": delta,
                "blocks": blocks,
                "blk_sm": blk_sm,
                "blk_sr": blk_sr,
                "ares_nw": ares_nw,
                "ares_qk": ares_qk,
                "in_nw": in_nw,
                "w_in": w_in,
                "w_fb": w_fb,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "on_w": on_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "st_stride": st_stride,
                "num_acc": num_acc,
                "core_out": core_out,
            },
        }
        for region in REGIONS:
            c[region] = sreg(scratch, lay[region][0], region)
        on, tls = key.timeline, lds.tls.ptr
        c["on"], c["tls"] = on, tls
        stamp_begin(on, tls, tid, TL_POINTS)
        if bid < s:
            stage_attn(c, bid)
        if bid >= s:
            gi = bid - s
            ng = BLOCKS - s
            rows_b = -(-(NPROJ - ROWS * ng) // ng)
            lr_a = ROWS * gi
            lr_b = ROWS * ng + rows_b * gi
            n_b = fx.max(fx.min(fx.Int32(rows_b), NPROJ - lr_b), 0)
            ops_a = gemv_loads(c, lr_a, ROWS, range(PER_WAVE))
            ops_b = gemv_loads(c, lr_b, n_b, range(PER_WAVE))
            wait_x(c)
            stamp(on, tls, tid, 4)
            load_x(c)
            xd = x_direct(c)
            vm_wait(0)
            gpu.barrier()
            stamp(on, tls, tid, 5)
            stage_gemv(c, lr_a, ops_a, lr_b, n_b, ops_b, xd)
            stamp(on, tls, tid, 21)
        kda_tasks(c, on, tls)
        stamp_flush(on, tls, tl, tid, bid, TL_POINTS)

    @flyc.jit
    def launch(
        prefix: Int64,
        delta: Int64,
        blocks: Int64,
        blk_sm: Int32,
        blk_sr: Int32,
        ares_nw: Int64,
        ares_qk: Int64,
        in_nw: Int64,
        w_in: Int64,
        w_fb: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        on_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        st_stride: Int32,
        num_acc: Int64,
        core_out: Int64,
        scratch: Int64,
        layer: Int32,
        epoch: Int64,
        health: Int64,
        alarm: Int64,
        tl: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kda_pre(
            prefix,
            delta,
            blocks,
            blk_sm,
            blk_sr,
            ares_nw,
            ares_qk,
            in_nw,
            w_in,
            w_fb,
            conv_w,
            conv_st,
            cs_seq,
            cs_dim,
            cs_tok,
            a_log,
            dt_bias,
            on_w,
            rstate,
            rs_seq,
            st_idx,
            st_stride,
            num_acc,
            core_out,
            scratch,
            layer,
            epoch,
            health,
            alarm,
            tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)

    ABI.check(kda_pre, launch)
    _BUILDS[key] = launch
    return launch
