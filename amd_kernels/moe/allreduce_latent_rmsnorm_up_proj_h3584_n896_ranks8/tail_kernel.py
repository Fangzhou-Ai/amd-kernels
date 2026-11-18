# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank latent tail of a Kimi-K3.1 MoE layer in one launch: all-reduce of the [rows, 3584] latent partial,
RMSNorm(3584) and the up projection of this rank's 896-column hidden shard, ``out = shared + rmsnorm(sum_r lat_r) @ W^T``.

Roles. Workgroups 0..63 are the all-reduce (workgroup ``rb`` handles rows 4 rb..4 rb + 3 when ``rb < nrb``); the rest
are the up projection, (``ng``, ``c``) tiles of 16 NJ columns x 128 NS K columns. Both roles start at once: the
up-projection workgroups load their weights, wait for the reduced latent they need, and run.

All-reduce (push protocol). Rank ``me`` owns columns [448 me, 448 (me + 1)) of every row. Hop 1: it pushes slice d of its
partial into rank d's inbox slot ``me`` (write-through remote stores), then raises F1 (rb, me) at d. When F1 of all seven
peers shows this call's epoch it adds the eight slices of its slice in rank order 0..7 in fp32 (its own from
registers), rounds once to bf16 and hop 2 pushes the result into the ``lat`` buffer of all eight ranks (itself
included), then raises F2 (rb, me) there. A rank's up-projection workgroup (ng, c) leaves its wait when F2 (rb, o) of
every row block and of the owners o of its K columns shows the epoch, so a finished kernel has the whole reduced latent
in its own ``lat`` and every owner has finished reading its inbox, which a peer's next call writes. Epochs are private
per workgroup (one word each, read at the start and written at the end by the same workgroup): every workgroup runs
once per call and every rank makes the same calls, so all epochs equal the call count. No input address is exchanged:
eager and graph calls are the same.

Up projection. A workgroup (4 waves) owns 16 NJ output columns (``ng``) and one K chunk ``c`` of 128 NS columns; wave
w takes the 32 NS columns [c * 128 NS + w * 32 NS, +32 NS) of it, so no wave shares a K column. The weights are the MFMA
A operand straight from global memory (16 rows x 4 lane groups, one 16 B pack per lane and K step; a lane takes NS
contiguous packs, K is permuted identically in both operands), kept in registers while the latent rows stream by: per
16-token tile the B operand is ``bf16(x * w_norm)`` (the per-row 1 / rms commutes with the GEMM and is applied in the
epilogue). Per tile the four waves' accumulators and row sums of squares meet in LDS (double buffered) and the sum goes
to a fp32 partial slab. When all chunks of a column group have stored their slabs (a ticket counter that counts to 32 per
call: chunk 0 adds 33 - SK, the others 1, so it needs no reset; the workgroups of a group are co-resident) tile m is
finished by chunk m % SK: the partials are added in chunk order, scaled by 1 / rms, added to ``shared`` and stored as
bf16 ``out`` (which may alias ``shared``). No atomic touches an output. The workgroups of one chunk share an XCD
(workgroup id % 8), so a latent chunk is read from HBM once per XCD and from L2 by its other column groups.

Modes: ``fused`` (above), ``ar`` (only the all-reduce, waiting for all peers' hop 2) and ``up`` (only the up projection
on a ``lat`` filled by an earlier launch).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .norm_math import (
    K, N, NT, PRELOAD_HINTS, SYS, WORLD, atomic_add, bf16x8, flag_store, load_ptrs, pk_bf16, rsq, sel8, spin,
)

RB = 4  # rows per all-reduce workgroup
SL = K // WORLD  # 448 columns of a slice
PK = SL // 8  # 56 packs of 8 bf16 per row slice
NTHR = RB * PK  # 224 active threads
MAX_ROWS = 256
MAX_RB = MAX_ROWS // RB
MAX_MT = MAX_ROWS // 16
LINE = 128
F1 = 0
F2 = MAX_RB * LINE
INBOX = 2 * MAX_RB * LINE
BUF_BYTES = INBOX + WORLD * MAX_ROWS * SL * 2
EPOCH = 32  # ticket cycle (>= every SK)
CNT_WORDS = 1024


def split_k(ns: int) -> int:
    return K // (4 * 32 * ns)


def groups(nj: int) -> int:
    return N // (16 * nj)


def up_grid(ns: int, nj: int) -> int:
    return 8 * groups(nj) * ((split_k(ns) + 7) // 8)


def mfma(a, b, c):
    """``c += a x b`` (v_mfma_f32_16x16x32_bf16) as inline asm with the destination tied to ``c``: the allocator gives
    the intrinsic form's destination the registers of a dead B or of a shared zero C, which the 8-pass MFMA still
    reads while it writes. The leading nops cover the VALU -> MFMA operand hazard the compiler cannot see in an asm."""
    r = llvm.inline_asm(
        T.vec(4, T.f32), [fx.as_ir_value(a), fx.as_ir_value(b), fx.as_ir_value(c)],
        "s_nop 4\n\tv_mfma_f32_16x16x32_bf16 $0, $1, $2, $0", "=v,v,v,0", has_side_effects=True)
    return Vec(r)


def mfma_wait():
    """Wait states between the last MFMA of a group and the first read of its accumulators."""
    llvm.inline_asm(None, [], "s_nop 7\n\ts_nop 7", "~{memory}", has_side_effects=True)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(byte_addr)))


@launcher
def compile_tail(mode: str, mt: int, ns: int, nj: int, pd: int):
    """``mt``: 16-token tiles (rows <= 16 mt); ``ns``: K steps of 32 per wave and chunk (1, 2 or 4); ``nj``: 16-column
    tiles per workgroup (1, 2 or 4); ``pd``: tiles of the latent loaded ahead."""
    assert mode in ("fused", "ar", "up") and ns in (1, 2, 4) and nj in (1, 2, 4) and 1 <= mt <= MAX_MT
    do_ar, do_up = mode != "up", mode != "ar"
    SK = split_k(ns)
    NG = groups(nj)
    KW = 32 * ns  # K columns per wave and chunk
    KC = 4 * KW  # per workgroup
    RED = 4 * nj * 64 * 16  # accumulator exchange, bytes
    BUF = RED + 16 * 16 * 4
    LDS = 2 * BUF + 16
    UP0 = MAX_RB if mode == "fused" else 0  # first up-projection workgroup
    name = f"latent_tail_h3584_n896_{mode}_mt{mt}_ns{ns}_nj{nj}_pd{pd}"
    grid = (MAX_RB if do_ar else 0) + (up_grid(ns, nj) if do_up else 0)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        i32_me: fx.Int32,
        arg_bufs: fx.Int64,
        arg_cnt: fx.Int64,
        arg_x: fx.Int64,
        i32_sx: fx.Int32,
        i32_m: fx.Int32,
        i32_nrb: fx.Int32,
        arg_nw: fx.Int64,
        arg_w: fx.Int64,
        arg_shared: fx.Int64,
        i32_ssh: fx.Int32,
        arg_out: fx.Int64,
        i32_sout: fx.Int32,
        arg_part: fx.Int64,
        arg_ss: fx.Int64,
        arg_tkt: fx.Int64,
        i32_eps: fx.Int32,
        i32_zero: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        bs = load_ptrs(arg_bufs)
        lats = load_ptrs(arg_bufs + WORLD * 8)
        mine = sel8(i32_me, bs)
        bs_lat = sel8(i32_me, lats)
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=CNT_WORDS * 4)

        def at(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)

        def pack(rsrc, o, mask=None, cm=0):
            return Vec(buffer_ops.buffer_load(rsrc, o, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=cm))

        def epoch(slot):
            return fx.Int32(buffer_ops.buffer_load(cr, slot, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1

        def ar_role(rb):
            e = epoch(rb)
            row = rb * RB + tx // PK
            p = tx % PK
            ok = (tx < NTHR) & (row < i32_m)
            peer = (wave == 0) & (lane < WORLD) & (lane != i32_me)
            flags = fx.Int64(rb) * LINE
            if rb < i32_nrb:
                xr = at(arg_x, i32_m * i32_sx * 2)
                mine_pk = [pack(xr, row * (i32_sx // 2) + d * (SL // 2) + p * 4, ok) for d in range_constexpr(WORLD)]
                for d in range_constexpr(WORLD):
                    dest = at(bs[d] + INBOX, WORLD * MAX_ROWS * SL * 2)
                    buffer_ops.buffer_store(mine_pk[d], dest, (i32_me * MAX_ROWS + row) * (SL // 2) + p * 4,
                                            mask=ok & (i32_me != d), cache_modifier=SYS)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if peer:
                    flag_store(sel8(lane, bs) + F1 + flags + fx.Int64(i32_me) * 4, e)
                    spin(mine + F1 + flags + fx.Int64(lane) * 4, e)
                fx.gpu.barrier()

                inbox = at(mine + INBOX, WORLD * MAX_ROWS * SL * 2)
                pk = []
                for s in range_constexpr(WORLD):
                    got = pack(inbox, (s * MAX_ROWS + row) * (SL // 2) + p * 4, ok & (i32_me != s), SYS)
                    pk.append([(i32_me == s).select(fx.Int32(mine_pk[s][i]), fx.Int32(got[i]))
                               for i in range_constexpr(4)])
                out = []
                for i in range_constexpr(4):
                    lo = (fx.Int32(pk[0][i]) << 16).bitcast(fx.Float32)
                    hi = (fx.Int32(pk[0][i]) & -65536).bitcast(fx.Float32)
                    for s in range_constexpr(1, WORLD):
                        lo = lo + (fx.Int32(pk[s][i]) << 16).bitcast(fx.Float32)
                        hi = hi + (fx.Int32(pk[s][i]) & -65536).bitcast(fx.Float32)
                    out.append(pk_bf16(lo, hi))
                res = Vec.from_elements(out, fx.Int32)
                for d in range_constexpr(WORLD):
                    lat = at(lats[d], MAX_ROWS * K * 2)
                    buffer_ops.buffer_store(res, lat, row * (K // 2) + i32_me * (SL // 2) + p * 4, mask=ok,
                                            cache_modifier=SYS)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if (wave == 0) & (lane < WORLD):
                    flag_store(sel8(lane, bs) + F2 + flags + fx.Int64(i32_me) * 4, e)
                if const_expr(mode == "ar"):
                    if peer:
                        spin(mine + F2 + flags + fx.Int64(lane) * 4, e)
                    fx.gpu.barrier()
            buffer_ops.buffer_store(e, cr, rb, mask=tx == 0)

        def up_role(bid2):
            lds = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(Shared).peek().buf.ptr))
            g = lane // 16
            l16 = lane % 16
            rr = bid2 // 8
            c = bid2 % 8 + 8 * (rr // NG)
            ng = rr % NG
            n0 = ng * (16 * nj)
            kb = c * KC + wave * KW + g * (8 * ns)  # first K column of this lane's packs

            def body():
                wr = at(arg_w, N * K * 2)
                a_op = [[pack(wr, ((n0 + 16 * j + l16) * K + kb + 8 * s) // 2, None, NT)
                         for s in range_constexpr(ns)] for j in range_constexpr(nj)]
                wnr = at(arg_nw, K * 2)
                nw = []
                for s in range_constexpr(ns):
                    nw += bf16x8(pack(wnr, (kb + 8 * s) // 2))
                latr = at(bs_lat, i32_m * (K * 2))
                shr = at(arg_shared, i32_m * (i32_ssh * 2))
                outr = at(arg_out, i32_m * (i32_sout * 2))
                part = at(arg_part, SK * MAX_ROWS * N * 4)
                ssr = at(arg_ss, NG * SK * MAX_ROWS * 4)
                eps = i32_eps.bitcast(fx.Float32)
                zero = fx.Int32(i32_zero).bitcast(fx.Float32)
                ticket = arg_tkt + fx.Int64(ng) * 128
                inc = (c == 0).select(fx.Int32(EPOCH + 1 - SK), fx.Int32(1))
                if const_expr(mode == "fused"):
                    if wave == 0:
                        o0 = (c * KC) // SL
                        o1 = (c * KC + KC - 1) // SL
                        if lane < i32_nrb:
                            spin(mine + F2 + fx.Int64(lane) * LINE + fx.Int64(o0) * 4, e2)
                            if o1 != o0:
                                spin(mine + F2 + fx.Int64(lane) * LINE + fx.Int64(o1) * 4, e2)
                    fx.gpu.barrier()

                def load_b(m):
                    r = m * 16 + l16
                    r = (r < i32_m).select(r, i32_m - 1)
                    return [pack(latr, (r * K + kb + 8 * s) // 2) for s in range_constexpr(ns)]

                pending = [load_b(m) for m in range_constexpr(min(pd, mt))]
                for m in range_constexpr(mt):
                    cur = pending[m]
                    if m + pd < mt:
                        pending.append(load_b(m + pd))
                    acc = [Vec.filled(4, zero, fx.Float32) for _ in range_constexpr(nj)]
                    ss = fx.Float32(0.0)
                    for s in range_constexpr(ns):
                        x = bf16x8(cur[s])
                        for i in range_constexpr(8):
                            ss = ss + x[i] * x[i]
                        bw = Vec.from_elements(
                            [pk_bf16(x[2 * i] * nw[8 * s + 2 * i], x[2 * i + 1] * nw[8 * s + 2 * i + 1])
                             for i in range_constexpr(4)], fx.Int32)
                        for j in range_constexpr(nj):
                            acc[j] = mfma(a_op[j][s], bw, acc[j])
                    mfma_wait()
                    red = lds + (m % 2) * BUF
                    for j in range_constexpr(nj):
                        llvm.StoreOp(fx.as_ir_value(acc[j]), _lds_ptr(red + ((wave * nj + j) * 64 + lane) * 16),
                                     alignment=16)
                    llvm.StoreOp(fx.as_ir_value(ss), _lds_ptr(red + RED + (l16 * 16 + wave * 4 + g) * 4),
                                 alignment=4)
                    fx.gpu.barrier()
                    # thread t sums the 4 waves' float2 of elements e = (j, lane, half), e = t + 256 it
                    for it in range_constexpr(max(1, nj // 2)):
                        e = tx + 256 * it
                        j = e // 128
                        ln = (e // 2) % 64
                        off = ((j * 64 + ln) * 16) + (e % 2) * 8
                        if j < nj:
                            tot = Vec(llvm.load(T.vec(2, T.f32), _lds_ptr(red + off), alignment=8))
                            for w2 in range_constexpr(1, 4):
                                tot = tot + Vec(llvm.load(T.vec(2, T.f32), _lds_ptr(red + w2 * nj * 1024 + off),
                                                          alignment=8))
                            prow = m * 16 + ln % 16
                            pcol = n0 + j * 16 + (ln // 16) * 4 + (e % 2) * 2
                            buffer_ops.buffer_store(tot, part, (c * MAX_ROWS + prow) * N + pcol,
                                                    cache_modifier=SYS)
                    if (wave == 0) & (g == 0):
                        q = [Vec(llvm.load(T.vec(4, T.f32), _lds_ptr(red + RED + (l16 * 16 + 4 * i) * 4),
                                           alignment=16)) for i in range_constexpr(4)]
                        v = (q[0] + q[1]) + (q[2] + q[3])
                        sst = (v[0] + v[1]) + (v[2] + v[3])
                        buffer_ops.buffer_store(sst, ssr, (ng * SK + c) * MAX_ROWS + m * 16 + l16,
                                                cache_modifier=SYS)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx == 0:
                    old = atomic_add(ticket, inc)
                    spin(ticket, old - (old & (EPOCH - 1)) + EPOCH)
                fx.gpu.barrier()

                def finish(m):
                    frow = m * 16 + tx // (4 * nj)
                    col = n0 + (tx % (4 * nj)) * 4
                    sums = [Vec(buffer_ops.buffer_load(part, (c2 * MAX_ROWS + frow) * N + col, vec_width=4,
                                                       dtype=fx.Float32, cache_modifier=SYS))
                            for c2 in range_constexpr(SK)]
                    sqs = [fx.Float32(buffer_ops.buffer_load(ssr, (ng * SK + c2) * MAX_ROWS + frow, vec_width=1,
                                                             dtype=fx.Float32, cache_modifier=SYS))
                           for c2 in range_constexpr(SK)]
                    sh = Vec(buffer_ops.buffer_load(shr, frow * (i32_ssh // 2) + col // 2, vec_width=2,
                                                    dtype=fx.Int32))
                    tot, sq = sums[0], sqs[0]
                    for c2 in range_constexpr(1, SK):
                        tot = tot + sums[c2]
                        sq = sq + sqs[c2]
                    inv = rsq(sq * fx.Float32(1.0 / K) + eps)
                    shf = bf16x8(Vec.from_elements([sh[0], sh[1], fx.Int32(0), fx.Int32(0)], fx.Int32))
                    res = [tot[i] * inv + shf[i] for i in range_constexpr(4)]
                    o = Vec.from_elements([pk_bf16(res[0], res[1]), pk_bf16(res[2], res[3])], fx.Int32)
                    buffer_ops.buffer_store(o, outr, frow * (i32_sout // 2) + col // 2, mask=frow < i32_m)

                for m in range_constexpr(mt):
                    if (c == m % SK) & (tx < 64 * nj):
                        finish(m)

            if const_expr(mode == "fused"):
                e2 = epoch(MAX_RB + bid2)
            if c < SK:
                body()
            if const_expr(mode == "fused"):
                buffer_ops.buffer_store(e2, cr, MAX_RB + bid2, mask=tx == 0)

        if const_expr(do_ar):
            if bid < MAX_RB:
                ar_role(bid)
        if const_expr(do_up):
            if bid >= UP0:
                up_role(bid - UP0)

    @flyc.jit
    def launch(
        i32_m: fx.Int32, i32_nrb: fx.Int32, i32_me: fx.Int32, arg_bufs: fx.Int64, arg_cnt: fx.Int64,
        arg_x: fx.Int64, i32_sx: fx.Int32, arg_nw: fx.Int64, arg_w: fx.Int64, arg_shared: fx.Int64,
        i32_ssh: fx.Int32, arg_out: fx.Int64, i32_sout: fx.Int32, arg_part: fx.Int64, arg_ss: fx.Int64,
        arg_tkt: fx.Int64, i32_eps: fx.Int32, i32_zero: fx.Int32, stream: fx.Stream,
    ):
        kernel(i32_me, arg_bufs, arg_cnt, arg_x, i32_sx, i32_m, i32_nrb, arg_nw, arg_w, arg_shared, i32_ssh,
               arg_out, i32_sout, arg_part, arg_ss, arg_tkt, i32_eps, i32_zero).launch(
            grid=(grid, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
