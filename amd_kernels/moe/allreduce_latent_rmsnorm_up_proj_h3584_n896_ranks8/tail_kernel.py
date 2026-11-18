# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank latent tail of a Kimi-K3.1 MoE layer in one launch: all-reduce of the [rows, 3584] latent partial,
RMSNorm(3584) and the up projection of this rank's 896-column hidden shard, ``out = shared + rmsnorm(sum_r lat_r) @ W^T``.

Roles (512-thread workgroups). Workgroups 0..31 are the all-reduce, workgroup ``rb`` handles rows 8 rb..8 rb + 7 (one
wave per row) when ``rb < nrb``; the other 224 are the up projection. Both start at once: an up-projection workgroup
loads its weights, waits for the reduced latent, and runs.

All-reduce (push protocol). Rank ``me`` owns columns [448 me, 448 (me + 1)) of every row. Hop 1: it pushes slice d of its
partial into rank d's inbox slot ``me`` (write-through remote stores), then raises F1 (rb, me) at d. When F1 of all seven
peers shows this call's epoch it adds the eight slices of its slice in rank order 0..7 in fp32 (its own from
registers) and rounds once to bf16 (the latent). It also does the norm's elementwise part once for everyone: hop 2
pushes ``xn = bf16(latent * w_norm)`` into the ``lat`` buffer of all eight ranks (itself included) and the row's sum of
squares of its slice into ``lat``'s side array ``ssq[row][me]``, then adds one to each rank's counter C2 (a workgroup
without rows adds at once). Every rank's C2 grows by 256 per call (32 workgroups x 8 owners), so an up-projection
workgroup waits for C2 >= 256 e and a finished kernel has the whole reduced, scaled latent in its own ``lat`` and every
owner has finished reading its inbox, which a peer's next call writes. Epochs are private per workgroup (one word each,
read at the start and written at the end by the same workgroup): every workgroup of the fixed grid runs once per call
(an idle one bumps its epoch), and every rank makes the same calls, so all epochs equal the call count. No input
address is exchanged: eager and graph calls are the same.

Up projection. A workgroup owns 16 output columns (n tile ``nt``, 56 tiles) and ``MPG`` 16-token tiles (m group ``g``,
4 groups) over the whole K; wave w of its 8 takes the 448 K columns [448 w, 448 (w + 1)). The weights are the MFMA A
operand straight from global memory (16 rows x 4 lane groups, 16 B per lane and 32 K columns per step) and stay in registers; the B operand is ``xn`` straight from ``lat``. The eight waves'
accumulators meet in LDS, and the epilogue adds the 8 owners' sums of squares (``1 / rms`` commutes with the GEMM) and
``shared`` and stores bf16 ``out`` (which may alias ``shared``). No atomic touches an output and nothing leaves the
workgroup before the epilogue. The four groups of an n tile share an XCD (workgroup id % 8): the weights come from HBM once.

Modes: ``fused`` (above), ``ar`` (only the all-reduce, waiting for all counters) and ``up`` (only the up projection
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
    spin_load, wave_sum_lane63,
)

RB = 8  # rows per all-reduce workgroup (one wave per row)
SL = K // WORLD  # 448 columns of a slice
PK = SL // 8  # 56 packs of 8 bf16 per row slice
MAX_ROWS = 256
MAX_RB = MAX_ROWS // RB  # all-reduce workgroups
MAX_MT = MAX_ROWS // 16
NTILES = N // 16  # 56 n tiles
GROUPS = 4  # m groups
UP_WGS = NTILES * GROUPS
NWAVES = 8
KW = K // NWAVES  # K columns per wave
KSTEPS = KW // 32  # 14
XROW = KW * 2 + 16  # LDS bytes of a staged row slice: 57 packs of 16 B, so 16 rows hit 16 different bank groups
LINE = 128
F1 = 0
C2 = MAX_RB * LINE
INBOX = C2 + LINE
LL_ROW = SL // 4  # 16 B packs of an LL slice row: two payload dwords, each followed by the epoch
INBOX_BYTES = WORLD * MAX_ROWS * LL_ROW * 16
BUF_BYTES = INBOX + INBOX_BYTES
LAT_BYTES = MAX_ROWS * K * 2
SSQ_BYTES = MAX_ROWS * WORLD * 4
LATLL_ROW = K * 4  # bytes of an LL latent row: every payload dword in an 8 B pack with the epoch
LATLL_BYTES = MAX_ROWS * LATLL_ROW
SSQLL_BYTES = MAX_ROWS * WORLD * 8
LAT_ALLOC = max(LAT_BYTES + SSQ_BYTES, LATLL_BYTES + SSQLL_BYTES)  # the ``lat`` buffer: xn rows, then ssq[row][owner]
CNT_WORDS = 1024
UP_EPOCH0 = MAX_RB


def mpg_of(mt: int) -> int:
    """16-token tiles of an m group for a call of ``mt`` tiles."""
    return (mt + GROUPS - 1) // GROUPS


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(byte_addr)))


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


@launcher
def compile_tail(mode: str, mpg: int, ts: bool = False, ll: bool = False, ll2: bool = False):
    """``mpg``: 16-token tiles of an m group (rows <= 64 mpg); ``ts``: each workgroup stores ``s_memrealtime`` (10 ns) stamps
    of its phases into ``arg_ts`` ([workgroup][wave][8] words, a profiling build)."""
    assert mode in ("fused", "ar", "up") and 1 <= mpg <= MAX_MT // GROUPS
    do_ar, do_up = mode != "up", mode != "ar"
    UP0 = MAX_RB if mode == "fused" else 0  # first up-projection workgroup
    LDS = NWAVES * 16 * XROW + NWAVES * mpg * 1024
    name = f"latent_tail_h3584_n896_{mode}_mpg{mpg}" + ("_ts" if ts else "") + ("_ll" if ll else "") + ("2" if ll2 else "")
    grid = (MAX_RB if do_ar else 0) + (UP_WGS if do_up else 0)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    @flyc.kernel(name=name, known_block_size=[512, 1, 1])
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
        i32_eps: fx.Int32,
        i32_zero: fx.Int32,
        arg_ts: fx.Int64,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        bs = load_ptrs(arg_bufs)
        lats = load_ptrs(arg_bufs + WORLD * 8)
        mine = sel8(i32_me, bs)
        my_lat = sel8(i32_me, lats)
        cr = buffer_ops.create_buffer_resource_from_addr(arg_cnt, num_records_bytes=CNT_WORDS * 4)

        def at(addr, nbytes):
            return buffer_ops.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)

        def pack(rsrc, o, mask=None, cm=0):
            return Vec(buffer_ops.buffer_load(rsrc, o, vec_width=4, dtype=fx.Int32, mask=mask, cache_modifier=cm))

        def epoch(slot):
            return fx.Int32(buffer_ops.buffer_load(cr, slot, vec_width=1, dtype=fx.Int32, is_scalar=True)) + 1

        def stamp(k, wait=False):
            if const_expr(ts):
                if const_expr(wait):
                    fx.rocdl.s_waitcnt(vmcnt=0)
                t = llvm.inline_asm(T.i64, [], "s_memrealtime $0\n\ts_waitcnt lgkmcnt(0)", "=s", has_side_effects=True)
                buffer_ops.buffer_store(fx.Int64(t).to(fx.Int32), at(arg_ts, 1 << 26), (bid * NWAVES + wave) * 8 + k,
                                        mask=lane == 0)

        def count_done():
            """One add to the C2 of every rank (the first eight lanes)."""
            if (wave == 0) & (lane < WORLD):
                atomic_add(sel8(lane, bs) + C2, 1)

        def ar_role(rb):
            e = epoch(rb)
            row = rb * RB + wave
            p = lane
            ok = (lane < PK) & (row < i32_m)
            flags = fx.Int64(rb) * LINE
            peer = (wave == 0) & (lane < WORLD) & (lane != i32_me)
            stamp(0)
            if rb < i32_nrb:
                xr = at(arg_x, i32_m * i32_sx * 2)
                mine_pk = [pack(xr, row * (i32_sx // 2) + d * (SL // 2) + p * 4, ok) for d in range_constexpr(WORLD)]
                wpk = bf16x8(pack(at(arg_nw, K * 2), i32_me * (SL // 2) + p * 4, ok))
                if const_expr(ll):
                    for d in range_constexpr(WORLD):
                        dest = at(bs[d] + INBOX, INBOX_BYTES)
                        base = ((i32_me * MAX_ROWS + row) * LL_ROW + 2 * p) * 4
                        for h in range_constexpr(2):
                            buffer_ops.buffer_store(
                                Vec.from_elements([mine_pk[d][2 * h], e, mine_pk[d][2 * h + 1], e], fx.Int32), dest,
                                base + 4 * h, mask=ok & (i32_me != d), cache_modifier=SYS)
                    stamp(1)
                    inbox = at(mine + INBOX, INBOX_BYTES)
                    pk = []
                    first = []
                    for s in range_constexpr(WORLD):
                        base = ((s * MAX_ROWS + row) * LL_ROW + 2 * p) * 4
                        first.append([pack(inbox, base + 4 * h, ok & (i32_me != s), SYS) for h in range_constexpr(2)])
                    for s in range_constexpr(WORLD):
                        base = ((s * MAX_ROWS + row) * LL_ROW + 2 * p) * 4
                        a0, a1 = first[s]
                        miss = ok & (i32_me != s) & ((fx.Int32(a0[1]) != e) | (fx.Int32(a0[3]) != e)
                                                     | (fx.Int32(a1[1]) != e) | (fx.Int32(a1[3]) != e))
                        while miss:
                            llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
                            a0 = pack(inbox, base, ok, SYS)
                            a1 = pack(inbox, base + 4, ok, SYS)
                            miss = ((fx.Int32(a0[1]) != e) | (fx.Int32(a0[3]) != e)
                                    | (fx.Int32(a1[1]) != e) | (fx.Int32(a1[3]) != e))
                        got = [a0[0], a0[2], a1[0], a1[2]]
                        pk.append([(i32_me == s).select(fx.Int32(mine_pk[s][i]), fx.Int32(got[i]))
                                   for i in range_constexpr(4)])
                    stamp(2)
                    stamp(3)
                else:
                    for d in range_constexpr(WORLD):
                        dest = at(bs[d] + INBOX, WORLD * MAX_ROWS * SL * 2)
                        buffer_ops.buffer_store(mine_pk[d], dest, (i32_me * MAX_ROWS + row) * (SL // 2) + p * 4,
                                                mask=ok & (i32_me != d), cache_modifier=SYS)
                    stamp(1, True)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                    stamp(2)
                    if peer:
                        flag_store(sel8(lane, bs) + F1 + flags + fx.Int64(i32_me) * 4, e)
                        spin(mine + F1 + flags + fx.Int64(lane) * 4, e)
                    fx.gpu.barrier()
                    stamp(3)
                    inbox = at(mine + INBOX, WORLD * MAX_ROWS * SL * 2)
                    pk = []
                    for s in range_constexpr(WORLD):
                        got = pack(inbox, (s * MAX_ROWS + row) * (SL // 2) + p * 4, ok & (i32_me != s), SYS)
                        pk.append([(i32_me == s).select(fx.Int32(mine_pk[s][i]), fx.Int32(got[i]))
                                   for i in range_constexpr(4)])
                lat_pk, xn_pk = [], []
                ssq = fx.Float32(0.0)
                for i in range_constexpr(4):
                    lo = (fx.Int32(pk[0][i]) << 16).bitcast(fx.Float32)
                    hi = (fx.Int32(pk[0][i]) & -65536).bitcast(fx.Float32)
                    for s in range_constexpr(1, WORLD):
                        lo = lo + (fx.Int32(pk[s][i]) << 16).bitcast(fx.Float32)
                        hi = hi + (fx.Int32(pk[s][i]) & -65536).bitcast(fx.Float32)
                    q = pk_bf16(lo, hi)  # the latent, rounded once
                    lo = (q << 16).bitcast(fx.Float32)
                    hi = (q & -65536).bitcast(fx.Float32)
                    ssq = ssq + lo * lo + hi * hi
                    xn_pk.append(pk_bf16(lo * wpk[2 * i], hi * wpk[2 * i + 1]))
                stamp(4, True)
                res = Vec.from_elements(xn_pk, fx.Int32)
                ssq = wave_sum_lane63(ssq)
                if const_expr(ll2):
                    for d in range_constexpr(WORLD):
                        lat = at(lats[d], LAT_ALLOC)
                        base = row * (LATLL_ROW // 4) + (i32_me * (SL // 4) + 2 * p) * 4
                        for h in range_constexpr(2):
                            buffer_ops.buffer_store(
                                Vec.from_elements([xn_pk[2 * h], e, xn_pk[2 * h + 1], e], fx.Int32), lat, base + 4 * h,
                                mask=ok, cache_modifier=SYS)
                        buffer_ops.buffer_store(
                            Vec.from_elements([ssq.bitcast(fx.Int32), e], fx.Int32), lat,
                            LATLL_BYTES // 4 + (row * WORLD + i32_me) * 2, mask=(lane == 63) & (row < i32_m),
                            cache_modifier=SYS)
                else:
                    for d in range_constexpr(WORLD):
                        lat = at(lats[d], LAT_ALLOC)
                        buffer_ops.buffer_store(res, lat, row * (K // 2) + i32_me * (SL // 2) + p * 4, mask=ok,
                                                cache_modifier=SYS)
                        buffer_ops.buffer_store(ssq, lat, (LAT_BYTES // 4) + row * WORLD + i32_me,
                                                mask=(lane == 63) & (row < i32_m), cache_modifier=SYS)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fx.gpu.barrier()
                stamp(5)
            if const_expr(not ll2):
                count_done()
                if const_expr(mode == "ar"):
                    if tx == 0:
                        spin_load(mine + C2, e * (MAX_RB * WORLD))
                    fx.gpu.barrier()
            buffer_ops.buffer_store(e, cr, rb, mask=tx == 0)
            stamp(6, True)

        def up_role(bid2):
            lds = fx.Int32(fx.ptrtoint(fx.SharedAllocator().allocate(Shared).peek().buf.ptr))
            g = lane // 16
            l16 = lane % 16
            xcd = bid2 % 8
            r8 = bid2 // 8
            nt = xcd * (NTILES // 8) + r8 // GROUPS
            mg = r8 % GROUPS
            n0 = nt * 16
            kb = wave * KW + g * 8  # K column of this lane's pack at step 0; step s adds 32 s
            e2 = epoch(UP_EPOCH0 + bid2)
            m0 = mg * mpg * 16

            def body():
                stamp(0)
                wr = at(arg_w, N * K * 2)
                a_op = [pack(wr, ((n0 + l16) * K + kb + 32 * s) // 2, None, NT) for s in range_constexpr(KSTEPS)]
                latr = at(my_lat, LATLL_BYTES if ll2 else LAT_BYTES)
                ssr = at(my_lat + (LATLL_BYTES if ll2 else LAT_BYTES), SSQLL_BYTES if ll2 else SSQ_BYTES)
                shr = at(arg_shared, i32_m * (i32_ssh * 2))
                outr = at(arg_out, i32_m * (i32_sout * 2))
                eps = i32_eps.bitcast(fx.Float32)
                zero = fx.Int32(i32_zero).bitcast(fx.Float32)
                stamp(1, True)
                if const_expr(ll2):
                    if const_expr(mode == "fused"):
                        if lane == 0:  # the last pack this wave's owner stores of the last row: the rest follows
                            spin_load(my_lat + (((i32_m - 1) * (LATLL_ROW // 4) + (wave * (SL // 4) + SL // 4 - 1) * 4
                                                 + 3) * 4).to(fx.Int64), e2)
                elif const_expr(mode == "fused"):
                    if tx == 0:
                        spin_load(mine + C2, e2 * (MAX_RB * WORLD))
                    fx.gpu.barrier()
                stamp(2)

                red = lds + NWAVES * 16 * XROW  # the waves' accumulators meet here
                stage = lds + wave * (16 * XROW)  # this wave's x rows, private

                def load_x(tt):
                    """16 token rows of this wave's 896 B K slice, one coalesced 16 B per lane (lanes 0..55)."""
                    out = []
                    for r in range_constexpr(16):
                        rr = m0 + tt * 16 + r
                        rr = (rr < i32_m).select(rr, i32_m - 1)
                        out.append(pack(latr, (rr * K + wave * KW) // 2 + lane * 4, lane < PK))
                    return out

                def load_x_ll(tt):
                    """The same rows from the LL buffer: two 16 B packs per lane, each re-read until both its epochs
                    match."""
                    first = []
                    for r in range_constexpr(16):
                        rr = m0 + tt * 16 + r
                        rr = (rr < i32_m).select(rr, i32_m - 1)
                        base = rr * (LATLL_ROW // 4) + (wave * (SL // 4) + 2 * lane) * 4
                        first.append((base, pack(latr, base, lane < PK, SYS), pack(latr, base + 4, lane < PK, SYS)))
                    out = []
                    for base, a0, a1 in first:
                        miss = (lane < PK) & ((fx.Int32(a0[1]) != e2) | (fx.Int32(a0[3]) != e2)
                                              | (fx.Int32(a1[1]) != e2) | (fx.Int32(a1[3]) != e2))
                        while miss:
                            llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
                            a0 = pack(latr, base, lane < PK, SYS)
                            a1 = pack(latr, base + 4, lane < PK, SYS)
                            miss = ((fx.Int32(a0[1]) != e2) | (fx.Int32(a0[3]) != e2)
                                    | (fx.Int32(a1[1]) != e2) | (fx.Int32(a1[3]) != e2))
                        out.append(Vec.from_elements([a0[0], a0[2], a1[0], a1[2]], fx.Int32))
                    return out

                def put_x(v):
                    if lane < PK:
                        for r in range_constexpr(16):
                            llvm.StoreOp(fx.as_ir_value(v[r]), _lds_ptr(stage + r * XROW + lane * 16), alignment=16)

                nxt = None if ll2 else load_x(0)
                ep = tx < mpg * 64
                trow = m0 + (tx // 64) * 16 + (tx % 16)
                tcol = n0 + ((tx % 64) // 16) * 4
                if const_expr(ll2):
                    sqs = []
                    for i in range_constexpr(4):
                        v = pack(ssr, (trow * WORLD + 2 * i) * 2, ep, SYS)
                        miss = ep & ((fx.Int32(v[1]) != e2) | (fx.Int32(v[3]) != e2))
                        while miss:
                            llvm.inline_asm(None, [], "", "~{memory}", has_side_effects=True)
                            v = pack(ssr, (trow * WORLD + 2 * i) * 2, ep, SYS)
                            miss = ep & ((fx.Int32(v[1]) != e2) | (fx.Int32(v[3]) != e2))
                        sqs += [fx.Int32(v[0]).bitcast(fx.Float32), fx.Int32(v[2]).bitcast(fx.Float32)]
                else:
                    sq0 = Vec(buffer_ops.buffer_load(ssr, trow * WORLD, vec_width=4, dtype=fx.Float32, mask=ep,
                                                     cache_modifier=SYS))
                    sq1 = Vec(buffer_ops.buffer_load(ssr, trow * WORLD + 4, vec_width=4, dtype=fx.Float32, mask=ep,
                                                     cache_modifier=SYS))
                    sqs = [sq0[0], sq0[1], sq0[2], sq0[3], sq1[0], sq1[1], sq1[2], sq1[3]]
                sh = Vec(buffer_ops.buffer_load(shr, trow * (i32_ssh // 2) + tcol // 2, vec_width=2, dtype=fx.Int32,
                                                mask=ep))
                stamp(3, True)
                for tt in range_constexpr(mpg):
                    if const_expr(ll2):
                        cur = load_x_ll(tt)
                    else:
                        cur = nxt
                        if tt + 1 < mpg:
                            nxt = load_x(tt + 1)
                    put_x(cur)
                    acc = Vec.filled(4, zero, fx.Float32)
                    for s in range_constexpr(KSTEPS):
                        b = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(stage + l16 * XROW + 64 * s + 16 * g),
                                          alignment=16))
                        acc = mfma(a_op[s], b, acc)
                    mfma_wait()
                    llvm.StoreOp(fx.as_ir_value(acc), _lds_ptr(red + ((wave * mpg + tt) * 64 + lane) * 16),
                                 alignment=16)
                stamp(4)
                fx.gpu.barrier()
                stamp(5)
                if ep:
                    tt = tx // 64
                    ln = tx % 64
                    tot = Vec(llvm.load(T.vec(4, T.f32), _lds_ptr(red + (tt * 64 + ln) * 16), alignment=16))
                    for w2 in range_constexpr(1, NWAVES):
                        tot = tot + Vec(llvm.load(T.vec(4, T.f32), _lds_ptr(red + ((w2 * mpg + tt) * 64 + ln) * 16),
                                                  alignment=16))
                    sq = ((sqs[0] + sqs[1]) + (sqs[2] + sqs[3])) + ((sqs[4] + sqs[5]) + (sqs[6] + sqs[7]))
                    inv = rsq(sq * fx.Float32(1.0 / K) + eps)
                    shf = bf16x8(Vec.from_elements([sh[0], sh[1], fx.Int32(0), fx.Int32(0)], fx.Int32))
                    o = [tot[i] * inv + shf[i] for i in range_constexpr(4)]
                    ov = Vec.from_elements([pk_bf16(o[0], o[1]), pk_bf16(o[2], o[3])], fx.Int32)
                    buffer_ops.buffer_store(ov, outr, trow * (i32_sout // 2) + tcol // 2, mask=trow < i32_m)
                stamp(6, True)

            if m0 < i32_m:
                body()
            buffer_ops.buffer_store(e2, cr, UP_EPOCH0 + bid2, mask=tx == 0)

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
        i32_ssh: fx.Int32, arg_out: fx.Int64, i32_sout: fx.Int32, i32_eps: fx.Int32, i32_zero: fx.Int32, arg_ts: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(i32_me, arg_bufs, arg_cnt, arg_x, i32_sx, i32_m, i32_nrb, arg_nw, arg_w, arg_shared, i32_ssh,
               arg_out, i32_sout, i32_eps, i32_zero, arg_ts).launch(grid=(grid, 1, 1), block=(512, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
