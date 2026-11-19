# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""16-row MXFP4 x MXFP4 down GEMM with the router weight applied once and a deterministic top-16 sum (no atomics into
outputs).

One workgroup = ``bm`` sorted rows x ``tn`` output columns (4 waves, ``tn / 64`` n16 tiles each). W2 is the aiter
preshuffled [E, 224 n16 tiles, 3 k128, 4, 16, 16 B]; its scales e8m0_shuffle'd [E * 3584, 12 -> 16]. The MFMA is
transposed (weights as the A operand), so a lane holds 4 consecutive columns of one row and stores them directly.

Rows go to the slot buffer in sorted-row order (a workgroup writes 16 consecutive rows). With ``reduce`` the 16 rows'
tickets (one counter per token and column tile, 16 arrivals) pick the workgroup that stored a token's last slot; it sums
that token's 16 slots in slot order (fp32) into the BF16 output and resets the counter. Counters only order work.
"""
import functools
import flydsl.compiler as flyc
from amd_kernels.warm_up import launcher
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import E, H, I, TOPK, hold, lds_ptr, mfma, raw
from .sync import READY_STRIDE, SC1, atomic_add, atomic_reset, poll_first, wait_count

KS = I // 128  # 3 k128 steps
W_EXPERT = H * I // 2
W_TILE = KS * 1024
KBLK = 2  # e8m0_shuffle pads the 12 scale columns to 16 = 2 k-blocks of 8
NB1 = 12  # GEMM1 tiles per m-block (kw 4): the ready count a GEMM2 tile waits for
TICKET_STRIDE = 64  # bytes between two tickets: each in its own line


@functools.cache
def compile_gemm2_tile(*, tn=256, bm=16, pf=2, wait_for=0, reduce=False, abl=0):
    """The tile function of GEMM2 (256 threads; ``base`` = 128 B of LDS for ``reduce``) and its column-tile count.

    ``wait_for`` > 0: the tile first streams its weights, then waits until ``ready`` (counters, ``READY_STRIDE`` bytes
    per m-block) shows ``wait_for`` GEMM1 tiles done, then reads the intermediate rows."""
    assert tn in (64, 128, 256, 512) and bm in (16, 32) and H % tn == 0 and (not reduce or tn == 256)
    ni, mr = tn // 64, bm // 16
    nblk = H // tn
    skw = {"cache_modifier": SC1} if reduce else {}

    @flyc.jit
    def tile(base, rb, A, AS, W, WS, SL, OUT, IDS, WT, EIDS, NV, REV, m, rows, ready, tick):
        tx = fx.thread_idx.x
        lane, wave = tx % 64, tx // 64
        l16, q16 = lane % 16, lane // 16
        mb, nb = rb // nblk, rb % nblk
        nr = fx.Int32(NV[0])
        if mb * bm < nr:
            expert = fx.Int32(EIDS[mb])
            ar = buffer_ops.create_buffer_resource(A, max_size=False, num_records_bytes=rows * (I // 2))
            sr = buffer_ops.create_buffer_resource(AS, max_size=False, num_records_bytes=rows * (I // 32))
            wr = buffer_ops.create_buffer_resource(W, max_size=False, num_records_bytes=W_EXPERT,
                                                   base_byte_offset=fx.Int64(expert) * W_EXPERT)
            sw = buffer_ops.create_buffer_resource(WS, max_size=False, num_records_bytes=E * (H // 32) * KBLK * 256)
            slr = buffer_ops.create_buffer_resource(SL, max_size=False, num_records_bytes=rows * H * 2)
            ad = [((mb * bm + mi * 16 + l16) * (I // 2) + q16 * 16) // 4 for mi in range_constexpr(mr)]
            sd = [(mb * bm + mi * 16 + l16) * (I // 32) // 4 for mi in range_constexpr(mr)]
            wd = [((nb * (tn // 16) + wave * ni + j) * W_TILE + q16 * 256 + l16 * 16) // 4 for j in range_constexpr(ni)]
            swd = [((expert * (H // 32) + (nb * (tn // 16) + wave * ni + j) // 2) * KBLK) * 64 + q16 * 16 + l16
                   for j in range_constexpr(ni)]
            pk_ids = [fx.Int32(IDS[mb * bm + mi * 16 + l16]) for mi in range_constexpr(mr)]
            pk_w = [fx.Float32(WT[mb * bm + mi * 16 + l16]) for mi in range_constexpr(mr)]

            def load_w(k):
                return [buffer_ops.buffer_load(wr, wd[j], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 1024,
                                               cache_modifier=2) for j in range_constexpr(ni)]

            # one dword per (n32 group, k-block) serves two n16 tiles and two k steps
            sbd = [[(buffer_ops.buffer_load(sw, swd[2 * gi], vec_width=1, dtype=fx.Int32, soffset_bytes=kb * 256)
                     if abl & 1 == 0 else fx.Int32(0x7F7F7F7F))
                    for kb in range_constexpr(2)] for gi in range_constexpr(ni // 2)]

            def load_a(k):
                aa = [(buffer_ops.buffer_load(ar, ad[mi], vec_width=4, dtype=fx.Int32, soffset_bytes=k * 64)
                       if abl & 4 == 0 else Vec.from_elements([fx.Int32(1)] * 4, fx.Int32))
                      for mi in range_constexpr(mr)]
                sa = [(fx.Int32(buffer_ops.buffer_load(sr, sd[mi], vec_width=1, dtype=fx.Int32,
                                                    soffset_bytes=k * 4)) >> (q16 * 8)) & 255
                      for mi in range_constexpr(mr)]
                return aa, sa

            acc = [[None] * ni for _ in range(mr)]
            if wait_for > 0:
                ready_addr = ready + fx.Int64(mb * READY_STRIDE)
                got0 = poll_first(ready_addr)
                wl = [load_w(k) for k in range_constexpr(KS)]
                wait_count(ready_addr, wait_for, nblk, got0)
                al = [load_a(k) for k in range_constexpr(KS)]
                for k in range_constexpr(KS):
                    for mi in range_constexpr(mr):
                        for j in range_constexpr(ni):
                            acc[mi][j] = mfma(al[k][0][mi], wl[k][j], acc[mi][j], al[k][1][mi], sbd[j // 2][k // 2],
                                              (k % 2) * 2 + j % 2, True)
            else:
                ring = [(load_a(k), load_w(k)) for k in range_constexpr(min(pf, KS))]
                for k in range_constexpr(KS):
                    if fx.const_expr(k + pf < KS):
                        ring.append((load_a(k + pf), load_w(k + pf)))
                    (aa, sa), bb = ring.pop(0)
                    for mi in range_constexpr(mr):
                        for j in range_constexpr(ni):
                            acc[mi][j] = mfma(aa[mi], bb[j], acc[mi][j], sa[mi], sbd[j // 2][k // 2],
                                              (k % 2) * 2 + j % 2, True)
            flat = hold([acc[mi][j] for mi in range(mr) for j in range(ni)])
            for mi in range_constexpr(mr):
                token = pk_ids[mi] & 0xFFFFFF
                for j in range_constexpr(ni):
                    v = Vec(flat[mi * ni + j]) * pk_w[mi]
                    lo = Vec.from_elements([v[0], v[1]], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    hi = Vec.from_elements([v[2], v[3]], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                    col = nb * tn + (wave * ni + j) * 16 + q16 * 4
                    buffer_ops.buffer_store(Vec.from_elements([lo, hi], fx.Int32), slr,
                                            ((mb * bm + mi * 16 + l16) * H + col) // 2,
                                            mask=((token < m) & (m < 0)) if abl & 2 else token < m, **skw)
            if reduce:
                outr = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=m * H * 2)
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if tx < 16 * mr:
                    row_packed = fx.Int32(IDS[mb * bm + tx])
                    rtok = row_packed & 0xFFFFFF
                    old = fx.Int32(-1)
                    if rtok < m:
                        old = atomic_add(tick + fx.Int64((rtok * nblk + nb) * TICKET_STRIDE), 1)
                    last = (rtok < m) & (old == TOPK - 1)
                    llvm.StoreOp(raw(last.select(rtok, fx.Int32(-1))), lds_ptr(base, tx * 4))
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
                for r in range_constexpr(16 * mr):
                    tk = fx.Int32(llvm.load(T.i32, lds_ptr(base, r * 4)))
                    if tk >= 0:
                        if tx == 0:
                            atomic_reset(tick + fx.Int64((tk * nblk + nb) * TICKET_STRIDE))
                        if tx < tn // 2:
                            rv = [fx.Int32(REV[tk * TOPK + s]) for s in range_constexpr(TOPK)]
                            vs = [fx.Int32(buffer_ops.buffer_load(slr, (rv[s] * H + nb * tn) // 2 + tx, vec_width=1,
                                                                  dtype=fx.Int32, cache_modifier=SC1))
                                  for s in range_constexpr(TOPK)]
                            a0 = fx.Float32(0.0)
                            a1 = fx.Float32(0.0)
                            for s in range_constexpr(TOPK):
                                a0 = a0 + (vs[s] << 16).bitcast(fx.Float32)
                                a1 = a1 + (vs[s] & -65536).bitcast(fx.Float32)
                            pk = Vec.from_elements([a0, a1], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                            buffer_ops.buffer_store(pk, outr, (tk * H + nb * tn) // 2 + tx)

    return tile, nblk


@functools.cache
@launcher
def compile_gemm2(*, tn=256, bm=16, pf=2, reduce=True, name_suffix="", abl=0):
    tile, nblk = compile_gemm2_tile(tn=tn, bm=bm, pf=pf, reduce=reduce, abl=abl)
    name = f"moe_gemm2_a4w4_h{H}_i{I}_bm{bm}_tn{tn}_pf{pf}{'_sum16' if reduce else '_slots'}{name_suffix}" + (f"_abl{abl}" if abl else "")

    @fx.struct
    class Shared:
        raw: fx.Array[fx.Uint8, 128, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, SL: fx.Tensor, OUT: fx.Tensor,
               IDS: fx.Tensor, WT: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, REV: fx.Tensor, m: fx.Int32,
               rows: fx.Int32, tick: fx.Int64):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.raw.ptr))
        tile(base, fx.Int32(fx.block_idx.x), A, AS, W, WS, SL, OUT, IDS, WT, EIDS, NV, REV, m, rows, fx.Int64(0), tick)

    @flyc.jit
    def launch(A: fx.Tensor, AS: fx.Tensor, W: fx.Tensor, WS: fx.Tensor, SL: fx.Tensor, OUT: fx.Tensor,
               IDS: fx.Tensor, WT: fx.Tensor, EIDS: fx.Tensor, NV: fx.Tensor, REV: fx.Tensor, m: fx.Int32,
               rows: fx.Int32, tick: fx.Int64, stream: fx.Stream):
        kernel(A, AS, W, WS, SL, OUT, IDS, WT, EIDS, NV, REV, m, rows, tick).launch(
            grid=(rows // bm * nblk, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
