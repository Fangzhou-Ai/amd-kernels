# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Router split-K partials [KS, M, 896] -> AITER's biased top-16 (``topk_reg_kernel``: sigmoid, + correction bias, one
group, renormalised), one wave per token (gfx950): its ids, slot order, tie rule and weights bit for bit.

The wave sums the KS slabs of its token in slice order, so the logits are the same bits whichever slice finished last.
Lane l holds the experts 256 q + 4 l + j (q 0-3, j 0-3; q 3 only on lanes 0-31), AITER's float4 order, so every slab
load is one coalesced 16-byte read and AITER's tie rank (the lane's DPP-tree priority, then 4 q + j) is a function of
(l, 4 q + j). The selection is the top 16 of the total order (biased score descending, tie rank ascending):

1. each lane sorts its 16 keys (the order-preserving unsigned key of the fp32 biased score); 16 rounds pop the
   wave-wide maximum (DPP tree, ballot count of the lanes holding it) until 16 keys are consumed. The last popped
   key is the 16th largest, and the keys consumed before its value first came up are the larger ones;
2. the elements with a larger key are selected. If the elements equal to the 16th key are exactly the slots left
   (always, but for exact ties), they are selected too; otherwise the ones with the lowest tie ranks are, counted
   over the lanes in tie-rank order;
3. the 16 selected go to LDS; lanes 0-15 each rank one of them against the other 15 (key, then tie rank), scatter
   it to its slot, sum the 16 unbiased sigmoids in slot order and write ``sigmoid * (scale / sum)``.
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

E, TOPK, NQ, EPL = 896, 16, 4, 16
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}
DPP_TREE = (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143)
NEG_LOG2E = -1.44269504088896340736


def _ir(v):
    return v.ir_value() if hasattr(v, "ir_value") else v


def _dpp(v, ctrl, row_mask=0xF):
    """DPP move; a lane without a source reads 0 (the identity of the unsigned max and of the sum it feeds)."""
    return fx.Uint32(_rocdl_ops.update_dpp(T.i32, _ir(fx.Int32(0)), _ir(v), ctrl, row_mask, 0xF, True))


def _scan(x):
    """Wave inclusive prefix sum of int32 per lane (AITER's DPP scan); lane 63 holds the total."""
    for ctrl in (0x111, 0x112, 0x114, 0x118):
        x = x + fx.Int32(_dpp(x, ctrl))
    x = x + fx.Int32(_dpp(x, 0x142, 0xA))
    return x + fx.Int32(_dpp(x, 0x143, 0xC))


def _batcher_pairs(n):
    """Odd-even merge sort comparators (i < j, ascending) for any n."""
    pairs, p = [], 1
    while p < n:
        k = p
        while k >= 1:
            for j in range(k % p, n - k, 2 * k):
                for i in range(min(k, n - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return pairs


SORT_PAIRS = _batcher_pairs(EPL)


def _ballot(pred):
    return fx.Int64(fx.rocdl.ballot(T.i64, pred))


def _popcount(mask):
    return fx.Int32(fx.math.ctpop(mask).to(fx.Int32))


def _lane_rank(lane):
    """AITER's legacy wave64 DPP tie priority of a lane (a permutation of 0-63; lower wins)."""
    low2 = (lane & 3) ^ (((lane >> 2) & 1) * 3)
    bit2 = (((lane >> 2) ^ (lane >> 3)) & 1) << 2
    return ((~lane) & 0x38) | bit2 | low2


@launcher
def compile_router_sigmoid_bias_top16_e896(ks: int):
    name = f"router_sigmoid_bias_top16_e896_ks{ks}"

    @fx.struct
    class Shared:
        ent: fx.Array[fx.Int32, 2 * TOPK * 4, 16]  # per slot: key, tie rank << 16 | expert, sigmoid bits, -
        osig: fx.Array[fx.Float32, TOPK, 16]
        oid: fx.Array[fx.Int32, TOPK, 16]
        tie: fx.Array[fx.Int32, 64, 16]
        tex: fx.Array[fx.Int32, 64, 16]
        tmask: fx.Array[fx.Int32, 64, 16]

    @flyc.kernel(name=name, known_block_size=[64, 1, 1])
    def kernel(arg_p: fx.Int64, arg_bias: fx.Int64, arg_w: fx.Int64, arg_ids: fx.Int64, i32_m: fx.Int32,
               i32_os: fx.Int32, i32_scale: fx.Int32):
        lane, tok = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_p, num_records_bytes=fx.Int64(ks) * fx.Int64(i32_m) * (E * 4))
        br = buffer_ops.create_buffer_resource_from_addr(arg_bias, num_records_bytes=E * 4)
        valid3 = lane < 32

        masks = [None, None, None, valid3]
        parts = [[Vec(buffer_ops.buffer_load(pr, (s * i32_m + tok) * E + 256 * q + 4 * lane, vec_width=4,
                                             dtype=fx.Float32, mask=masks[q])) for s in range_constexpr(ks)]
                 for q in range_constexpr(NQ)]
        bvs = [Vec(buffer_ops.buffer_load(br, 256 * q + 4 * lane, vec_width=4, dtype=fx.Float32, mask=masks[q]))
               for q in range_constexpr(NQ)]
        x, b = [], []
        for q in range_constexpr(NQ):
            acc = parts[q][0]
            for s in range_constexpr(1, ks):
                acc = acc + parts[q][s]
            for j in range_constexpr(4):
                x.append(fx.Float32(acc[j]))
                b.append(fx.Float32(bvs[q][j]))
        sig = []
        for v in x:
            y = (v.to(fx.Float64) * fx.Float64(NEG_LOG2E)).to(fx.Float32)
            sig.append(fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(1.0) + y.exp2())))
        key = []
        for i in range_constexpr(EPL):
            u = (sig[i] + b[i]).bitcast(fx.Int32)
            k = fx.Uint32(u ^ ((u >> 31) | fx.Int32(-(2**31))))
            key.append(valid3.select(k, fx.Uint32(0)) if i >= 4 * (NQ - 1) else k)

        srt = list(key)
        for i, j in SORT_PAIRS:
            lt = srt[i] < srt[j]
            srt[i], srt[j] = lt.select(srt[i], srt[j]), lt.select(srt[j], srt[i])
        consumed = fx.Int32(0)
        cand = fx.Uint32(0)
        before = fx.Int32(0)
        found = fx.Int32(0)
        prev = fx.Uint32(0)
        start = fx.Int32(0)
        for _ in range_constexpr(TOPK):
            head = srt[EPL - 1]
            m = head
            for ctrl in DPP_TREE:
                r = _dpp(m, ctrl)
                m = (r > m).select(r, m)
            gmax = fx.Uint32(fx.rocdl.readlane(T.i32, m, fx.Int32(63)))
            at = head == gmax
            n = _popcount(_ballot(at))
            for t in range_constexpr(EPL - 1, 0, -1):
                srt[t] = at.select(srt[t - 1], srt[t])
            srt[0] = at.select(fx.Uint32(0), srt[0])
            start = (gmax != prev).select(consumed, start)
            prev = gmax
            hit = (found == 0) & (consumed + n >= TOPK)
            cand = hit.select(gmax, cand)
            before = hit.select(start, before)
            found = hit.select(fx.Int32(1), found)
            consumed = consumed + n

        need = fx.Int32(TOPK) - before
        strict = [key[i] > cand for i in range_constexpr(EPL)]
        tie = [key[i] == cand for i in range_constexpr(EPL)]
        nties = tie[0].select(fx.Int32(1), fx.Int32(0))
        for i in range_constexpr(1, EPL):
            nties = nties + tie[i].select(fx.Int32(1), fx.Int32(0))
        total_ties = fx.Int32(fx.rocdl.readlane(T.i32, _scan(nties), fx.Int32(63)))
        rl = _lane_rank(lane)

        # more exact ties than slots: the ones with the lowest tie ranks (lanes counted in tie-rank order, then 4 q + j)
        if total_ties != need:
            smem.tie.ptr[rl] = nties
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.rocdl.sched_barrier(0)
            at_rank = fx.Int32(smem.tie.ptr[lane])
            smem.tex.ptr[lane] = _scan(at_rank) - at_rank
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.rocdl.sched_barrier(0)
            run = fx.Int32(smem.tex.ptr[rl])
            bits = fx.Int32(0)
            for i in range_constexpr(EPL):
                take = tie[i] & (run < need)
                bits = bits | take.select(fx.Int32(1 << i), fx.Int32(0))
                run = run + tie[i].select(fx.Int32(1), fx.Int32(0))
            smem.tmask.ptr[lane] = bits
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.rocdl.sched_barrier(0)
        slow_bits = fx.Int32(smem.tmask.ptr[lane])
        exact = total_ties == need
        sel = [strict[i] | (exact.select(tie[i], ((slow_bits >> i) & 1) != 0)) for i in range_constexpr(EPL)]

        nsel = sel[0].select(fx.Int32(1), fx.Int32(0))
        for i in range_constexpr(1, EPL):
            nsel = nsel + sel[i].select(fx.Int32(1), fx.Int32(0))
        run = _scan(nsel) - nsel
        for i in range_constexpr(EPL):
            slot_i = sel[i].select(run, fx.Int32(TOPK + i))
            tie_rank = (rl << 5) | i
            smem.ent.ptr[slot_i * 4] = fx.Int32(key[i])
            smem.ent.ptr[slot_i * 4 + 1] = (tie_rank << 16) | (256 * (i // 4) + 4 * lane + (i % 4))
            smem.ent.ptr[slot_i * 4 + 2] = sig[i].bitcast(fx.Int32)
            run = run + sel[i].select(fx.Int32(1), fx.Int32(0))
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.rocdl.sched_barrier(0)

        slot = lane % TOPK
        mykey = fx.Uint32(smem.ent.ptr[slot * 4])
        mymeta = fx.Int32(smem.ent.ptr[slot * 4 + 1])
        rank = fx.Int32(0)
        for e in range_constexpr(TOPK):
            ke = fx.Uint32(smem.ent.ptr[e * 4])
            me = fx.Int32(smem.ent.ptr[e * 4 + 1])
            better = (ke > mykey) | ((ke == mykey) & ((me >> 16) < (mymeta >> 16)))
            rank = rank + better.select(fx.Int32(1), fx.Int32(0))
        if lane < TOPK:
            smem.osig.ptr[rank] = fx.Int32(smem.ent.ptr[slot * 4 + 2]).bitcast(fx.Float32)
            smem.oid.ptr[rank] = mymeta & 0xFFFF
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.rocdl.sched_barrier(0)
        total = fx.Float32(smem.osig.ptr[0])
        for k in range_constexpr(1, TOPK):
            total = total + fx.Float32(smem.osig.ptr[k])
        scale = i32_scale.bitcast(fx.Float32)
        factor = scale / total
        w = (total > 0.0).select(fx.Float32(smem.osig.ptr[slot]) * factor, fx.Float32(0.0))
        ids = fx.Int32(smem.oid.ptr[slot])
        keep = lane < TOPK
        out_bytes = fx.Int64(i32_m) * fx.Int64(i32_os) * 4
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=out_bytes)
        ir = buffer_ops.create_buffer_resource_from_addr(arg_ids, num_records_bytes=out_bytes)
        buffer_ops.buffer_store(w, wr, tok * i32_os + lane, mask=keep)
        buffer_ops.buffer_store(ids, ir, tok * i32_os + lane, mask=keep)

    @flyc.jit
    def launch(arg_p: fx.Int64, arg_bias: fx.Int64, arg_w: fx.Int64, arg_ids: fx.Int64, i32_m: fx.Int32,
               i32_os: fx.Int32, i32_scale: fx.Int32, stream: fx.Stream):
        kernel(arg_p, arg_bias, arg_w, arg_ids, i32_m, i32_os, i32_scale).launch(
            grid=(i32_m, 1, 1), block=(64, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
