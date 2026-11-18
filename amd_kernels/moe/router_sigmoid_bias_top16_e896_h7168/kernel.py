# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""bf16 [M, 7168] x bf16 [896, 7168]^T -> fp32 split-K partials [KS, M, 896] of the MoE router logits, 1-192 rows.

Tiles: 14 column tiles of 64 weight rows (wave w owns the 16-row group w) x KS K slices of 112 / KS blocks of 64
(grid 14 KS). A workgroup covers all M rows: both operands go HBM / L2 -> LDS by asynchronous `buffer_load ... lds`
into a ring of R slots (one slot = the 4 weight fragments + MT activation fragments, 2 KB each), with the XOR-swizzled
fragment reads and the half-block software pipeline of gemm/bf16_n50016_k6144 (v_mfma_f32_16x16x32_bf16, K ascending).
Weight loads are non-temporal: the weight streams through.

Epilogue: each workgroup stores its fp32 accumulators to the partial slab [KS, M, 896] (plain stores: the
top-k launch that follows sums the KS slabs in slice order, so the logits' bits do not depend on which slice ends
last). No tickets, no atomics.

Weight layouts: ROW (the checkpoint's [896, 7168]) or SHUF ([56 groups, 112 K blocks, 16 rows, 64 K]: one DMA reads
1 KB in one piece, a bit-exact permutation made once at load).
"""
import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T, Vector as Vec

from .reference import E as N, H as K
from .aux import (FLAG_BASE, MASK_WORDS, aux_counts, aux_expert_blocks, aux_max_blocks, aux_pair_row, aux_prefill,
                  aux_publish_cum, atomic_or)
from .topk import (LDS_WORDS, MAXC, NQ, RANK_UNROLL, TOPK, _ballot, _popcount, topk_fast_write, topk_new, topk_pivot, topk_q_scores,
                   topk_rank_start, topk_rank_step, topk_slow_out, topk_slow_select,
                   topk_finish, topk_tie_masks)

SC1 = 0x10  # agent-scope (cross-XCD) stores and loads of the slabs and counters
FLAG_WORDS = 256  # one flag word per workgroup (a call advances each by one)
import os
POLL_SLEEP = int(os.environ.get('KK_POLL', '2'))  # s_sleep units (64 clocks) between two polls of the tile counters
NB = K // 64  # 112 K blocks
NWV = 4
W_CM = 2  # nt
MT_MAX = 12
RING_BYTES = 146 * 1024


def ring_depth(frags: int) -> int:
    """Ring slots that fit the LDS next to the top-k arrays (one slot = ``frags`` fragments of 2 KB)."""
    return 4 if frags * 2048 * 4 <= RING_BYTES else 3


def tiles(nf: int) -> int:
    """Column tiles of ``nf`` 16-column fragments."""
    return N // 16 // nf


def partial_bytes(m: int, ks: int) -> int:
    return ks * m * N * 4


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


def _sb():
    fx.rocdl.sched_barrier(0)


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), _raw(fx.Int64(byte_addr)))


@launcher
def compile_router_e896_h7168(mt: int, ks: int, shuf: int, fused: int = 1, wn: int = 4, nrw: int = 1, aux: int = 0):
    """``fused`` 0: the GEMM launch alone (partials out); 1: the whole router (the top-16 follows in the same launch).
    The workgroup tile is ``wn nrw`` 16-column weight fragments; its ``wn`` x ``4 / wn`` waves split the columns
    (``nrw`` fragments each) and the row fragments (``MTP / (4 / wn)`` each). ``aux`` 1: the launch also writes the BM=16
    MoE sort of the top-16 and zeroes the moe buffer (``aux.py``)."""
    MT, KS, SHUF, FUSED, WN, NRW, AUX = mt, ks, shuf, fused, wn, nrw, aux
    assert not AUX or FUSED
    WM = NWV // WN
    NF = WN * NRW  # weight fragments of the tile
    TN = 16 * NF
    TILES = tiles(NF)
    MTP = (MT + WM - 1) // WM * WM  # row fragments, a multiple of the row waves
    MW = MTP // WM  # row fragments of a wave
    assert NWV % WN == 0 and N // 16 % NF == 0 and (NF * 2) % NWV == 0
    assert 1 <= MT <= MT_MAX and NB % KS == 0 and (not FUSED or TILES * KS >= 16 * MT)  # a token a workgroup
    NBK = NB // KS  # K blocks per workgroup
    MR = 16 * MT
    SLOT_FRAGS = NF + MTP
    R = min(ring_depth(SLOT_FRAGS), NBK)
    WF = NF  # weight positions per slot
    SLOT = SLOT_FRAGS * 2048
    LDS = R * SLOT
    XD = MTP * 2  # activation DMAs per block
    XPW = (XD + NWV - 1) // NWV
    WPW = NF * 2 // NWV  # weight DMAs per wave per block
    PER = WPW + XPW  # DMAs per wave per block
    assert LDS <= RING_BYTES and (R - 1) * PER <= 63, (LDS, R, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]
        ent: fx.Array[fx.Int32, NWV * LDS_WORDS["ent"], 16]
        osig: fx.Array[fx.Float32, NWV * LDS_WORDS["osig"], 16]
        oid: fx.Array[fx.Int32, NWV * LDS_WORDS["oid"], 16]
        tie: fx.Array[fx.Int32, NWV * LDS_WORDS["tie"], 16]
        tex: fx.Array[fx.Int32, NWV * LDS_WORDS["tex"], 16]
        tmask: fx.Array[fx.Int32, NWV * LDS_WORDS["tmask"], 16]

    name = f"router_{'fused' if FUSED else 'partials'}_e896_h7168_mt{MT}_ks{KS}_w{WN}n{NRW}" + ("_aux" if AUX else "") + ("_shuf" if SHUF else "_row")

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_p: fx.Int64, arg_c: fx.Int64, arg_bias: fx.Int64,
               arg_ow: fx.Int64, arg_oid: fx.Int64, arg_aux: fx.Int64, arg_sti: fx.Int64, arg_sw: fx.Int64,
               arg_sei: fx.Int64, arg_nv: fx.Int64, arg_mi: fx.Int64, arg_rs: fx.Int64, arg_mb: fx.Int64,
               i32_m: fx.Int32, i32_os: fx.Int32, i32_scale: fx.Int32, i32_cap: fx.Int32, i32_zvecs: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        kz = bid % KS
        tile = bid // KS
        n0 = tile * TN
        wn = wave % WN  # the wave's column fragments wn NRW ..
        wm = wave // WN  # and row fragments wm MW ..
        b0 = kz * NBK
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        if const_expr(FUSED):
            # this call's tag: every workgroup's flag advances by one a call, so the flag read now plus one is the value all
            # workgroups publish when their slabs are stored
            flags = buffer_ops.create_buffer_resource_from_addr(arg_c, num_records_bytes=FLAG_WORDS * 4)
            tag = fx.Int32(buffer_ops.buffer_load(flags, bid, vec_width=1, dtype=fx.Int32,
                                                  cache_modifier=SC1 | 1)) + 1

        # DMA jj of a fragment: rows 8 jj + r8, 16-byte piece (lane % 8) ^ ((r8 >> 1) | (jj << 2)) of the line
        w_src, w_dst = [], []
        for i in range_constexpr(WPW):
            d = wave + i * NWV  # weight DMA 2 f + jj: rows 8 jj + r8 of fragment f
            f = d // 2
            jj = d % 2
            pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
            if const_expr(SHUF):
                off = (tile * NF + f) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
            else:
                off = (n0 + f * 16 + jj * 8 + r8) * (K * 2) + pc * 16
            w_src.append(off)
            w_dst.append(d * 1024)
        x_src, x_dst = [], []
        for i in range_constexpr(XPW):
            d = wave + i * NWV
            d = (d < XD).select(d, fx.Int32(XD - 1))  # spare DMAs repeat the last one (same bytes, same place)
            jx = d % 2
            row = (d // 2) * 16 + jx * 8 + r8
            row = (row < i32_m).select(row, i32_m - 1)  # rows past M read the last row; never stored
            px = (lane % 8) ^ ((r8 // 2) + jx * 4)
            x_src.append(row * (K * 2) + px * 16)
            x_dst.append((WF * 2 + d) * 1024)
        aux_w = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), W_CM)
        aux_x = ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0)

        def issue(rs, dst, src, b, soff, aux):
            d = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + (b % R) * SLOT + dst))
            lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(d)))
            _rocdl_ods.raw_ptr_buffer_load_async_lds(
                rs, lp, fx.Int32(16).ir_value(), fx.Int32(src).ir_value(),
                fx.Int32(soff).ir_value(), fx.Int32(0).ir_value(), aux=aux)

        def dma(b):
            bg = b0 + b
            for q in range_constexpr(WPW):
                issue(wr, w_dst[q], w_src[q], b, bg * (2048 if SHUF else 128), aux_w)
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, bg * 128, aux_x)

        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % R) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MW)] for _ in range_constexpr(NRW)]

        def read_ops(b, kk):
            return ([rd(b, wn * NRW + i, kk) for i in range_constexpr(NRW)],
                    [rd(b, WF + wm * MW + m, kk) for m in range_constexpr(MW)])

        def mfmas(ops):
            wv, xv = ops
            for i in range_constexpr(NRW):
                for m in range_constexpr(MW):
                    acc[i][m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                        Vec.make_type(4, fx.Float32), [wv[i], xv[m], acc[i][m], 0, 0, 0]))

        def barrier():
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        for b in range_constexpr(R):
            dma(b)
        _sb()
        fx.rocdl.s_waitcnt(vmcnt=(R - 1) * PER)
        barrier()
        _sb()
        ops_a = read_ops(0, 0)
        for b in range_constexpr(NBK):
            _sb()
            ops_b = read_ops(b, 1)
            _sb()
            mfmas(ops_a)
            _sb()
            if b + 1 < NBK:
                fx.rocdl.s_waitcnt(lgkmcnt=0, vmcnt=(min(b + R - 1, NBK - 1) - (b + 1)) * PER)
                barrier()
                _sb()
                ops_a = read_ops(b + 1, 0)
                _sb()
                if b + R < NBK:
                    dma(b + R)
                _sb()
            mfmas(ops_b)

        pr = buffer_ops.create_buffer_resource_from_addr(arg_p, num_records_bytes=KS * i32_m * (N * 4))
        for i in range_constexpr(NRW):
            for m in range_constexpr(MW):
                row = (wm * MW + m) * 16 + l16
                buffer_ops.buffer_store(acc[i][m].bitcast(fx.Int32), pr,
                                        ((kz * i32_m + row) * N + n0 + (wn * NRW + i) * 16 + g * 4) * 4,
                                        offset_is_bytes=True, mask=row < i32_m, cache_modifier=SC1 if FUSED else 0)
        if const_expr(AUX):
            pre_bytes = fx.Int64(i32_cap) * 4
            aux_prefill(bid * (64 * NWV) + tx, TILES * KS * 64 * NWV, i32_m, i32_cap, i32_zvecs,
                        buffer_ops.create_buffer_resource_from_addr(arg_aux, num_records_bytes=FLAG_BASE * 4),
                        buffer_ops.create_buffer_resource_from_addr(arg_sti, num_records_bytes=pre_bytes),
                        buffer_ops.create_buffer_resource_from_addr(arg_sw, num_records_bytes=pre_bytes),
                        buffer_ops.create_buffer_resource_from_addr(arg_mi, num_records_bytes=pre_bytes),
                        buffer_ops.create_buffer_resource_from_addr(arg_mb, num_records_bytes=fx.Int64(i32_zvecs) * 16))
        if const_expr(FUSED):
            # every slab store is acknowledged before the arrival; workgroup ``t`` (t < M) then waits for all tiles and
            # takes token t in its wave 0 (a token's 50 KB of slabs go through one CU's load path)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            buffer_ops.buffer_store(tag, flags, bid, mask=tx == 0, cache_modifier=SC1)
            # token ``bid`` (bid < M; a token per workgroup: the grid has at least 16 MT workgroups): wave 0 polls, then
            # the four waves each take one group of 256 experts (slab loads, sums, sigmoids, keys) into LDS (the dead
            # GEMM ring) and wave 0 selects
            if bid < i32_m:
                if wave == 0:
                    done = fx.Int32(0)
                    while done < 4 * 64:
                        done = fx.Int32(0)
                        for r in range_constexpr(4):
                            f = fx.Int32(buffer_ops.buffer_load(flags, lane + 64 * r, vec_width=1, dtype=fx.Int32,
                                                                mask=(lane + 64 * r) < TILES * KS,
                                                                cache_modifier=SC1 | 1))
                            done = done + _popcount(_ballot((f == tag) | ((lane + 64 * r) >= TILES * KS)))
                        fx.rocdl.s_sleep(POLL_SLEEP)
                fx.gpu.barrier()
                br = buffer_ops.create_buffer_resource_from_addr(arg_bias, num_records_bytes=N * 4)
                for j, (sg, key) in enumerate(topk_q_scores(pr, br, lane, bid, i32_m, KS, SC1, wave)):
                    llvm.store(_raw(Vec.from_elements([fx.Int32(key), sg.bitcast(fx.Int32)], fx.Int32)),
                               _lds_ptr(lds_base + (lane * 16 + 4 * wave + j) * 8), alignment=8)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                fx.gpu.barrier()
            if (bid < i32_m) & (wave == 0):
                tok = bid
                out_bytes = fx.Int64(i32_m) * fx.Int64(i32_os) * 4
                owr = buffer_ops.create_buffer_resource_from_addr(arg_ow, num_records_bytes=out_bytes)
                oir = buffer_ops.create_buffer_resource_from_addr(arg_oid, num_records_bytes=out_bytes)
                st = topk_new(lane)
                for i in range_constexpr(8):
                    pair = Vec(llvm.load(T.vec(4, T.i32), _lds_ptr(lds_base + (lane * 16 + 2 * i) * 8), alignment=16))
                    st.key.append(fx.Uint32(pair[0]))
                    st.sig.append(fx.Int32(pair[1]).bitcast(fx.Float32))
                    st.key.append(fx.Uint32(pair[2]))
                    st.sig.append(fx.Int32(pair[3]).bitcast(fx.Float32))
                topk_pivot(st, smem, wave, lane)
                out_id = fx.Int32(0)
                out_w = fx.Float32(0.0)
                if st.nc <= MAXC:
                    topk_fast_write(st, smem, wave, lane)
                    topk_rank_start(st, smem, wave, lane)
                    rank = fx.Int32(0)
                    e = fx.Int32(0)
                    while e < st.nc:
                        rank = topk_rank_step(rank, e, st, smem, wave)
                        e = e + RANK_UNROLL
                    out_id, out_w = topk_finish(rank, lane < st.nc, st.mysig, st.myid, lane, smem, wave, owr, oir, tok,
                                                i32_os, i32_scale)
                else:
                    ss = topk_slow_select(st)
                    if ss.total_ties != ss.need:
                        topk_tie_masks(ss, smem, wave, lane)
                    rank, mysig, myid = topk_slow_out(ss, smem, wave, lane)
                    out_id, out_w = topk_finish(rank, lane < TOPK, mysig, myid, lane, smem, wave, owr, oir, tok, i32_os, i32_scale)
                if const_expr(AUX):
                    # publish the token (its bit in the mask of each routed expert, then its flag), wait for every token
                    # and write the token's 16 rows of the sort
                    if lane < TOPK:
                        atomic_or(arg_aux + fx.Int64((out_id * MASK_WORDS + (tok >> 5)) * 4), fx.Int32(1) << (tok & 31))
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    fres = buffer_ops.create_buffer_resource_from_addr(arg_aux + FLAG_BASE * 4, num_records_bytes=256 * 4)
                    buffer_ops.buffer_store(tag, fres, tok, mask=lane == 0, cache_modifier=SC1)
                    fx.rocdl.s_waitcnt(vmcnt=0)
                    done = fx.Int32(0)
                    while done < 192:
                        done = fx.Int32(0)
                        for r in range_constexpr(3):
                            f = fx.Int32(buffer_ops.buffer_load(fres, lane + 64 * r, vec_width=1, dtype=fx.Int32,
                                                                mask=(lane + 64 * r) < i32_m, cache_modifier=SC1 | 1))
                            done = done + _popcount(_ballot((f == tag) | ((lane + 64 * r) >= i32_m)))
                        fx.rocdl.s_sleep(POLL_SLEEP)
                    mres = buffer_ops.create_buffer_resource_from_addr(arg_aux, num_records_bytes=FLAG_BASE * 4)
                    ac = aux_counts(mres, lane)
                    aux_publish_cum(ac, lds_base, wave, lane)
                    row = aux_pair_row(lds_base, wave, mres, out_id, tok)
                    cap_bytes = fx.Int64(i32_cap) * 4
                    sti_r = buffer_ops.create_buffer_resource_from_addr(arg_sti, num_records_bytes=cap_bytes)
                    sw_r = buffer_ops.create_buffer_resource_from_addr(arg_sw, num_records_bytes=cap_bytes)
                    mi_r = buffer_ops.create_buffer_resource_from_addr(arg_mi, num_records_bytes=cap_bytes)
                    rs_r = buffer_ops.create_buffer_resource_from_addr(arg_rs, num_records_bytes=fx.Int64(i32_m) * 64)
                    keep = lane < TOPK
                    buffer_ops.buffer_store(tok | (lane << 24), sti_r, row, mask=keep, cache_modifier=SC1)
                    buffer_ops.buffer_store(out_w, sw_r, row, mask=keep, cache_modifier=SC1)
                    buffer_ops.buffer_store(tok, mi_r, row, mask=keep, cache_modifier=SC1)
                    buffer_ops.buffer_store(row, rs_r, tok * TOPK + lane, mask=keep, cache_modifier=SC1)
                    if tok == 0:
                        sei_r = buffer_ops.create_buffer_resource_from_addr(arg_sei, num_records_bytes=cap_bytes)
                        nv_r = buffer_ops.create_buffer_resource_from_addr(arg_nv, num_records_bytes=8)
                        buffer_ops.buffer_store(ac.total, nv_r, 0, mask=lane == 0, cache_modifier=SC1)
                        buffer_ops.buffer_store(i32_m, nv_r, 1, mask=lane == 0, cache_modifier=SC1)
                        maxb = aux_max_blocks(ac)
                        blk = fx.Int32(0)
                        while blk < maxb:
                            aux_expert_blocks(ac, blk, lane, sei_r)
                            blk = blk + 1

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_p: fx.Int64, arg_c: fx.Int64, arg_bias: fx.Int64,
               arg_ow: fx.Int64, arg_oid: fx.Int64, arg_aux: fx.Int64, arg_sti: fx.Int64, arg_sw: fx.Int64,
               arg_sei: fx.Int64, arg_nv: fx.Int64, arg_mi: fx.Int64, arg_rs: fx.Int64, arg_mb: fx.Int64,
               i32_m: fx.Int32, i32_os: fx.Int32, i32_scale: fx.Int32, i32_cap: fx.Int32, i32_zvecs: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_w, arg_p, arg_c, arg_bias, arg_ow, arg_oid, arg_aux, arg_sti, arg_sw, arg_sei, arg_nv,
               arg_mi, arg_rs, arg_mb, i32_m, i32_os, i32_scale, i32_cap, i32_zvecs).launch(
            grid=(TILES * KS, 1, 1), block=(64 * NWV, 1, 1), stream=stream)

    return launch
