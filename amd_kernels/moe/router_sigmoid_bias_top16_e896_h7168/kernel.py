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
from .topk import (LDS_WORDS, MAXC, NQ, RANK_UNROLL, TOPK, _ballot, _popcount, topk_fast_write, topk_new, topk_pivot, topk_q_finish,
                   topk_q_load, topk_rank_start, topk_rank_step, topk_slow_out, topk_slow_select,
                   topk_finish, topk_tie_masks)

TN = 64
TILES = N // TN  # 14
SC1 = 0x10  # agent-scope (cross-XCD) stores and loads of the slabs and counters
CNT_STRIDE = 128  # bytes: one counter per tile, each on its own line
EPOCH = 16  # what one call adds to each tile counter
POLL_SLEEP = 2  # s_sleep units (64 clocks) between two polls of the tile counters
NB = K // 64  # 112 K blocks
NWV = 4
W_CM = 2  # nt
MT_MAX = 12
RING_BYTES = 160 * 1024


def ring_depth(mt: int) -> int:
    return 4 if (4 + mt) * 2048 * 4 <= RING_BYTES else 3


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


def _atomic_add(addr, v):
    """Returning scalar global RMW by one active lane; the caller drained its sc1 stores first."""
    low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int64(addr).to(fx.Int32)))
    high = fx.Int32(fx.rocdl.readfirstlane(T.i32, (fx.Int64(addr) >> 32).to(fx.Int32)))
    pointer = (low.to(fx.Uint32).to(fx.Uint64) | (high.to(fx.Uint32).to(fx.Uint64) << 32)).bitcast(fx.Int64)
    inc = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(v)))
    return fx.Int32(llvm.inline_asm(T.i32, [fx.as_ir_value(inc), fx.as_ir_value(pointer)],
                                    "s_atomic_add $0, $2, 0 glc\n\ts_waitcnt lgkmcnt(0)", "=s,0,s,~{memory}",
                                    has_side_effects=True, is_align_stack=False,
                                    asm_dialect=llvm.AsmDialect.AD_ATT))


@launcher
def compile_router_e896_h7168(mt: int, ks: int, shuf: int, fused: int = 1):
    """``fused`` 0: the GEMM launch alone (partials out); 1: the whole router (the top-16 follows in the same launch)."""
    MT, KS, SHUF, FUSED = mt, ks, shuf, fused
    assert 1 <= MT <= MT_MAX and NB % KS == 0 and KS <= EPOCH and (not FUSED or TILES * KS * NWV >= 16 * MT)  # four tokens a workgroup at most
    NBK = NB // KS  # K blocks per workgroup
    MR = 16 * MT
    R = min(ring_depth(MT), NBK)
    WF = NWV  # weight positions per slot
    SLOT = (WF + MT) * 2048
    LDS = R * SLOT
    XD = MT * 2  # activation DMAs per block
    XPW = (XD + NWV - 1) // NWV
    PER = 2 + XPW  # DMAs per wave per block
    assert LDS <= RING_BYTES and (R - 1) * PER <= 63, (LDS, R, PER)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]
        flag: fx.Array[fx.Int32, 4]
        ent: fx.Array[fx.Int32, NWV * LDS_WORDS["ent"], 16]
        tie: fx.Array[fx.Int32, NWV * LDS_WORDS["tie"], 16]
        tex: fx.Array[fx.Int32, NWV * LDS_WORDS["tex"], 16]
        tmask: fx.Array[fx.Int32, NWV * LDS_WORDS["tmask"], 16]

    name = f"router_{'fused' if FUSED else 'partials'}_e896_h7168_mt{MT}_ks{KS}" + ("_shuf" if SHUF else "_row")

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(arg_x: fx.Int64, arg_w: fx.Int64, arg_p: fx.Int64, arg_c: fx.Int64, arg_bias: fx.Int64,
               arg_ow: fx.Int64, arg_oid: fx.Int64, i32_m: fx.Int32, i32_os: fx.Int32, i32_scale: fx.Int32):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        kz = bid % KS
        tile = bid // KS
        n0 = tile * TN
        b0 = kz * NBK
        g = lane // 16
        l16 = lane % 16
        r8 = lane // 8

        xr = buffer_ops.create_buffer_resource_from_addr(arg_x, num_records_bytes=i32_m * (K * 2))
        wr = buffer_ops.create_buffer_resource_from_addr(arg_w, num_records_bytes=N * K * 2)
        lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))

        # DMA jj of a fragment: rows 8 jj + r8, 16-byte piece (lane % 8) ^ ((r8 >> 1) | (jj << 2)) of the line
        w_src, w_dst = [], []
        for jj in range_constexpr(2):
            pc = (lane % 8) ^ ((r8 // 2) + jj * 4)
            if const_expr(SHUF):
                off = (tile * NWV + wave) * (K * 32) + (jj * 8 + r8) * 128 + pc * 16
            else:
                off = (n0 + wave * 16 + jj * 8 + r8) * (K * 2) + pc * 16
            w_src.append(off)
            w_dst.append((wave * 2 + jj) * 1024)
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
            for q in range_constexpr(2):
                issue(wr, w_dst[q], w_src[q], b, bg * (2048 if SHUF else 128), aux_w)
            for q in range_constexpr(XPW):
                issue(xr, x_dst[q], x_src[q], b, bg * 128, aux_x)

        r8r = l16 % 8
        hh = l16 // 8
        rd_off = [hh * 1024 + (r8r * 8 + ((kk * 4 + g) ^ ((r8r // 2) + hh * 4))) * 16 for kk in range_constexpr(2)]

        def rd(b, pos, kk):
            p = _lds_ptr(lds_base + (b % R) * SLOT + pos * 2048 + rd_off[kk])
            return Vec(llvm.load(T.vec(4, T.i32), p, alignment=16)).bitcast(fx.BFloat16)

        acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(MT)]

        def read_ops(b, kk):
            return rd(b, wave, kk), [rd(b, WF + m, kk) for m in range_constexpr(MT)]

        def mfmas(ops):
            wv, xv = ops
            for m in range_constexpr(MT):
                acc[m] = Vec(fx.rocdl.mfma_f32_16x16x32_bf16(
                    Vec.make_type(4, fx.Float32), [wv, xv[m], acc[m], 0, 0, 0]))

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
        for m in range_constexpr(MT):
            buffer_ops.buffer_store(acc[m].bitcast(fx.Int32), pr,
                                    ((kz * i32_m + m * 16 + l16) * N + n0 + wave * 16 + g * 4) * 4,
                                    offset_is_bytes=True, mask=(m * 16 + l16) < i32_m,
                                    cache_modifier=SC1 if FUSED else 0)
        if const_expr(FUSED):
            # every slab store is acknowledged before the arrival; workgroup ``t`` (t < M) then waits for all tiles and
            # takes token t in its wave 0 (a token's 50 KB of slabs go through one CU's load path)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                inc = fx.Int32(EPOCH // KS) + (kz < EPOCH % KS).select(fx.Int32(1), fx.Int32(0))
                old = _atomic_add(arg_c + fx.Int64(tile) * CNT_STRIDE, inc)
                smem.flag.ptr[0] = old - (old & (EPOCH - 1))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            tok = bid + wave * (TILES * KS)  # the token of this wave: one per workgroup, up to four when M exceeds the grid
            if tok < i32_m:
                ebase = fx.Int32(smem.flag.ptr[0])
                cr = buffer_ops.create_buffer_resource_from_addr(arg_c, num_records_bytes=TILES * CNT_STRIDE)
                br = buffer_ops.create_buffer_resource_from_addr(arg_bias, num_records_bytes=N * 4)
                out_bytes = fx.Int64(i32_m) * fx.Int64(i32_os) * 4
                owr = buffer_ops.create_buffer_resource_from_addr(arg_ow, num_records_bytes=out_bytes)
                oir = buffer_ops.create_buffer_resource_from_addr(arg_oid, num_records_bytes=out_bytes)
                st = topk_new(lane)
                # the experts 256 q + 4 lane + j live in tiles 4 q .. 4 q + 3: take each group as its tiles finish
                for q in range_constexpr(NQ):
                    in_group = (lane >= 4 * q) & (lane < min(4 * q + 4, TILES))
                    ready = fx.Int32(0)
                    while ready < min(4, TILES - 4 * q):
                        c = fx.Int32(buffer_ops.buffer_load(cr, lane * (CNT_STRIDE // 4), vec_width=1,
                                                            dtype=fx.Int32, mask=lane < TILES,
                                                            cache_modifier=SC1 | 1))
                        ready = _popcount(_ballot(((c - ebase) >= EPOCH) & in_group))
                        fx.rocdl.s_sleep(POLL_SLEEP)
                    parts, bv = topk_q_load(pr, br, lane, tok, i32_m, KS, SC1, q)
                    topk_q_finish(st, lane, q, parts, bv)
                topk_pivot(st, smem, wave, lane)
                if st.nc <= MAXC:
                    topk_fast_write(st, smem, wave, lane)
                    topk_rank_start(st, smem, wave, lane)
                    rank = fx.Int32(0)
                    e = fx.Int32(0)
                    while e < st.nc:
                        rank = topk_rank_step(rank, e, st, smem, wave)
                        e = e + RANK_UNROLL
                    topk_finish(rank, lane < st.nc, st.mysig, st.myid, lane, owr, oir, tok, i32_os, i32_scale)
                else:
                    ss = topk_slow_select(st)
                    if ss.total_ties != ss.need:
                        topk_tie_masks(ss, smem, wave, lane)
                    rank, mysig, myid = topk_slow_out(ss, smem, wave, lane)
                    topk_finish(rank, lane < TOPK, mysig, myid, lane, owr, oir, tok, i32_os, i32_scale)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_w: fx.Int64, arg_p: fx.Int64, arg_c: fx.Int64, arg_bias: fx.Int64,
               arg_ow: fx.Int64, arg_oid: fx.Int64, i32_m: fx.Int32, i32_os: fx.Int32, i32_scale: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_w, arg_p, arg_c, arg_bias, arg_ow, arg_oid, i32_m, i32_os, i32_scale).launch(
            grid=(TILES * KS, 1, 1), block=(64 * NWV, 1, 1), stream=stream)

    return launch
