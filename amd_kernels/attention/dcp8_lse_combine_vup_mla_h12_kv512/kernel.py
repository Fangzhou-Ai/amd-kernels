# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DCP8 combine in one launch: W_UV up-projection of all 96 heads, push to the owner rank, LSE-weighted combine.

Hand-off protocol (control words in uncached memory, one block per rank; the data pool is hipMalloc'd and mapped on the
peers by IPC): a call takes ``seq = counter + 1`` and the slot ``seq & 1`` of every pool.

Compute workgroups (``192 * passes``, 64 token rows per pass; XCD x serves the 12 heads of owner rank x, two 64-column
halves per head): four waves split the head's K = 512 and read W_UV^T and the partial output directly; the waves' fp32
partial sums meet in LDS and are added in wave order, rounded to bf16 and stored with system-scope stores to the
owner's slot, row ``(src, t, j)``. The pass's workgroups with the first column half also store the masked LSE of their
rows the same way. A workgroup waits for its stores, then stores ``seq`` into its own word of the owner's flag array
(``flags[src][workgroup]``): no atomics.

Combine workgroups (after the compute workgroups in the grid, so they never starve them): wait for the flags of the
passes their rows belong to, read the 8 sources' rows (local memory, system-coherent loads), weigh by
``exp(lse_s - max) / sum`` and sum in source order in fp32. Every workgroup stores ``seq`` into its word of the start
array after reading the counter; the last workgroup of the grid (dispatched last, so every other one has started) waits
for all start words and then bumps the counter, so a CUDA graph replays with no host state and the grid may change
between calls. A slot is rewritten two calls later, after every peer's flags of the call in between, which it sets
after its own previous call finished.

Control block (int32 words): flags [0, 576), counter 700, start words 1024 + workgroup.
"""
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .reference import H, HPR, L, V, WORLD

NT = 256
UNITS = 2 * H  # (head, 64-column half)
PADN = 68
SYS = 0x11
CTL_BYTES = 32768
DBG_WORD = 2048
DBG = int(os.environ.get('DCP_DBG', '0'))
FLAG_WORD, SEQ_WORD, START_WORD = 0, 700, 1024
ROWS_PER_WG = 32
LOG2E = 1.4426950408889634


def _ptr(elem, addr_i64, align):
    ty = fx.PointerType.get(elem, address_space=fx.AddressSpace.Global, alignment=align)
    return fx.inttoptr(ty, fx.Int64(addr_i64))


def _atomic_add(addr_i64, v):
    """Atomic add at system scope on an uncached word; returns the old value."""
    return fx.Int32(
        _llvm.AtomicRMWOp(
            _llvm.AtomicBinOp.add,
            _ptr(T.i32, addr_i64, 4).llvm_ptr,
            fx.Int32(v).ir_value(),
            _llvm.AtomicOrdering.monotonic,
            syncscope="",
            alignment=4,
        ).result
    )


def _volatile_load(addr_i64):
    """A system-coherent load the compiler neither hoists out of a spin loop nor drops."""
    return fx.Int32(_llvm.LoadOp(T.i32, _ptr(T.i32, addr_i64, 4).llvm_ptr, alignment=4, volatile_=True).result)


def _volatile_load_v4(addr_i64):
    ty = ir.VectorType.get([4], T.i32)
    return Vec(_llvm.LoadOp(ty, _ptr(T.i32, addr_i64, 16).llvm_ptr, alignment=16, volatile_=True).result)


def _pk_bf16(a, b):
    return fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b)))
    )


def passes_of(m_tiles, pass_tiles=4):
    return -(-m_tiles // pass_tiles)


def combine_grid(m_tiles, pass_tiles=4):
    return UNITS * passes_of(m_tiles, pass_tiles) + -(-m_tiles * 16 * HPR // ROWS_PER_WG)


@launcher
def compile_combine(m_tiles: int, base_e: bool, gated: bool = False, pass_tiles: int = 4, lds_pad: int = 0, abl: int = 0, order: int = 0):
    """``m_tiles``: 16-row tiles of the padded token count (1, 2, 4, 8 or 12). ``gated``: the output is
    ``bf16(bf16(combine) * sigmoid(gate))`` with a bf16 ``gate`` [T, 12 * 128]."""
    assert 1 <= m_tiles <= 12 and m_tiles in (1, 2, 4, 8, 12)
    PASS_TILES = pass_tiles
    P = passes_of(m_tiles, PASS_TILES)
    mp = min(PASS_TILES, m_tiles)
    RROWS = 16 * PASS_TILES
    CPT = 4 * PASS_TILES
    TPR = 64 // CPT
    NC = UNITS * P
    per_src = NC // 8
    per_pass = UNITS // 8
    CW = RROWS * HPR // ROWS_PER_WG
    PW = UNITS + CW
    base = combine_grid(m_tiles, PASS_TILES) if order == 0 else P * PW
    grid = base + 1
    name = f"dcp8_lse_combine_vup_mla_h12_kv512_mt{m_tiles}_t{PASS_TILES}_p{lds_pad}_a{abl}_o{order}" + ("_e" if base_e else "_2") + ("_gate" if gated else "") + f"_dbg{DBG}"

    @fx.struct
    class Shared:
        red: fx.Array[fx.Int32, 4 * RROWS * PADN, 16]
        valid: fx.Array[fx.Int32, NT]
        pad: fx.Array[fx.Int32, max(lds_pad, 1)]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [T, >=96, 512] bf16 partial output
        i32_sxt: fx.Int32,  # token stride (dwords)
        i32_sxh: fx.Int32,  # head stride (dwords)
        arg_lse: fx.Int64,  # [T, >=96] fp32
        i32_slt: fx.Int32,
        i32_slh: fx.Int32,
        arg_wt: fx.Int64,  # [96, 128, 512] bf16 (W_UV transposed)
        arg_seq: fx.Int64,  # [num_seqs] int32 local seq lens
        arg_qsl: fx.Int64,  # [num_seqs + 1] int32
        i32_nseq: fx.Int32,
        arg_out: fx.Int64,  # [T, 12, 128] bf16
        arg_gate: fx.Int64,  # gated: [T, 1536] bf16
        arg_tab: fx.Int64,  # int64 [16]: pool of each rank, control block of each rank
        i32_m: fx.Int32,
        i32_rank: fx.Int32,
        i32_mmax: fx.Int32,
        i64_slot_bytes: fx.Int64,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        tab = _ptr(T.i64, arg_tab, 8)
        if const_expr(order == 0):
            is_comp = bid < NC
            lbid = bid
            pas_w = 0
        else:
            pas_w = bid // PW
            lbid = bid % PW
            is_comp = (lbid < UNITS) & (bid < base)
        xcd = lbid % 8
        pool_me = fx.Int64(tab[i32_rank])
        me_ctl = fx.Int64(tab[8 + i32_rank])
        pool_dst = fx.Int64(tab[xcd])
        ctl_dst = fx.Int64(tab[8 + xcd])
        ctl = buffer_ops.create_buffer_resource_from_addr(me_ctl, num_records_bytes=CTL_BYTES)
        neg_inf = fx.Float32(float("-inf"))

        def read_seq():
            return fx.Int32(
                fx.rocdl.readfirstlane(
                    T.i32,
                    fx.Int32(buffer_ops.buffer_load(ctl, SEQ_WORD, vec_width=1, dtype=fx.Int32, cache_modifier=SYS)) + 1,
                )
            )

        def stamp(slot):
            if const_expr(abl & 16):
                t64 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                if tx == 0:
                    buffer_ops.buffer_store(fx.Int32(t64), ctl, DBG_WORD + bid * 8 + slot, cache_modifier=SYS)

        not_comp = ((bid >= NC) if order == 0 else (lbid >= UNITS)) & (bid < base)
        rows = i32_mmax * HPR
        lse_off = fx.Int64(WORLD) * fx.Int64(rows) * 256
        neg_inf = fx.Float32(float("-inf"))

        def stamp(slot):
            if const_expr(abl & 16):
                t64 = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))
                if tx == 0:
                    buffer_ops.buffer_store(fx.Int32(t64), ctl, DBG_WORD + bid * 8 + slot, cache_modifier=SYS)

        stamp(0)

        rows = i32_mmax * HPR
        lse_off = fx.Int64(WORLD) * fx.Int64(rows) * 256

        if is_comp:
            if const_expr(order == 0):
                idx = lbid // 8
                pas = idx // (UNITS // 8)
                r = idx % (UNITS // 8)
            else:
                pas = pas_w
                r = lbid // 8
                idx = pas * (UNITS // 8) + r
            h = xcd * HPR + r // 2
            ns = r % 2
            j = h % HPR
            lm = lane % 16
            lg = lane // 16
            p0 = pas * PASS_TILES

            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_wt + fx.Int64(h * (V * L * 2) + ns * (64 * L * 2)), num_records_bytes=64 * L * 2
            )
            if const_expr(abl & 1):
                wv = [[Vec.filled(4, 0, fx.Int32) for nt in range_constexpr(4)] for ks in range_constexpr(4)]
            else:
                wv = [
                    [
                        Vec(
                            buffer_ops.buffer_load(
                                wr, (nt * 16 + lm) * (L // 2) + w * 64 + ks * 16 + lg * 4, vec_width=4, dtype=fx.Int32
                            )
                        )
                        for nt in range_constexpr(4)
                    ]
                    for ks in range_constexpr(4)
                ]
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x + fx.Int64(h * i32_sxh) * 4,
                num_records_bytes=((i32_m - 1) * i32_sxt + L // 2) * 4,
            )
            if const_expr(abl & 1):
                xv = [[Vec.filled(4, 0, fx.Int32) for mt in range_constexpr(mp)] for ks in range_constexpr(4)]
            else:
                xv = [
                    [
                        Vec(
                            buffer_ops.buffer_load(
                                xr,
                                (lm + (p0 + mt) * 16) * i32_sxt + w * 64 + ks * 16 + lg * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                        for mt in range_constexpr(mp)
                    ]
                    for ks in range_constexpr(4)
                ]

            seq = read_seq()
            slot_off = fx.Int64(seq & 1) * i64_slot_bytes
            dst = pool_dst + slot_off
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, START_WORD + bid, cache_modifier=SYS)
            stamp(0)

            sq = _ptr(T.i32, arg_qsl, 4)
            sl = _ptr(T.i32, arg_seq, 4)
            live = tx < i32_nseq
            at = live.select(tx, fx.Int32(0))
            s0 = fx.Int32(sq[at])
            e0 = fx.Int32(sq[at + 1])
            l0 = fx.Int32(sl[at])
            er = fx.Int32(sq[i32_nseq])
            lt = pas * RROWS + tx
            lx = fx.Float32(_ptr(T.f32, arg_lse, 4)[((lt < i32_m) & (tx < RROWS)).select(lt, fx.Int32(0)) * i32_slt + h * i32_slh])

            dr = buffer_ops.create_buffer_resource_from_addr(dst, num_records_bytes=rows * (WORLD * 256))
            red = fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, smem.red.ptr), fx.make_layout(4 * RROWS * PADN, 1)),
                fx.make_layout(4, 1),
            )
            atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            acc = [[Vec.filled(4, 0.0, fx.Float32) for nt in range_constexpr(4)] for mt in range_constexpr(mp)]
            for ks in range_constexpr(4):
                for mt in range_constexpr(mp):
                    for nt in range_constexpr(4):
                        acc[mt][nt] = Vec(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                Vec.make_type(4, fx.Float32),
                                [wv[ks][nt].bitcast(fx.BFloat16), xv[ks][mt].bitcast(fx.BFloat16), acc[mt][nt], 0, 0, 0],
                            )
                        )
            stamp(1)
            for mt in range_constexpr(mp):
                for nt in range_constexpr(4):
                    rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    rm.store(acc[mt][nt].bitcast(fx.Int32))
                    fx.copy(atom, rm, fx.slice(red, (None, (w * RROWS + mt * 16 + lm) * (PADN // 4) + nt * 4 + lg)))
            fx.gpu.barrier()
            row0 = tx // 8
            c8 = tx % 8
            for it in range_constexpr(-(-mp * 16 // (NT // 8))):
                row = it * (NT // 8) + row0
                if row < mp * 16:
                    t = p0 * 16 + row
                    flat = []
                    for q in range_constexpr(2):
                        s4 = None
                        for ww in range_constexpr(4):
                            rm = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(atom, fx.slice(red, (None, (ww * RROWS + row) * (PADN // 4) + c8 * 2 + q)), rm)
                            v4 = Vec(rm.load()).bitcast(fx.Float32)
                            s4 = v4 if s4 is None else s4 + v4
                        flat += [s4[0], s4[1], s4[2], s4[3]]
                    pk = [_pk_bf16(flat[2 * e], flat[2 * e + 1]) for e in range_constexpr(4)]
                    buffer_ops.buffer_store(
                        Vec.from_elements(pk, fx.Int32),
                        dr,
                        ((((i32_rank * HPR + j) * 2 + ns) * i32_mmax + t) * 32 + c8 * 4),
                        mask=(t < i32_m) & (abl & 2 == 0),
                        cache_modifier=SYS,
                    )

            if ns == 0:
                end_row = (i32_nseq > 0).select(er, fx.Int32(0x7FFFFFFF))
                smem.valid.ptr[tx] = (tx < end_row).select(fx.Int32(1), fx.Int32(0))
                fx.gpu.barrier()
                if live & (l0 == 0):
                    rr = s0
                    while rr < e0:
                        if rr < NT:
                            smem.valid.ptr[rr] = fx.Int32(0)
                        rr = rr + 1
                fx.gpu.barrier()
                if (tx < RROWS) & (lt < i32_m):
                    bad = (lx != lx) | (lx == fx.Float32(float("inf"))) | (fx.Int32(smem.valid.ptr[lt]) == 0)
                    lr = buffer_ops.create_buffer_resource_from_addr(dst + lse_off, num_records_bytes=rows * (WORLD * 4))
                    buffer_ops.buffer_store(
                        bad.select(neg_inf, lx), lr, ((lt * HPR + j) * WORLD + i32_rank), cache_modifier=SYS
                    )

            stamp(2)
            fx.rocdl.s_waitcnt(vmcnt=0)
            if const_expr(abl & 8):
                _llvm.FenceOp(_llvm.AtomicOrdering.release, syncscope="")
            fx.gpu.barrier()
            stamp(3)
            if (tx == 0) & True:
                fr = buffer_ops.create_buffer_resource_from_addr(ctl_dst, num_records_bytes=CTL_BYTES)
                buffer_ops.buffer_store(seq, fr, FLAG_WORD + i32_rank * per_src + idx, cache_modifier=SYS)

        if bid == base:
            seq = read_seq()
            for k in range_constexpr(-(-base // (NT * 4))):
                wid = (tx + k * NT) * 4
                lane_ok = wid < base
                addr = me_ctl + (START_WORD + lane_ok.select(wid, fx.Int32(0))) * 4
                fv = _volatile_load_v4(addr)
                pending = lane_ok & (
                    ((fv[0] - seq) < 0)
                    | ((wid + 1 < base) & ((fv[1] - seq) < 0))
                    | ((wid + 2 < base) & ((fv[2] - seq) < 0))
                    | ((wid + 3 < base) & ((fv[3] - seq) < 0))
                )
                while pending:
                    fv = _volatile_load_v4(addr)
                    pending = lane_ok & (
                        ((fv[0] - seq) < 0)
                        | ((wid + 1 < base) & ((fv[1] - seq) < 0))
                        | ((wid + 2 < base) & ((fv[2] - seq) < 0))
                        | ((wid + 3 < base) & ((fv[3] - seq) < 0))
                    )
            fx.gpu.barrier()
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, SEQ_WORD, cache_modifier=SYS)

        if not_comp:
            cb = lbid - (NC if order == 0 else UNITS)
            seq = read_seq()
            slot_off = fx.Int64(seq & 1) * i64_slot_bytes
            if tx == 0:
                buffer_ops.buffer_store(seq, ctl, START_WORD + bid, cache_modifier=SYS)
            stamp(0)
            mine = pool_me + slot_off
            dr = buffer_ops.create_buffer_resource_from_addr(mine, num_records_bytes=rows * (WORLD * 256))
            lr = buffer_ops.create_buffer_resource_from_addr(mine + lse_off, num_records_bytes=rows * (WORLD * 4))
            outr = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i32_m * (HPR * V * 2))
            if const_expr(order == 0):
                r0 = cb * ROWS_PER_WG
                pa = (r0 // HPR) // RROWS
                pb = ((r0 + ROWS_PER_WG - 1) // HPR // RROWS)
                pb = (pb * RROWS < i32_m).select(pb, pa)
            else:
                r0 = pas_w * (RROWS * HPR) + cb * ROWS_PER_WG
                pa = pas_w
                pb = pas_w
            sub = tx % 16
            total = i32_m * HPR
            gvs = []
            if const_expr(gated):
                gr = buffer_ops.create_buffer_resource_from_addr(arg_gate, num_records_bytes=i32_m * (HPR * V * 2))
                for it in range_constexpr(ROWS_PER_WG // 16):
                    row = r0 + it * 16 + tx // 16
                    gvs.append(Vec(buffer_ops.buffer_load(gr, row * 64 + sub * 4, vec_width=4, dtype=fx.Int32, mask=row < total)))

            if tx < 64:
                chunk_src = tx // 6
                chunk = tx % 6
                lane_ok = tx < WORLD * 6
                for pp in (pa, pb):
                    word = FLAG_WORD + (lane_ok).select(chunk_src, fx.Int32(0)) * per_src + pp * per_pass + (lane_ok).select(chunk, fx.Int32(0)) * 4
                    fv = _volatile_load_v4(me_ctl + word * 4)
                    pending = lane_ok & (((fv[0] - seq) < 0) | ((fv[1] - seq) < 0) | ((fv[2] - seq) < 0) | ((fv[3] - seq) < 0))
                    while pending:
                        fv = _volatile_load_v4(me_ctl + word * 4)
                        pending = lane_ok & (((fv[0] - seq) < 0) | ((fv[1] - seq) < 0) | ((fv[2] - seq) < 0) | ((fv[3] - seq) < 0))
            fx.gpu.barrier()
            stamp(1)
            loads = []
            for it in range_constexpr(ROWS_PER_WG // 16):
                row = r0 + it * 16 + tx // 16
                ok = (row < total) & (abl & 4 == 0)
                rt = row // HPR
                rj = row % HPR
                vs = [
                    Vec(
                        buffer_ops.buffer_load(
                            dr, ((((s * HPR + rj) * 2 + sub // 8) * i32_mmax + rt) * 32 + (sub % 8) * 4), vec_width=4, dtype=fx.Int32, mask=ok, cache_modifier=SYS
                        )
                    )
                    for s in range_constexpr(WORLD)
                ]
                l4 = [
                    Vec(buffer_ops.buffer_load(lr, row * WORLD + 4 * q, vec_width=4, dtype=fx.Float32, mask=ok, cache_modifier=SYS))
                    for q in range_constexpr(2)
                ]
                ls = [fx.Float32(l4[s // 4][s % 4]) for s in range_constexpr(WORLD)]
                loads.append((row, ok, vs, ls, gvs[it] if const_expr(gated) else None))
            stamp(2)
            for row, ok, vs, ls, gv in loads:
                m = ls[0]
                for s in range_constexpr(1, WORLD):
                    m = (ls[s] > m).select(ls[s], m)
                m = (m == neg_inf).select(fx.Float32(0.0), m)
                scale = fx.Float32(LOG2E if base_e else 1.0)
                es = [fx.Float32(fx.math.exp2((ls[s] - m) * scale)) for s in range_constexpr(WORLD)]
                den = es[0]
                for s in range_constexpr(1, WORLD):
                    den = den + es[s]
                acc = [fx.Float32(0.0)] * 8
                for s in range_constexpr(WORLD):
                    ws = es[s] / den
                    ws = (ws != ws).select(fx.Float32(0.0), ws)
                    for k in range_constexpr(4):
                        d = fx.Int32(vs[s][k])
                        lo = (d << 16).bitcast(fx.Float32)
                        hi = (d & fx.Int32(-65536)).bitcast(fx.Float32)
                        acc[2 * k] = acc[2 * k] + (ws == 0.0).select(fx.Float32(0.0), lo * ws)
                        acc[2 * k + 1] = acc[2 * k + 1] + (ws == 0.0).select(fx.Float32(0.0), hi * ws)
                pk = [_pk_bf16(acc[2 * k], acc[2 * k + 1]) for k in range_constexpr(4)]
                if const_expr(gated):
                    gated_pk = []
                    for k in range_constexpr(4):
                        a, g = fx.Int32(pk[k]), fx.Int32(gv[k])
                        outs = []
                        for x, y in (((a << 16), (g << 16)), ((a & fx.Int32(-65536)), (g & fx.Int32(-65536)))):
                            xf, yf = x.bitcast(fx.Float32), y.bitcast(fx.Float32)
                            outs.append(xf * (fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(-yf))))
                        gated_pk.append(_pk_bf16(outs[0], outs[1]))
                    pk = gated_pk
                buffer_ops.buffer_store(Vec.from_elements(pk, fx.Int32), outr, row * 64 + sub * 4, mask=ok)
            stamp(3)


    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        i32_sxt: fx.Int32,
        i32_sxh: fx.Int32,
        arg_lse: fx.Int64,
        i32_slt: fx.Int32,
        i32_slh: fx.Int32,
        arg_wt: fx.Int64,
        arg_seq: fx.Int64,
        arg_qsl: fx.Int64,
        i32_nseq: fx.Int32,
        arg_out: fx.Int64,
        arg_gate: fx.Int64,
        arg_tab: fx.Int64,
        i32_m: fx.Int32,
        i32_rank: fx.Int32,
        i32_mmax: fx.Int32,
        i64_slot_bytes: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            arg_x, i32_sxt, i32_sxh, arg_lse, i32_slt, i32_slh, arg_wt, arg_seq, arg_qsl, i32_nseq, arg_out,
            arg_gate, arg_tab, i32_m, i32_rank, i32_mmax, i64_slot_bytes,
        ).launch(grid=(grid, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
