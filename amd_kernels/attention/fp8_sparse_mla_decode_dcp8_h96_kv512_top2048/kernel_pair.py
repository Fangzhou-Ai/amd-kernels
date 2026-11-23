# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode of the DSA layers for 257..512 units (129..256 rows): the tile loop of ``kernel_tile.py`` in 256
workgroups that each run up to two units (row, 3 head groups) back to back.

Record, per-tile roles (QK waves 0-2, DMA wave 3, PV waves 4-7), numerics and preconditions are those of
``kernel_tile.py``: the per-unit arithmetic is the same instruction for instruction, so the output is bit for bit the tile
kernel's.

Grid: 256 workgroups, workgroup p on XCD p % 8 runs units ``(p % 8) * ceil(units / 8) + p // 8 + 32 k`` (k = 0, 1) of its
XCD's range, so a row's two head halves and the rows of a request share an XCD's L2 (a workgroup past the range
recomputes the last unit, as the tile kernel's spare workgroups do). The three-tile LDS ring continues
from the first unit into the second (which starts at slot ``tiles_0 % 3``), so the second unit skips the cold start of a
new workgroup:

  * every role reads both units' CSR bounds at the start, the DMA wave also the second unit's first indices, and it
    gathers the second unit's tile 0 in the first unit's last iteration (PV only; the slot was consumed three iterations
    earlier);
  * the QK waves issue the second unit's q loads before the first unit's closing barrier, so they fly while the PV waves
    run its last tile;
  * the QK and PV waves sync with a bare s_barrier after their LDS accesses, so the first unit's output stores do not
    hold up the second.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.warm_up import launcher

from ._primitives import (
    _L,
    HEADS,
    HG_HEADS,
    LN2,
    RECORD_BYTES,
    SCALE_CAP_BITS,
    XCDS,
    _BoundedI32,
    _cat,
    _cvt_v8,
    _fmax,
    _global_llvm_ptr,
    _guarded,
    _imin,
    _lds_llvm_ptr,
    _lds_load,
    _lds_store,
    _mfma_bf16,
    _mfma_f16,
    _mfma_fp8,
    _q_load,
    _q_prologue,
    _raw,
    _readfirstlane,
    _tr8,
    _xmax,
    _xsum,
)
from .kernel_row import q_prequant_load
from .kernel_tile import (
    AL_BUF,
    AL_OFF,
    BUF,
    CPW,
    DMA_WAVE,
    L_OFF,
    LDS_BYTES,
    MARGIN,
    NB,
    NH,
    NW,
    P_BUF,
    P_OFF,
    ROPE_OFF,
    SCL_OFF,
    SHIFT,
    SPLIT,
    THR,
    TT,
    _f32_ptr,
    _i32_ptr,
)

WGS = 256
WGS_PER_XCD = WGS // XCDS
MAX_UNITS = 2 * WGS


def tile_dma(ixs, bbase, kv_base, lane):
    """The 44 DMA operations of a tile (32 latent, 8 rope, 4 scale), all by the DMA wave (``kernel_tile.py``)."""
    ops = []
    for i in range_constexpr(32):
        x = 2 * i + (lane >> 5)
        ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(x * 4), _raw(ixs)))
        fs = (x & 15) ^ ((x >> 1) & 8)
        ops.append((kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(((lane & 31) ^ fs) * 16), bbase + i * 1024, 16))
    for kk in range_constexpr(8):
        x = 8 * kk + (lane >> 3)
        ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(x * 4), _raw(ixs)))
        ops.append(
            (
                kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(_L + 16 + ((lane & 7) ^ (lane >> 3)) * 16),
                bbase + ROPE_OFF + kk * 1024,
                16,
            )
        )
    for b in range_constexpr(4):
        ops.append((kv_base + fx.Int64(ixs) * RECORD_BYTES + fx.Int64(_L) + fx.Int64(4 * b), bbase + SCL_OFF + b * 256, 4))
    for src, dst, size in ops:
        rocdl.global_load_lds(_global_llvm_ptr(src), _lds_llvm_ptr(dst), size, 0)


def tile_qk_softmax(bb, m_old, l_old, pos0, r_len, q8, qs, q8l, qsl, qr, t, g, c_log2):
    """S of one 16-head group over the tile at ``bb`` and the online softmax step (``kernel_tile.py``):
    (m_new, l_new, alpha, P)."""
    neg_inf = fx.Float32(-1.0e30)

    def kread(s, j):
        c1 = 8 * (g // 2) + 2 * j + g % 2
        a1 = bb + (16 * s + t) * 512 + ((c1 ^ (t ^ ((s & 1) << 3))) << 4)
        return _cat(_lds_load(T.i32, 4, a1, 16), _lds_load(T.i32, 4, a1 + 256, 16), 4)

    ses = [
        (fx.Int32(fx.ptr_load(_i32_ptr(bb + SCL_OFF + (g * 64 + 16 * s + t) * 4), result_type=T.i32)) >> 23) & 0xFF
        for s in range_constexpr(4)
    ]
    sa = [Vec.filled(4, 0.0, fx.Float32) for _ in range(4)]
    for j in range_constexpr(4):
        kr = [kread(s, j) for s in range(4)]
        for s in range_constexpr(4):
            acc = _mfma_fp8(kr[s], q8[j], sa[s], ses[s], qs)
            sa[s] = _mfma_fp8(kr[s], q8l[j], acc, ses[s], qsl)
    for s in range_constexpr(4):
        tk = 16 * s + t
        ra = bb + ROPE_OFF + (tk // 8) * 1024 + (t % 8) * 128
        for i in range_constexpr(2):
            sa[s] = _mfma_bf16(_lds_load(T.i32, 4, ra + (((g + 4 * i) ^ (t % 8)) << 4), 16), qr[i], sa[s])
    sl = []
    tmax = neg_inf
    for s in range_constexpr(4):
        for r in range_constexpr(4):
            x = fx.Float32(Vec(sa[s])[r]) * c_log2
            x = ((pos0 + 16 * s + 4 * g + r) < r_len).select(x, neg_inf)
            sl.append(x)
            tmax = _fmax(tmax, x)
    tmax = _xmax(tmax)
    m_new = (tmax > m_old + THR).select(tmax + MARGIN, m_old)
    alpha = (m_old - m_new).exp2(fastmath="afn")
    m_cl = _fmax(m_new - SHIFT, fx.Float32(-1.0e20))
    pl = [(sl[k] - m_cl).exp2(fastmath="afn") for k in range(16)]
    psum = fx.Float32(0.0)
    for x in pl:
        psum = psum + x
    return m_new, l_old * alpha + psum, alpha, pl


@launcher
def compile_decode_pair(pre_quantized: bool = False):
    """The decode launcher (rows is a runtime argument)."""
    H = HG_HEADS
    PQ = pre_quantized
    name = "fp8_sparse_mla_decode_pair" + ("_pq" if PQ else "")

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kern(
        Q: fx.Tensor,
        KV: fx.Tensor,
        IDX: fx.Tensor,
        IPTR: fx.Tensor,
        OUT: fx.Tensor,
        LSE: fx.Tensor,
        stride_q: fx.Int32,
        stride_qh: fx.Int32,
        stride_o: fx.Int32,
        stride_oh: fx.Int32,
        c_log2: fx.Float32,
        n_q: fx.Int32,
        idx_bytes: fx.Int32,
    ):
        tid = fx.Int32(fx.thread_idx.x)
        w = _readfirstlane(tid // 64)
        lane = tid % 64
        t = lane % 16
        g = lane // 16
        pid = fx.Int32(fx.block_idx.x)
        units = n_q * SPLIT
        per_xcd = (units + (XCDS - 1)) // XCDS
        xcd = pid % XCDS
        slot_x = pid // XCDS
        u_lo = xcd * per_xcd
        cnt = _imin(per_xcd, units - u_lo)
        two = (cnt - slot_x) > WGS_PER_XCD

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_base = buf_base_i64(KV)
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))

        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        def unit(k):
            """(row, first head group) of this workgroup's unit k (0 or 1; past its units: the last unit, recomputed)."""
            u = _imin(u_lo + slot_x + k * WGS_PER_XCD, units - 1)
            return u // SPLIT, (u % SPLIT) * NH

        def row_bounds(qi):
            r_beg = _readfirstlane(ip_t[qi])
            r_end = _readfirstlane(ip_t[qi + 1])
            r_len = r_end - r_beg
            r_len = (r_len < 0).select(fx.Int32(0), r_len)
            return r_beg, r_len, _readfirstlane((r_len + (TT - 1)) // TT)

        def sync_tile():
            """The DMA wave's barrier: its tile has landed."""
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()

        def lds_barrier():
            """The QK / PV waves' barrier: their LDS accesses are done; unlike gpu.barrier's release fence it does not wait
            for their global stores (the previous unit's output)."""
            rocdl.s_waitcnt(lgkmcnt=0)
            llvm.inline_asm(None, [], "s_barrier", "~{memory}", has_side_effects=True)

        def slot_buf(s0, i):
            return base + ((s0 + i) % NB) * BUF

        neg_inf = fx.Float32(-1.0e30)
        units_of = [unit(0), unit(1)]

        def qk_role():
            def issue_q(k):
                """This lane's q loads of unit k (raw bf16, or the pre-quantized record)."""
                qi, hg0 = units_of[k]
                hq = (hg0 + w) * H + t
                if const_expr(PQ):
                    return q_prequant_load(q_t, (qi * stride_q + hq * stride_qh) // 4, g)
                q_row = (qi * stride_q + hq * stride_qh) // 2
                qr = [Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4)) for i in range_constexpr(2)]
                return _q_load(q_t, q_row, g), qr

            def finish_q(raw):
                """The MFMA operands of the loads ``issue_q`` started, as flat raw values."""
                if const_expr(PQ):
                    q8, qs, q8l, qsl, qr = raw
                else:
                    q8, qs, q8l, qsl = _q_prologue(raw[0], g)
                    qr = raw[1]
                return [_raw(Vec(v).ir_value()) for v in q8 + q8l + qr] + [_raw(fx.Int32(qs)), _raw(fx.Int32(qsl))]

            def unpack_q(st):
                q8 = [Vec(st[k]) for k in range(4)]
                q8l = [Vec(st[4 + k]) for k in range(4)]
                qr = [Vec(st[8 + k]) for k in range(2)]
                return q8, fx.Int32(st[10]), q8l, fx.Int32(st[11]), qr

            def run(ku, s0, bounds, qst):
                """Unit ku; returns the next unit's q (unit 0 only)."""
                qi, hg0 = units_of[ku]
                _, r_len, nt = bounds
                q8, qs, q8l, qsl, qr = unpack_q(qst)
                init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
                for tl, st in range(0, nt, 1, init=init):
                    i = fx.Int32(tl)
                    lds_barrier()
                    m_a, l_a, alpha_a, pl_a = tile_qk_softmax(
                        slot_buf(s0, i), fx.Float32(st[0]), fx.Float32(st[1]), i * TT, r_len, q8, qs, q8l, qsl, qr, t, g,
                        c_log2,
                    )
                    for kh in range_constexpr(2):
                        vals = [pl_a[(2 * kh + sp) * 4 + r] for sp in range(2) for r in range(4)]
                        hv = Vec.from_elements(vals, fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                        _lds_store(hv, T.i32, base + P_OFF + (i & 1) * P_BUF + (w * 2 + kh) * 1024 + lane * 16, 16)
                    _lds_store(
                        Vec.from_elements([alpha_a], fx.Float32),
                        T.f32,
                        base + AL_OFF + (i & 1) * AL_BUF + (w * 16 + t) * 4,
                        4,
                    )
                    fin = yield [m_a, l_a]
                lds_barrier()
                l_tot = _xsum(fx.Float32(fin[1]))
                _lds_store(Vec.from_elements([l_tot], fx.Float32), T.f32, base + L_OFF + (w * 16 + t) * 4, 4)
                has_l = l_tot > fx.Float32(0.0)
                lse2 = fx.Float32(fin[0]) + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0)))) - SHIFT
                lse = has_l.select(lse2 * LN2, fx.Float32(float("-inf")))
                lse_t.store(qi * HEADS + (hg0 + w) * H + t, lse.bitcast(fx.Int32))
                nxt = None
                if const_expr(ku == 0):
                    # the second unit's q loads fly while the PV waves run the last tile (the barrier waits for LDS only)
                    raw = issue_q(1)
                lds_barrier()
                if const_expr(ku == 0):
                    nxt = [_raw(Vec.filled(8, 0, fx.Int32).ir_value()) for _ in range(8)]
                    nxt += [_raw(Vec.filled(4, 0, fx.Int32).ir_value()) for _ in range(2)]
                    nxt += [_raw(fx.Int32(0).ir_value()) for _ in range(2)]
                    if two:
                        nxt = finish_q(raw)
                return nxt

            b0, b1 = row_bounds(units_of[0][0]), row_bounds(units_of[1][0])
            q1 = run(0, fx.Int32(0), b0, finish_q(issue_q(0)))
            _guarded(two, lambda: run(1, b0[2] % NB, b1, q1))

        def pv_role():
            pw = w - 4
            vi = t // 2
            vtok = 16 * (vi // 4) + 4 * g + vi % 4
            fv = (4 * g + vi % 4) ^ (8 * (vi // 4))
            v_off = vtok * 512 + (((CPW * pw) ^ fv) << 4) + (t % 2) * 8

            def run(ku, s0, nt):
                qi, hg0 = units_of[ku]
                init = [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(NH * CPW)]
                for tl, st in range(0, nt + 1, 1, init=init):
                    o = [Vec(st[k]) for k in range(NH * CPW)]
                    i = fx.Int32(tl)
                    lds_barrier()
                    if i >= 1:
                        tt_ = i - 1
                        bT = slot_buf(s0, tt_)
                        pb = tt_ & 1
                        alphas = [
                            fx.Float32(
                                fx.ptr_load(_f32_ptr(base + AL_OFF + pb * AL_BUF + (hl * 16 + t) * 4), result_type=T.f32)
                            )
                            for hl in range_constexpr(NH)
                        ]
                        scv = []
                        for kh in range_constexpr(2):
                            vals = []
                            for sp in range_constexpr(2):
                                v4 = _lds_load(T.i32, 4, bT + SCL_OFF + (pw * 64 + 32 * kh + 16 * sp + 4 * g) * 4, 16)
                                for r in range_constexpr(4):
                                    vals.append(_imin(fx.Int32(v4[r]), fx.Int32(SCALE_CAP_BITS)).bitcast(fx.Float32))
                            scv.append(Vec.from_elements(vals, fx.Float32).to(fx.Float16))
                        chg = alphas[0] != fx.Float32(1.0)
                        for hl in range_constexpr(1, NH):
                            chg = chg | (alphas[hl] != fx.Float32(1.0))
                        any_chg = rocdl.ballot(T.i64, _raw(chg.ir_value()))
                        if (tt_ > 0) & (fx.Int64(any_chg) != 0):
                            o = [
                                Vec.from_elements(
                                    [fx.Float32(Vec(o[k])[j]) * alphas[k // CPW] for j in range(4)], fx.Float32
                                )
                                for k in range(NH * CPW)
                            ]
                        for kh in range_constexpr(2):
                            pfk = []
                            for hl in range_constexpr(NH):
                                pf = _lds_load(
                                    T.i32, 4, base + P_OFF + pb * P_BUF + (hl * 2 + kh) * 1024 + lane * 16, 16
                                )
                                pfk.append((Vec(pf).bitcast(fx.Float16) * scv[kh]).bitcast(fx.Int32))
                            rd = [_tr8(bT + (v_off ^ (d * 16)), kh * 16384) for d in range(CPW)]
                            for d in range_constexpr(CPW):
                                a = _cvt_v8(rd[d])
                                for hl in range_constexpr(NH):
                                    o[hl * CPW + d] = _mfma_f16(a, pfk[hl], o[hl * CPW + d])
                    fin = yield o
                lds_barrier()
                for hl in range_constexpr(NH):
                    lv = fx.Float32(fx.ptr_load(_f32_ptr(base + L_OFF + (hl * 16 + t) * 4), result_type=T.f32))
                    inv_l = (lv > fx.Float32(0.0)).select(fx.Float32(1.0) / lv, fx.Float32(0.0))
                    o_row = (qi * stride_o + ((hg0 + hl) * H + t) * stride_oh) >> 1
                    for d in range_constexpr(CPW):
                        ov = Vec(fin[hl * CPW + d])
                        bf = Vec.from_elements([fx.Float32(ov[j]) * inv_l for j in range(4)], fx.Float32).to(
                            fx.BFloat16
                        )
                        o_t.store(o_row + 8 * (CPW * pw + d) + 2 * g, bf.bitcast(fx.Int32), vec_size=2)

            nt0 = row_bounds(units_of[0][0])[2]
            nt1 = row_bounds(units_of[1][0])[2]
            run(0, fx.Int32(0), nt0)
            _guarded(two, lambda: run(1, nt0 % NB, nt1))

        def dma_role():
            def first_indices(k):
                """CSR bounds of unit k, its padding slot and its tile 0 / tile 1 indices (lane l: key l of the tile)."""
                r_beg, r_len, nt = row_bounds(units_of[k][0])
                raw0 = fx.Int32(idx_t[r_beg + lane])
                # an empty row (CUDA-graph padding, or a rank with no keys) reads no record: slots[r_beg] is then past
                # the CSR end (stale / uninitialized), so pad with the always-valid record 0
                first_slot = (r_len > 0).select(_readfirstlane(raw0), fx.Int32(0))
                ix0 = (lane < r_len).select(raw0, first_slot)
                ix1 = fx.Int32(idx_t[r_beg + TT + lane])
                ix1 = (TT + lane < r_len).select(ix1, first_slot)
                return r_beg, r_len, nt, first_slot, ix0, ix1

            def run(s0, f, nxt):
                """One unit from its first indices ``f``; ``nxt``: the next unit's (tiles, tile-0 indices) to prefetch."""
                r_beg, r_len, nt, first_slot, _, ix1 = f
                b_unit1 = slot_buf(s0, nt)

                def ld_idx(p):
                    ix = fx.Int32(idx_t[r_beg + p])
                    return (p < r_len).select(ix, first_slot)

                for tl, st in range(0, nt + 1, 1, init=[_raw(fx.Int32(ix1).ir_value())]):
                    ixn = fx.Int32(st[0])
                    i = fx.Int32(tl)
                    sync_tile()
                    ixnn = ld_idx((i + 2) * TT + lane)
                    b_next = slot_buf(s0, i + 1)
                    _guarded(i + 1 < nt, lambda _i=ixn, _b=b_next: tile_dma(_i, _b, kv_base, lane))
                    if const_expr(nxt is not None):
                        _guarded(
                            (i == nt) & two & (nxt[0] > 0),
                            lambda _b=b_unit1: tile_dma(nxt[1], _b, kv_base, lane),
                        )
                    _fin = yield [ixnn]
                gpu.barrier()

            f0 = first_indices(0)
            _guarded(f0[2] > 0, lambda: tile_dma(f0[4], base, kv_base, lane))
            f1 = first_indices(1)
            run(fx.Int32(0), f0, (f1[2], f1[4]))
            _guarded(two, lambda: run(f0[2] % NB, f1, None))

        _guarded(w < NH, qk_role)
        _guarded(w >= 4, pv_role)
        _guarded(w == DMA_WAVE, dma_role)

    @flyc.jit
    def launch_decode(
        Q,
        KV,
        IDX,
        IPTR,
        OUT,
        LSE,
        n_q,
        stride_q,
        stride_qh,
        stride_o,
        stride_oh,
        c_log2,
        idx_bytes,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(Q, KV, IDX, IPTR, OUT, LSE, stride_q, stride_qh, stride_o, stride_oh, c_log2, n_q, idx_bytes).launch(
            grid=(WGS, 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_decode.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch_decode
