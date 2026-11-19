# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode, tile-loop variant: one workgroup per (row, NH head groups), the KV tile shared by all of them."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.warm_up import launcher

from .kernel_wr import q_prequant_load
from .kernel import (
    HEADS,
    HG_HEADS,
    N_HG,
    RECORD_BYTES,
    _BoundedI32,
    _L,
    _LN2,
    _SCALE_CAP_BITS,
    _XCDS,
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

TT = 64
THR = 3.0
SHIFT = 5.0
ISTR = 1056
LAT_B = (TT // 2) * ISTR
SCL_OFF = LAT_B
ROPE_OFF = SCL_OFF + 4 * TT * 4
BUF = ROPE_OFF + (TT // 8) * ISTR


def _now():
    return fx.Int32(fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])))


def _pf32(addr):
    return fx.inttoptr(fx.PointerType.get(T.f32, fx.AddressSpace.Shared, 4), fx.Int32(addr))


def _pi32(addr):
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Shared, 4), fx.Int32(addr))


@launcher
def compile_decode_kt(nhg: int = 6, timing: bool = False, opts: tuple = ()):
    NH = nhg
    NW = 8 if "w8" in opts else 4
    NB = 2 if "nb2" in opts else 3
    PQ = "pq" in opts
    NOQK = "noqk" in opts
    NOPV = "nopv" in opts
    NODMA = "nodma" in opts
    assert NH in (3, 6)
    SPLIT = N_HG // NH
    TWO = NW == 4 and NH > 4
    GUARD_QK = NW > NH
    CPW = 32 // NW
    NLAT = 32 // NW
    NROPE = 8 // NW
    H = HG_HEADS
    P_OFF = NB * BUF
    AL_OFF = P_OFF + NH * 2 * 1024
    L_OFF = AL_OFF + 96 * 4
    lds_bytes = L_OFF + 96 * 4 + 64
    name = f"fp8_sparse_mla_decode_kt_h{NH}" + "".join("_" + o for o in sorted(opts))

    @fx.struct
    class Smem:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

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
        per_xcd = (units + (_XCDS - 1)) // _XCDS
        u = (pid % _XCDS) * per_xcd + pid // _XCDS
        u = (u < units).select(u, units - 1)
        qi = u // SPLIT
        hg0 = (u % SPLIT) * NH

        q_t = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_base = buf_base_i64(KV)
        idx_t = _BoundedI32(IDX, idx_bytes)
        ip_t = GTensor(IPTR, dtype=T.i32, shape=(-1,))
        o_t = GTensor(OUT, dtype=T.i32, shape=(-1,))
        lse_t = GTensor(LSE, dtype=T.i32, shape=(-1,))

        def stamp(k):
            if const_expr(timing):

                def _s(_k=k):
                    lse_t.store(n_q * HEADS + u * 16 + _k, _now())

                _guarded(tid == 0, _s)

        stamp(0)
        lds = fx.SharedAllocator().allocate(Smem).peek()
        base = _readfirstlane(fx.ptrtoint(lds.raw.ptr))

        r_beg = _readfirstlane(ip_t[qi])
        r_end = _readfirstlane(ip_t[qi + 1])
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)
        nt = _readfirstlane((r_len + (TT - 1)) // TT)

        hl_a = w if NW >= NH else _imin(w, fx.Int32(NH - 1))
        hl_b = 4 + (w % 2)

        raw0 = fx.Int32(idx_t[r_beg + lane])
        first_slot = _readfirstlane(raw0)

        def ld_idx(p):
            ix = fx.Int32(idx_t[r_beg + p])
            return (p < r_len).select(ix, first_slot)

        ix_p = [(lane < r_len).select(raw0, first_slot)] + [ld_idx(TT * k + lane) for k in range(1, NB)]
        stamp(1)

        def dma_addrs(ixs):
            out = []
            for k in range_constexpr(NLAT):
                i = NW * k + w
                x = 2 * i + (lane >> 5)
                ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(x * 4), _raw(ixs)))
                out.append(kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64((lane & 31) * 16))
            for k in range_constexpr(NROPE):
                kk = w + NW * k
                x = 8 * kk + (lane >> 3)
                ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(x * 4), _raw(ixs)))
                out.append(kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(_L + 16 + (lane & 7) * 16))
            out.append(kv_base + fx.Int64(ixs) * RECORD_BYTES + fx.Int64(_L) + fx.Int64(4 * (w % 4)))
            return out

        def dma_issue(addrs, bbase):
            for k in range_constexpr(NLAT):
                rocdl.global_load_lds(
                    _global_llvm_ptr(addrs[k]), _lds_llvm_ptr(bbase + (NW * k + w) * ISTR), 16, 0
                )
            for k in range_constexpr(NROPE):
                rocdl.global_load_lds(
                    _global_llvm_ptr(addrs[NLAT + k]),
                    _lds_llvm_ptr(bbase + ROPE_OFF + (w + NW * k) * ISTR),
                    16,
                    0,
                )

            def _scl():
                rocdl.global_load_lds(
                    _global_llvm_ptr(addrs[NLAT + NROPE]), _lds_llvm_ptr(bbase + SCL_OFF + (w % 4) * 256), 4, 0
                )

            if const_expr(NW == 4):
                _scl()
            else:
                _guarded(w < 4, _scl)

        def dma_tile(ixs, bbase):
            dma_issue(dma_addrs(ixs), bbase)

        adr_p = [dma_addrs(ix_p[k]) for k in range(NB - 1)]
        for k in range_constexpr(NB - 1):
            _guarded(fx.Int32(k) < nt, lambda _a=adr_p[k], _b=base + k * BUF: dma_issue(_a, _b))
        stamp(2)

        def q_slot(hl):
            hq = (hg0 + hl) * H + t
            if const_expr(PQ):
                return q_prequant_load(q_t, (qi * stride_q + hq * stride_qh) // 4, g)
            q_row = (qi * stride_q + hq * stride_qh) // 2
            raw = _q_load(q_t, q_row, g)
            qr = [Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4)) for i in range_constexpr(2)]
            return raw, qr

        if const_expr(PQ):
            q8a, qsa, q8la, qsla, qr_a = q_slot(_imin(hl_a, fx.Int32(NH - 1)))
            if const_expr(TWO):
                q8b, qsb, q8lb, qslb, qr_b = q_slot(hl_b)
            stamp(3)
        else:
            raw_a, qr_a = q_slot(_imin(hl_a, fx.Int32(NH - 1)))
            if const_expr(TWO):
                raw_b, qr_b = q_slot(hl_b)
            stamp(3)
            q8a, qsa, q8la, qsla = _q_prologue(raw_a, g)
            if const_expr(TWO):
                q8b, qsb, q8lb, qslb = _q_prologue(raw_b, g)
        stamp(4)

        neg_inf = fx.Float32(-1.0e30)

        def kread(bb, s, j):
            tk = 16 * s + t
            c1 = 8 * (g // 2) + 2 * j + g % 2
            a1 = bb + (tk // 2) * ISTR + (tk % 2) * 512 + (c1 << 4)
            return _cat(_lds_load(T.i32, 4, a1, 16), _lds_load(T.i32, 4, a1 + 256, 16), 4)

        vi = t // 2
        vtok = 16 * (vi // 4) + 4 * g + vi % 4
        v_off = (vtok // 2) * ISTR + (vtok % 2) * 512 + (t % 2) * 8

        def softmax_slot(sacc, m_old, l_old, pos0):
            sl = []
            tmax = neg_inf
            for s in range_constexpr(4):
                for r in range_constexpr(4):
                    x = fx.Float32(Vec(sacc[s])[r]) * c_log2
                    x = ((pos0 + 16 * s + 4 * g + r) < r_len).select(x, neg_inf)
                    sl.append(x)
                    tmax = _fmax(tmax, x)
            tmax = _xmax(tmax)
            upd = tmax > m_old + THR
            m_new = upd.select(tmax, m_old)
            alpha = (m_old - m_new).exp2(fastmath="afn")
            m_cl = _fmax(m_new - SHIFT, fx.Float32(-1.0e20))
            pl = [(sl[k] - m_cl).exp2(fastmath="afn") for k in range(16)]
            psum = fx.Float32(0.0)
            for x in pl:
                psum = psum + x
            l_new = l_old * alpha + psum
            return m_new, l_new, alpha, pl

        def p_store(hl, pl, alpha):
            for kk in range_constexpr(2):
                vals = [pl[(2 * kk + sp) * 4 + r] for sp in range(2) for r in range(4)]
                hv = Vec.from_elements(vals, fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                _lds_store(hv, T.i32, base + P_OFF + (hl * 2 + kk) * 1024 + lane * 16, 16)
            _lds_store(Vec.from_elements([alpha], fx.Float32), T.f32, base + AL_OFF + (hl * 16 + t) * 4, 4)

        def qk_slot(bb, q8, qs, q8l, qsl, qr, m_old, l_old, pos0, ses):
            sa = [Vec.filled(4, 0.0, fx.Float32) for _ in range(4)]
            if const_expr(not NOQK):
                for j in range_constexpr(4):
                    kr = [kread(bb, s, j) for s in range(4)]
                    for s in range_constexpr(4):
                        acc = _mfma_fp8(kr[s], q8[j], sa[s], ses[s], qs)
                        sa[s] = _mfma_fp8(kr[s], q8l[j], acc, ses[s], qsl)
                for s in range_constexpr(4):
                    tk = 16 * s + t
                    ra = bb + ROPE_OFF + (tk // 8) * ISTR + (tk % 8) * 128 + 16 * g
                    for i in range_constexpr(2):
                        sa[s] = _mfma_bf16(_lds_load(T.i32, 4, ra + 64 * i, 16), qr[i], sa[s])
            return softmax_slot(sa, m_old, l_old, pos0)

        init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
        if const_expr(TWO):
            init += [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
        ns = len(init)
        init += [_raw(fx.Int32(ix_p[NB - 1]).ir_value())]
        init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(NH * CPW)]
        stamp(5)
        for tl, st in range(0, nt, 1, init=init):
            m_a = fx.Float32(st[0])
            l_a = fx.Float32(st[1])
            if const_expr(TWO):
                m_b = fx.Float32(st[2])
                l_b = fx.Float32(st[3])
            ixn = fx.Int32(st[ns])
            o = [Vec(st[ns + 1 + i]) for i in range(NH * CPW)]
            tli = fx.Int32(tl)
            bb = base + (tli % NB) * BUF
            if const_expr(NB == 2):
                rocdl.s_waitcnt(vmcnt=0)
            else:
                if tli + 1 < nt:
                    if const_expr(NW == 4):
                        rocdl.s_waitcnt(vmcnt=NLAT + NROPE + 1)
                    else:
                        if w < 4:
                            rocdl.s_waitcnt(vmcnt=NLAT + NROPE + 1)
                        else:
                            rocdl.s_waitcnt(vmcnt=NLAT + NROPE)
                else:
                    rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            stamp(6)

            ixnn = ld_idx((tli + NB) * TT + lane)
            if const_expr(not NODMA):
                _guarded(
                    tli + (NB - 1) < nt,
                    lambda _i=ixn, _b=base + ((tli + (NB - 1)) % NB) * BUF: dma_tile(_i, _b),
                )

            ses = []
            for s in range_constexpr(4):
                scg = fx.Int32(
                    fx.ptr_load(_pi32(bb + SCL_OFF + (g * 64 + 16 * s + t) * 4), result_type=T.i32)
                )
                ses.append((scg >> 23) & 0xFF)

            pos0 = tli * TT
            alpha_a = fx.Float32(1.0)
            if const_expr(GUARD_QK):
                if w < NH:
                    m_a, l_a, alpha_a, pl_a = qk_slot(bb, q8a, qsa, q8la, qsla, qr_a, m_a, l_a, pos0, ses)
                    p_store(hl_a, pl_a, alpha_a)
            else:
                m_a, l_a, alpha_a, pl_a = qk_slot(bb, q8a, qsa, q8la, qsla, qr_a, m_a, l_a, pos0, ses)
                p_store(hl_a, pl_a, alpha_a)
            if const_expr(TWO):
                m_b, l_b, alpha_b, pl_b = qk_slot(bb, q8b, qsb, q8lb, qslb, qr_b, m_b, l_b, pos0, ses)
                p_store(hl_b, pl_b, alpha_b)
            stamp(7)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            stamp(8)

            # PV: wave w owns dim chunks CPW*w .. CPW*w + CPW - 1 (16 dims each) of all NH head groups
            alphas = [
                fx.Float32(fx.ptr_load(_pf32(base + AL_OFF + (hl * 16 + t) * 4), result_type=T.f32))
                for hl in range_constexpr(NH)
            ]
            blk = (CPW * w) // 8
            scv = []
            for kk in range_constexpr(2):
                vals = []
                for sp in range_constexpr(2):
                    v4 = _lds_load(T.i32, 4, bb + SCL_OFF + (blk * 64 + 32 * kk + 16 * sp + 4 * g) * 4, 16)
                    for r in range_constexpr(4):
                        vals.append(_imin(fx.Int32(v4[r]), fx.Int32(_SCALE_CAP_BITS)).bitcast(fx.Float32))
                scv.append(Vec.from_elements(vals, fx.Float32).to(fx.Float16))
            chg = alphas[0] != fx.Float32(1.0)
            for hl in range_constexpr(1, NH):
                chg = chg | (alphas[hl] != fx.Float32(1.0))
            any_chg = rocdl.ballot(T.i64, _raw(chg.ir_value()))
            if (tli > 0) & (fx.Int64(any_chg) != 0):
                o = [
                    Vec.from_elements([fx.Float32(Vec(o[i])[j]) * alphas[i // CPW] for j in range(4)], fx.Float32)
                    for i in range(NH * CPW)
                ]
            for kk in range_constexpr(0 if NOPV else 2):
                pfk = []
                for hl in range_constexpr(NH):
                    pf = _lds_load(T.i32, 4, base + P_OFF + (hl * 2 + kk) * 1024 + lane * 16, 16)
                    pfk.append((Vec(pf).bitcast(fx.Float16) * scv[kk]).bitcast(fx.Int32))
                rd = [_tr8(bb + v_off + (CPW * w + d) * 16, kk * 16 * ISTR) for d in range(CPW)]
                for d in range_constexpr(CPW):
                    a = _cvt_v8(rd[d])
                    for hl in range_constexpr(NH):
                        o[hl * CPW + d] = _mfma_f16(a, pfk[hl], o[hl * CPW + d])
            stamp(9)
            fin_state = [m_a, l_a]
            if const_expr(TWO):
                fin_state += [m_b, l_b]
            fin = yield fin_state + [ixnn] + o

        def slot_final(hl, m, l):
            l_tot = _xsum(fx.Float32(l))
            _lds_store(Vec.from_elements([l_tot], fx.Float32), T.f32, base + L_OFF + (hl * 16 + t) * 4, 4)
            has_l = l_tot > fx.Float32(0.0)
            lse2 = fx.Float32(m) + fx.Float32(fx.log2(has_l.select(l_tot, fx.Float32(1.0)))) - SHIFT
            lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
            lse_t.store(qi * HEADS + (hg0 + hl) * H + t, lse.bitcast(fx.Int32))

        if const_expr(GUARD_QK):
            if w < NH:
                slot_final(hl_a, fin[0], fin[1])
        else:
            slot_final(hl_a, fin[0], fin[1])
        if const_expr(TWO):
            slot_final(hl_b, fin[2], fin[3])
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        for hl in range_constexpr(NH):
            lv = fx.Float32(fx.ptr_load(_pf32(base + L_OFF + (hl * 16 + t) * 4), result_type=T.f32))
            inv_l = (lv > fx.Float32(0.0)).select(fx.Float32(1.0) / lv, fx.Float32(0.0))
            for d in range_constexpr(CPW):
                ov = Vec(fin[ns + 1 + hl * CPW + d])
                bf = Vec.from_elements([fx.Float32(ov[j]) * inv_l for j in range(4)], fx.Float32).to(fx.BFloat16)
                o_t.store(
                    (qi * stride_o + ((hg0 + hl) * H + t) * stride_oh + 16 * (CPW * w + d) + 4 * g) // 2,
                    bf.bitcast(fx.Int32),
                    vec_size=2,
                )
        stamp(10)

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
            grid=(fx.Int64(((n_q * SPLIT + _XCDS - 1) // _XCDS) * _XCDS), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_decode.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch_decode
