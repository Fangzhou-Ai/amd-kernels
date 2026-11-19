# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode, whole-row variant: one workgroup per (row, head group) holds the row's KV in LDS at once."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels.tensor_shim import GTensor, buf_base_i64
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.tensor_shim import buf_load_scalar
from amd_kernels.warm_up import launcher

from .kernel import (
    HEADS,
    HG_HEADS,
    N_HG,
    RECORD_BYTES,
    _BoundedI32,
    _L,
    _LN2,
    _P_SHIFT,
    _SCALE_CAP_BITS,
    _XCDS,
    _cat,
    _cvt_v8,
    _fmax,
    _global_i32,
    _global_llvm_ptr,
    _guarded,
    _imin,
    _lds_llvm_ptr,
    _lds_load,
    _lds_store,
    _lo_scale,
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

RT = 288
NSUB = RT // 16
KKN = RT // 32
LAT_B = RT * 512
P_OFF = LAT_B
SC_OFF = P_OFF + KKN * 1024
EX_OFF = SC_OFF + 4 * RT * 2
LDS_BYTES = max(EX_OFF + 1024, P_OFF + 16384)


def _now():
    return fx.Int32(fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])))


def _ptr(elem, addr, align):
    return fx.inttoptr(fx.PointerType.get(elem, fx.AddressSpace.Shared, align), fx.Int32(addr))


def q_prequant_load(q_t, q_row, g):
    """Pre-quantized q record (``reference.quantize_q`` layout) of this lane's head at i32 offset ``q_row``:
    (hi operands, hi E8M0, lo operands, lo E8M0, rope vectors)."""
    q8, q8l = [], []
    for j in range(4):
        c = 2 * j + g % 2
        for lo, dst in ((0, q8), (128, q8l)):
            a = Vec(q_t.load(q_row + lo + (g // 2) * 32 + c * 4, vec_size=4))
            b = Vec(q_t.load(q_row + lo + (2 + g // 2) * 32 + c * 4, vec_size=4))
            dst.append(_cat(a, b, 4))
    sw = fx.Int32(q_t[q_row + 288])
    qs = (sw >> (8 * g)) & 0xFF
    qr = [Vec(q_t.load(q_row + 256 + 16 * i + g * 4, vec_size=4)) for i in range(2)]
    return q8, qs, q8l, _lo_scale(qs), qr


def q_staged_frags(base, q_off, t, g):
    """hi / lo operands of this lane's head from the staged q record (head t at ``q_off + 1024 t``, chunk c at c ^ t)."""
    a0 = t * 1024 + (((8 * (g // 2) + g % 2) ^ t) << 4)
    q8, q8l = [], []
    for j in range(4):
        a = base + q_off + (a0 ^ (32 * j))
        q8.append(_cat(_lds_load(T.i32, 4, a, 16), _lds_load(T.i32, 4, a + 256, 16), 4))
        q8l.append(_cat(_lds_load(T.i32, 4, a + 512, 16), _lds_load(T.i32, 4, a + 768, 16), 4))
    return q8, q8l


@launcher
def compile_decode_wr(timing: bool = False, opts: tuple = ()):
    H = HG_HEADS
    PQ = "pq" in opts or "qs" in opts
    NOG = "nog" in opts
    SX = "sx" in opts
    QL = "ql" in opts
    QS = "qs" in opts
    NW = 8 if "w8" in opts else 4
    NSLOT = (NSUB + NW - 1) // NW
    CPW = 32 // NW
    NOQK = "noqk" in opts
    NOPV = "nopv" in opts
    NODMA = "nodma" in opts
    NOQ = "noq" in opts
    NOROPE = "norope" in opts
    QF = "qf" in opts
    LD = "ld" in opts
    name = "fp8_sparse_mla_decode_wr" + "".join("_" + o for o in sorted(opts)) + ""

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
        units = n_q * N_HG
        per_xcd = (units + (_XCDS - 1)) // _XCDS
        u = (pid % _XCDS) * per_xcd + pid // _XCDS
        u = (u < units).select(u, units - 1)
        qi = u // N_HG
        hg = u % N_HG

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

        if const_expr(SX):
            r_beg = fx.Int32(buf_load_scalar(ip_t.rsrc, qi, 1))
            r_end = fx.Int32(buf_load_scalar(ip_t.rsrc, qi + 1, 1))
        else:
            r_beg = _readfirstlane(ip_t[qi])
            r_end = _readfirstlane(ip_t[qi + 1])
        r_len = r_end - r_beg
        r_len = (r_len < 0).select(fx.Int32(0), r_len)
        npass = _readfirstlane((r_len + (RT - 1)) // RT)

        hq = hg * H + t

        q_base_g = buf_base_i64(Q)

        def load_q():
            if const_expr(NOQ):
                z8 = Vec.filled(8, 0, fx.Int32)
                return [z8] * 4, fx.Int32(120), [z8] * 4, fx.Int32(115), [Vec.filled(4, 0, fx.Int32)] * 2, None
            if const_expr(QS):
                q_row8 = (qi * stride_q + hq * stride_qh) // 4
                sw = fx.Int32(q_t[q_row8 + 288])
                qs_ = (sw >> (8 * g)) & 0xFF
                qr_ = [Vec(q_t.load(q_row8 + 256 + 16 * i + g * 4, vec_size=4)) for i in range(2)]
                return None, qs_, None, _lo_scale(qs_), qr_, None
            if const_expr(PQ):
                q8, qs, q8l, qsl, qr = q_prequant_load(q_t, (qi * stride_q + hq * stride_qh) // 4, g)
                return q8, qs, q8l, qsl, qr, None
            q_row = (qi * stride_q + hq * stride_qh) // 2
            raw_q = _q_load(q_t, q_row, g)
            qr = [Vec(q_t.load(q_row + (_L + 32 * i) // 2 + g * 4, vec_size=4)) for i in range_constexpr(2)]
            return None, None, None, None, qr, raw_q

        def stage_q():
            for k in range_constexpr(16 // NW):
                i = w + NW * k
                src = _global_llvm_ptr(
                    q_base_g
                    + fx.Int64(qi * stride_q + (hg * H + i) * stride_qh)
                    + fx.Int64(((lane ^ i) & 63) * 16)
                )
                rocdl.global_load_lds(src, _lds_llvm_ptr(base + P_OFF + i * 1024), 16, 0)

        if const_expr(not QL):
            q8, qs, q8l, qsl, qr, raw_q = load_q()
        if const_expr(QF):
            rocdl.sched_barrier(0)

        if const_expr(SX):
            first_slot = fx.Int32(buf_load_scalar(idx_t.rsrc, r_beg, 1))
        else:
            raw0 = fx.Int32(idx_t[r_beg + lane])
            first_slot = _readfirstlane(raw0)

        def ld_idx(p):
            ix = fx.Int32(idx_t[r_beg + p])
            return (p < r_len).select(ix, first_slot)

        stamp(1)
        neg_inf = fx.Float32(-1.0e30)

        c0 = 8 * (g // 2) + g % 2
        vi = t // 2
        vtok = 16 * (vi // 4) + 4 * g + vi % 4
        fv = (4 * g + vi % 4) ^ (8 * (vi // 4))
        v_off = vtok * 512 + (((CPW * w) ^ fv) << 4) + (t % 2) * 8

        ssl = [w + NW * k for k in range(NSLOT)]
        valid_last = (w + NW * (NSLOT - 1)) < NSUB

        def dma_addrs(s, ixs):
            """Global addresses of the 8 DMA operations of subtile s (the slot lookups run here, outside any branch)."""
            par = s & 1
            out = []
            if const_expr(SX):
                for m in range_constexpr(8):
                    tok_lo = 2 * m + (lane >> 5)
                    f = tok_lo ^ (par << 3)
                    ix = (lane < 32).select(ixs[2 * m], ixs[2 * m + 1])
                    out.append(kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(((lane & 31) ^ f) * 16))
                return out
            for m in range_constexpr(8):
                tok_lo = 2 * m + (lane >> 5)
                f = tok_lo ^ (par << 3)
                ix = fx.Int32(rocdl.ds_bpermute(T.i32, _raw(tok_lo * 4), _raw(ixs)))
                out.append(kv_base + fx.Int64(ix) * RECORD_BYTES + fx.Int64(((lane & 31) ^ f) * 16))
            return out

        def dma_group(s, addrs):
            for m in range_constexpr(8):
                rocdl.global_load_lds(_global_llvm_ptr(addrs[m]), _lds_llvm_ptr(base + (16 * s + 2 * m) * 512), 16, 0)

        def side_loads(ixs):
            row = kv_base + fx.Int64(ixs) * RECORD_BYTES
            scl = fx.Int32(_global_i32(row + fx.Int64(_L + 4 * g), 1))
            if const_expr(NOROPE):
                rp = [Vec.filled(4, 0, fx.Int32)] * 2
            else:
                rp = [_global_i32(row + fx.Int64(_L + 16 + 16 * g + 64 * i), 4) for i in range(2)]
            return scl, rp

        def qk_subtile(s, scl, rp):
            if const_expr(NOQK):
                return Vec.filled(4, 0.0, fx.Float32)
            tokl = 16 * s + t
            fk = t ^ ((s & 1) << 3)
            koff = tokl * 512 + ((c0 ^ fk) << 4)
            ses = (scl >> 23) & 0xFF
            sacc = Vec.filled(4, 0.0, fx.Float32)
            for j in range_constexpr(4):
                a1 = base + (koff ^ (32 * j))
                kf = _cat(_lds_load(T.i32, 4, a1, 16), _lds_load(T.i32, 4, a1 + 256, 16), 4)
                acc = _mfma_fp8(kf, q8[j], sacc, ses, qs)
                sacc = _mfma_fp8(kf, q8l[j], acc, ses, qsl)
            for i in range_constexpr(2):
                sacc = _mfma_bf16(rp[i], qr[i], sacc)
            return sacc

        def pass_vals(cb):
            len_p = _imin(r_len - cb, fx.Int32(RT))
            if const_expr(NOG):
                vals = [None] * (NSLOT - 1) + [valid_last]
            else:
                vals = [(32 * (ssl[k] // 2)) < len_p for k in range(NSLOT)]
                vals[NSLOT - 1] = vals[NSLOT - 1] & valid_last
            return len_p, vals

        def maybe(pred, fn):
            if pred is None:
                fn()
            else:
                _guarded(pred, fn)

        def issue_pass(cb):
            """DMA of the pass's tiles into LDS plus the scale / rope loads; returns (scales, rope vectors)."""
            _, vals = pass_vals(cb)
            if const_expr(SX):
                addrs = []
                for k in range_constexpr(NSLOT):
                    raw = Vec(buf_load_scalar(idx_t.rsrc, r_beg + cb + 16 * ssl[k], 16))
                    ixv = [((cb + 16 * ssl[k] + j) < r_len).select(fx.Int32(raw[j]), first_slot) for j in range(16)]
                    addrs.append(dma_addrs(ssl[k], ixv))
            else:
                ixs_l = [ld_idx(cb + 16 * ssl[k] + t) for k in range(NSLOT)]
                addrs = [dma_addrs(ssl[k], ixs_l[k]) for k in range(NSLOT)]
            stamp(9)
            if const_expr(LD):
                data = [[_global_i32(addrs[k][m], 4) for m in range(8)] for k in range(NSLOT)]
                for k in range_constexpr(NSLOT):
                    def _wr(_k=k):
                        for m in range_constexpr(8):
                            _lds_store(data[_k][m], T.i32, base + (16 * ssl[_k] + 2 * m) * 512 + lane * 16, 16)

                    maybe(vals[k], _wr)
            else:
                for k in range_constexpr(0 if NODMA else NSLOT):
                    maybe(vals[k], lambda _s=ssl[k], _a=addrs[k]: dma_group(_s, _a))
            stamp(10)
            if const_expr(SX):
                ixs_l = [ld_idx(cb + 16 * ssl[k] + t) for k in range(NSLOT)]
            scs, rps = [], []
            for k in range_constexpr(NSLOT):
                scl_k = fx.Int32(0x3F800000)
                rp_k = [Vec.filled(4, 0, fx.Int32), Vec.filled(4, 0, fx.Int32)]
                if const_expr(vals[k] is None):
                    scl_k, rp_k = side_loads(ixs_l[k])
                else:
                    if vals[k]:
                        scl_k, rp_k = side_loads(ixs_l[k])
                scs.append(scl_k)
                rps += rp_k
            stamp(11)
            return scs, rps

        sc0, rp0 = issue_pass(fx.Int32(0))
        stamp(2)
        if const_expr(QS):
            stage_q()
        if const_expr(QL):
            q8, qs, q8l, qsl, qr, raw_q = load_q()
        if const_expr(QS):
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            q8, q8l = q_staged_frags(base, P_OFF, t, g)
        if const_expr(not PQ and not NOQ):
            q8, qs, q8l, qsl = _q_prologue(raw_q, g)
        stamp(3)

        init = [_raw(neg_inf.ir_value()), _raw(fx.Float32(0.0).ir_value())]
        init += [_raw(Vec.filled(4, 0.0, fx.Float32).ir_value()) for _ in range(CPW)]
        init += [_raw(fx.Int32(x).ir_value()) for x in sc0]
        init += [_raw(Vec(x).ir_value()) for x in rp0]
        for pas, st in range(0, npass, 1, init=init):
            m_run = fx.Float32(st[0])
            l_run = fx.Float32(st[1])
            o = [Vec(st[2 + i]) for i in range(CPW)]
            scs = [fx.Int32(st[2 + CPW + i]) for i in range(NSLOT)]
            rps = [Vec(st[2 + CPW + NSLOT + i]) for i in range(2 * NSLOT)]
            cb = fx.Int32(pas) * RT
            len_p, vals = pass_vals(cb)

            rocdl.s_waitcnt(vmcnt=0)
            stamp(4)

            sacc_all = []
            for k in range_constexpr(NSLOT):
                sacc_k = Vec.filled(4, 0.0, fx.Float32)
                if const_expr(vals[k] is None):
                    sacc_k = qk_subtile(ssl[k], scs[k], rps[2 * k : 2 * k + 2])
                else:
                    if vals[k]:
                        sacc_k = qk_subtile(ssl[k], scs[k], rps[2 * k : 2 * k + 2])
                sacc_all.append(sacc_k)
            stamp(5)

            tmax = neg_inf
            sl = []
            for k in range_constexpr(NSLOT):
                for r in range_constexpr(4):
                    x = fx.Float32(Vec(sacc_all[k])[r]) * c_log2
                    ok = (cb + 16 * ssl[k] + 4 * g + r) < r_len
                    if k == NSLOT - 1:
                        ok = ok & valid_last
                    x = ok.select(x, neg_inf)
                    sl.append(x)
                    tmax = _fmax(tmax, x)
            tmax = _xmax(tmax)
            _lds_store(Vec.from_elements([tmax], fx.Float32), T.f32, base + EX_OFF + (w * 16 + t) * 4, 4)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            mv = [
                fx.Float32(fx.ptr_load(_ptr(T.f32, base + EX_OFF + (j * 16 + t) * 4, 4), result_type=T.f32))
                for j in range_constexpr(NW)
            ]
            m_g = mv[0]
            for j in range_constexpr(1, NW):
                m_g = _fmax(m_g, mv[j])
            m_new = _fmax(m_run, m_g)
            alpha = (m_run - m_new).exp2(fastmath="afn")
            m_cl = _fmax(m_new - _P_SHIFT, fx.Float32(-1.0e20))
            pl = [(sl[i] - m_cl).exp2(fastmath="afn") for i in range(4 * NSLOT)]
            psum = fx.Float32(0.0)
            for x in pl:
                psum = psum + x
            psum = _xsum(psum)
            _lds_store(Vec.from_elements([psum], fx.Float32), T.f32, base + EX_OFF + 16 * 4 * NW + (w * 16 + t) * 4, 4)

            def store_ps(k):
                s = ssl[k]
                kk = s // 2
                sp = s % 2
                hv = Vec.from_elements([pl[4 * k + r] for r in range(4)], fx.Float32).to(fx.Float16).bitcast(fx.Int32)
                _lds_store(hv, T.i32, base + P_OFF + kk * 1024 + lane * 16 + sp * 8, 8)
                scb = _imin(scs[k], fx.Int32(_SCALE_CAP_BITS)).bitcast(fx.Float32)
                s16 = Vec.from_elements([scb], fx.Float32).to(fx.Float16)
                _lds_store(s16, T.f16, base + SC_OFF + (g * RT + 16 * s + t) * 2, 2)

            for k in range_constexpr(NSLOT):
                maybe(vals[k], lambda _k=k: store_ps(_k))
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            stamp(6)

            lps = [
                fx.Float32(fx.ptr_load(_ptr(T.f32, base + EX_OFF + 16 * 4 * NW + (j * 16 + t) * 4, 4), result_type=T.f32))
                for j in range_constexpr(NW)
            ]
            l_pass = lps[0]
            for j in range_constexpr(1, NW):
                l_pass = l_pass + lps[j]
            l_new = l_run * alpha + l_pass
            o = [
                Vec.from_elements([fx.Float32(Vec(o[i])[j]) * alpha for j in range(4)], fx.Float32)
                for i in range(CPW)
            ]

            def pv_kk(kk, o):
                pf = _lds_load(T.i32, 4, base + P_OFF + kk * 1024 + lane * 16, 16)
                sv = []
                for sp in range_constexpr(2):
                    sv.append(_lds_load(T.i32, 2, base + SC_OFF + (((CPW * w) // 8) * RT + 32 * kk + 16 * sp + 4 * g) * 2, 8))
                scv = _cat(sv[0], sv[1], 2)
                bfr = (Vec(pf).bitcast(fx.Float16) * Vec(scv).bitcast(fx.Float16)).bitcast(fx.Int32)
                o = list(o)
                for d in range_constexpr(CPW):
                    a = _cvt_v8(_tr8(base + ((v_off ^ (d * 16))), kk * 16384))
                    o[d] = _mfma_f16(a, bfr, o[d])
                return o

            for kk in range_constexpr(0 if NOPV else KKN):
                if const_expr(NOG):
                    o = pv_kk(kk, o)
                else:
                    if 32 * kk < len_p:
                        o = pv_kk(kk, o)
            stamp(7)
            scs_n = [fx.Int32(0x3F800000) for _ in range(NSLOT)]
            rps_n = [Vec.filled(4, 0, fx.Int32) for _ in range(2 * NSLOT)]
            if fx.Int32(pas) + 1 < npass:
                gpu.barrier()
                scs_n, rps_n = issue_pass(cb + RT)
            fin = yield [m_new, l_new] + o + scs_n + rps_n

        l_fin = fx.Float32(fin[1])
        has_l = l_fin > fx.Float32(0.0)
        inv_l = has_l.select(fx.Float32(1.0) / l_fin, fx.Float32(0.0))
        for d in range_constexpr(CPW):
            ov = Vec(fin[2 + d])
            bf = Vec.from_elements([fx.Float32(ov[j]) * inv_l for j in range(4)], fx.Float32).to(fx.BFloat16)
            o_t.store(
                (qi * stride_o + (hg * H + t) * stride_oh + 16 * (CPW * w + d) + 4 * g) // 2,
                bf.bitcast(fx.Int32),
                vec_size=2,
            )
        lse2 = fx.Float32(fin[0]) + fx.Float32(fx.log2(has_l.select(l_fin, fx.Float32(1.0)))) - _P_SHIFT
        lse = has_l.select(lse2 * _LN2, fx.Float32(float("-inf")))
        lse_t.store(qi * HEADS + hg * H + t, lse.bitcast(fx.Int32))
        stamp(8)

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
            grid=(fx.Int64(((n_q * N_HG + _XCDS - 1) // _XCDS) * _XCDS), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_decode.compile_hints = {"waves_per_eu": 1, "fast_fp_math": False}
    return launch_decode
