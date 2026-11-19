# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Per-row top-2048 of the prefill indexer's logits over each row's window ``[ks, ke)`` (gfx950, FlyDSL), in three launches.

``threshold``: one workgroup per row estimates the value of rank ~1.5 * 2048 from a stratified sample of 64-key blocks (about
3.6% of a long window) and writes it as a float.
``stream``: one workgroup per row streams the window once (nontemporal 16-byte loads, one chunk prefetched) and keeps every
value >= the threshold in an LDS list that is ascending in position by construction (each wave owns a contiguous slice of every
chunk, the waves' counts are exchanged through LDS once per chunk), then copies the list out. A row whose count is outside
[2048, CAP] is streamed again with the threshold re-estimated from the observed count (up to three passes).
``select``: one workgroup per row takes the exact 2048th largest of the list (12-bit radix rounds over the value range, then a
brute-force rank of the few entries sharing the final bin) and writes the selected positions in ascending order, ties going to
the smaller position. A row whose passes all failed is selected by an exact radix over the window itself (12 / 12 / 8-bit digits,
three passes over the row, then an ordered emit): slow, rare, and never wrong.

Rows whose window has at most CAP keys skip the threshold: the list is the window.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import T, as_ir_value

from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher

I, F, I64 = fx.Int32, fx.Float32, fx.Int64
INT_MIN = -2147483648
KMAXV = 0x7FFFFFFF
NUM_REC = 0x7FFFF000
K = 2048
NBMAX = 160
SDIV = 80
TMID, TA, TC = 3584, 11264, 2560
NBIN = 4096
HSZ = NBIN + 64 + (NBIN + 64) // 64 + 4
CAP = 6144


def _update_dpp(old, src, ctrl, row_mask, bank_mask):
    return rocdl.update_dpp(I.ir_type, I(old).ir_value(), I(src).ir_value(), ctrl, row_mask, bank_mask, False)


def _wave_scan(v, op):
    """Inclusive scan over the 64 lanes (lane 63 holds the reduction) with op in add / max / min."""
    v = I(v)
    for ctrl, rmask in ((0x111, 0xF), (0x112, 0xF), (0x114, 0xF), (0x118, 0xF), (0x142, 0xA), (0x143, 0xC)):
        o = I(_update_dpp(I(0) if op == "add" else v, v, ctrl, rmask, 0xF))
        v = v + o if op == "add" else (fx.max(v, o) if op == "max" else fx.min(v, o))
    return v


def _readlane(v, lane):
    return I(rocdl.readlane(T.i32, I(v), I(lane)))


def _uniform(v):
    return I(rocdl.readfirstlane(T.i32, I(v)))


def _ballot(pred):
    return I64(rocdl.ballot(T.i64, pred))


def _lane_prefix(mask):
    lo = mask.to(I)
    hi = (mask >> I64(32)).to(I)
    z = I(0)
    return I(rocdl.mbcnt_hi(T.i32, as_ir_value(hi), as_ir_value(I(rocdl.mbcnt_lo(T.i32, as_ir_value(lo), as_ir_value(z))))))


def _key_to_float(key):
    return (key < 0).select(key ^ I(0x7FFFFFFF), key).bitcast(F)


def _float_to_key(x):
    bits = (F(x) + 0.0).bitcast(I)
    return (bits < 0).select(bits ^ I(0x7FFFFFFF), bits)


def _sts(field, idx, v):
    field.ptr[idx] = v


def _popc(mask):
    return I(fx.math.ctpop(mask).to(I))


# =====================================================================================================================
# K1: per-row threshold
# =====================================================================================================================
@launcher
def compile_threshold(NT=512, CAP=6144):
    NW = NT // 64
    NLS = (NBMAX + NW * 4 - 1) // (NW * 4)
    NKS = NLS * 4

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, HSZ]
        wred: fx.Array[fx.Int32, 64]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k1(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, i64_stride: fx.Int64):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        g = lane // 16
        l16 = lane % 16
        row = I(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_r = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        meta_r = buffer_ops.create_buffer_resource_from_addr(arg_meta)
        ks = _uniform(buffer_ops.buffer_load(ks_r, row, vec_width=1, dtype=I))
        ke = _uniform(buffer_ops.buffer_load(ke_r, row, vec_width=1, dtype=I))
        L = ke - ks
        ks_al = ks & I(-64)
        exact = L <= CAP
        if exact:
            if tx == 0:
                buffer_ops.buffer_store(fx.Vector.from_elements([F(float("-inf")).bitcast(I), F(float("-inf")).bitcast(I), F(float("-inf")).bitcast(I), I(0)], dtype=I), meta_r, row * 4)
        else:
            rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)

            def zero(field, n):
                for i in range_constexpr(n // NT + 1):
                    p = tx + NT * i
                    if p < n:
                        _sts(field, p, I(0))

            def lsr(x, s):
                return (s == 0).select(x, (x >> s) & ((I(1) << (I(32) - s)) - 1))

            def pad(b):
                return b + (b >> 6)

            def scan_hist(need_):
                ssum = I(smem.hist.ptr[(63 - lane) * 65])
                for t in range_constexpr(1, 64):
                    ssum = ssum + I(smem.hist.ptr[(63 - lane) * 65 + t])
                incl = _wave_scan(ssum, "add")
                hit = _ballot(incl >= need_)
                lstar = I(fx.math.cttz(hit).to(I))
                excl = _readlane(incl - ssum, lstar)
                cbase = (63 - lstar) * 64
                c = I(smem.hist.ptr[cbase + (63 - lane) + (63 - lstar)])
                incl2 = _wave_scan(c, "add")
                cum = excl + incl2
                hit2 = _ballot(cum >= need_)
                l2 = I(fx.math.cttz(hit2).to(I))
                return cbase + 63 - l2, _readlane(cum - c, l2), _readlane(c, l2)

            def blk_reduce(vals, ops):
                outs = []
                for q in range_constexpr(len(vals)):
                    w_ = _wave_scan(vals[q], ops[q])
                    if lane == 63:
                        _sts(smem.wred, q * NW + wave, w_)
                fx.gpu.barrier()
                for q in range_constexpr(len(vals)):
                    acc = I(smem.wred.ptr[q * NW])
                    for w in range_constexpr(1, NW):
                        x = I(smem.wred.ptr[q * NW + w])
                        acc = acc + x if ops[q] == "add" else (fx.max(acc, x) if ops[q] == "max" else fx.min(acc, x))
                    outs.append(_uniform(acc))
                fx.gpu.barrier()
                return outs

            span_blocks = (ke - ks_al + 63) // 64
            ns = fx.min(fx.max(I(2048), L // SDIV), I(NBMAX * 64))
            nblk = (ns + 63) // 64

            def sample():
                keys = []
                nvalid = I(0)
                for t in range_constexpr(NLS):
                    j = t * (NW * 4) + wave * 4 + g
                    u0 = (j * span_blocks) // nblk
                    wb = ((j + 1) * span_blocks) // nblk - u0
                    h = row * 0x45D9F3B + j * 0x2C1B3C6D + 12345
                    h = ((h >> 16) ^ h) * 0x45D9F3B
                    h = ((h >> 16) ^ h) * 0x45D9F3B
                    h = ((h >> 16) ^ h) & 0x7FFFFFFF
                    start = ks_al + (u0 + h % fx.max(wb, I(1))) * 64
                    v = fx.Vector(buffer_ops.buffer_load(rsrc, start + l16 * 4, vec_width=4, dtype=F, mask=j < nblk))
                    for k in range_constexpr(4):
                        e = start + l16 * 4 + k
                        ok = (j < nblk) & (e >= ks) & (e < ke)
                        keys.append(ok.select(_float_to_key(v[k]), I(INT_MIN)))
                        nvalid = nvalid + ok.select(I(1), I(0))
                return keys, nvalid

            keys, nvalid = sample()

            def stats():
                rem = keys
                top = []
                for m in range_constexpr(3):
                    mx = rem[0]
                    for t in range_constexpr(1, NKS):
                        mx = fx.max(mx, rem[t])
                    top.append(mx)
                    rem = [(rem[t] == mx).select(I(INT_MIN), rem[t]) for t in range_constexpr(NKS)]
                kmin_all = (keys[0] == INT_MIN).select(I(KMAXV), keys[0])
                for t in range_constexpr(1, NKS):
                    kmin_all = fx.min(kmin_all, (keys[t] == INT_MIN).select(I(KMAXV), keys[t]))
                tmin3 = fx.min(fx.min((top[0] == INT_MIN).select(I(KMAXV), top[0]), (top[1] == INT_MIN).select(I(KMAXV), top[1])),
                               (top[2] == INT_MIN).select(I(KMAXV), top[2]))
                return top, top[0], kmin_all, tmin3

            top, kmax_all, kmin_all, tmin3 = stats()
            nv, pmax, pmin_all, pmin3 = blk_reduce([nvalid, kmax_all, kmin_all, tmin3], ["add", "max", "min", "min"])
            def rank_of(t_):
                return fx.max(fx.min((nv * t_ + (L - 1)) // fx.max(L, I(1)), nv), I(1))

            r_mid = rank_of(TMID)
            r_a = rank_of(TA)
            r_c = rank_of(TC)
            use3 = r_a <= 192
            pmin = use3.select(pmin3, pmin_all)
            dmax = pmax - pmin
            bl = I(32) - I(fx.math.ctlz(dmax | 1).to(I))
            sh = fx.max(bl - 12, I(0))
            zero(smem.hist, HSZ)
            fx.gpu.barrier()
            if use3:
                for t in range_constexpr(3):
                    kk = top[t]
                    idx = (kk != INT_MIN).select(lsr(kk - pmin, sh), I(NBIN) + lane)
                    fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(idx)), I(1), syncscope="workgroup")
            else:
                for t in range_constexpr(NKS):
                    kk = keys[t]
                    idx = (kk != INT_MIN).select(lsr(kk - pmin, sh), I(NBIN) + lane)
                    fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(idx)), I(1), syncscope="workgroup")
            fx.gpu.barrier()
            b0, ab0, pp0 = scan_hist(r_mid)
            ba, aba, ppa = scan_hist(r_a)
            bc, abc, ppc = scan_hist(r_c)
            fx.gpu.barrier()
            sh1 = fx.max(sh - 12, I(0))
            zero(smem.hist, HSZ)
            fx.gpu.barrier()
            if use3:
                for t in range_constexpr(3):
                    kk = top[t]
                    mem = (kk != INT_MIN) & (lsr(kk - pmin, sh) == b0)
                    fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(mem.select(lsr(kk - pmin, sh1) - (b0 << (sh - sh1)), I(NBIN) + lane))), I(1), syncscope="workgroup")
            else:
                for t in range_constexpr(NKS):
                    kk = keys[t]
                    mem = (kk != INT_MIN) & (lsr(kk - pmin, sh) == b0)
                    fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(mem.select(lsr(kk - pmin, sh1) - (b0 << (sh - sh1)), I(NBIN) + lane))), I(1), syncscope="workgroup")
            fx.gpu.barrier()
            b1, ab1, pp1 = scan_hist(r_mid - ab0)
            thr_mid = _key_to_float(pmin + (((b0 << (sh - sh1)) + b1) << sh1))
            thr_a = _key_to_float(pmin + (ba << sh))
            thr_c = _key_to_float(pmin + (bc << sh))
            if tx == 0:
                buffer_ops.buffer_store(fx.Vector.from_elements([thr_mid.bitcast(I), thr_a.bitcast(I), thr_c.bitcast(I), nv], dtype=I), meta_r, row * 4)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, i64_stride: fx.Int64,
               rows: fx.Int32, stream: fx.Stream):
        k1(arg_lg, arg_ks, arg_ke, arg_meta, i64_stride).launch(grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =====================================================================================================================
# K2: stream the window, keep every value >= threshold in an index-ascending list; retry with a corrected threshold
# =====================================================================================================================
@launcher
def compile_stream(NT=512, V=4, CAP=6144, MARG=4000, NC_ST=2, stamps=0, ac=1, depth=1):
    NW = NT // 64
    CH = NT * 4 * V
    SW = V * 256
    NLS = (NBMAX + NW * 4 - 1) // (NW * 4)
    NKS = NLS * 4
    LSZ = CAP + 64

    @fx.struct
    class Shared:
        lkey: fx.Array[fx.Int32, LSZ]
        lidx: fx.Array[fx.Int32, LSZ]
        hist: fx.Array[fx.Int32, HSZ]
        wred: fx.Array[fx.Int32, 64]
        cbuf: fx.Array[fx.Int32, 32]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k2(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64,
           i64_stride: fx.Int64, i32_cstride: fx.Int32, arg_order: fx.Int64):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        g = lane // 16
        l16 = lane % 16
        row = I(fx.block_idx.x)
        if arg_order != I64(0):
            row = _uniform(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_order), row, vec_width=1, dtype=I))
        smem = fx.SharedAllocator().allocate(Shared).peek()

        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_r = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        meta_r = buffer_ops.create_buffer_resource_from_addr(arg_meta)
        cnt_r = buffer_ops.create_buffer_resource_from_addr(arg_cnt)
        cand_r = buffer_ops.create_buffer_resource_from_addr(arg_cand, num_records_bytes=NUM_REC)
        ks = _uniform(buffer_ops.buffer_load(ks_r, row, vec_width=1, dtype=I))
        ke = _uniform(buffer_ops.buffer_load(ke_r, row, vec_width=1, dtype=I))
        mt = fx.Vector(buffer_ops.buffer_load(meta_r, row * 4, vec_width=4, dtype=I))
        tm0 = _uniform(mt[0])
        ta0 = _uniform(mt[1])
        tc0 = _uniform(mt[2])
        nv = _uniform(mt[3])
        L = ke - ks
        ks_al = ks & I(-64)
        rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)
        nit = (ke - ks_al + (CH - 1)) // CH
        exact = L <= CAP

        def zero(field, n):
            for i in range_constexpr(n // NT + 1):
                p = tx + NT * i
                if p < n:
                    _sts(field, p, I(0))

        def load_chunk(i, ok=None):
            out = []
            for v in range_constexpr(V):
                out.append(buffer_ops.buffer_load(rsrc, ks_al + i * CH + wave * SW + (v * 64 + lane) * 4, vec_width=4, dtype=F,
                                                  cache_modifier=NC_ST, mask=ok))
            return out

        first = []
        for d_ in range_constexpr(depth):
            first = first + load_chunk(I(d_), nit > d_)

        def lsr(x, s):
            return (s == 0).select(x, (x >> s) & ((I(1) << (I(32) - s)) - 1))

        def pad(b):
            return b + (b >> 6)

        def scan_hist(need_):
            ssum = I(smem.hist.ptr[(63 - lane) * 65])
            for t in range_constexpr(1, 64):
                ssum = ssum + I(smem.hist.ptr[(63 - lane) * 65 + t])
            incl = _wave_scan(ssum, "add")
            hit = _ballot(incl >= need_)
            lstar = I(fx.math.cttz(hit).to(I))
            excl = _readlane(incl - ssum, lstar)
            cbase = (63 - lstar) * 64
            c = I(smem.hist.ptr[cbase + (63 - lane) + (63 - lstar)])
            incl2 = _wave_scan(c, "add")
            cum = excl + incl2
            hit2 = _ballot(cum >= need_)
            l2 = I(fx.math.cttz(hit2).to(I))
            return cbase + 63 - l2, _readlane(cum - c, l2), _readlane(c, l2)

        def blk_reduce(vals, ops):
            outs = []
            for q in range_constexpr(len(vals)):
                w_ = _wave_scan(vals[q], ops[q])
                if lane == 63:
                    _sts(smem.wred, q * NW + wave, w_)
            fx.gpu.barrier()
            for q in range_constexpr(len(vals)):
                acc = I(smem.wred.ptr[q * NW])
                for w in range_constexpr(1, NW):
                    x = I(smem.wred.ptr[q * NW + w])
                    acc = acc + x if ops[q] == "add" else (fx.max(acc, x) if ops[q] == "max" else fx.min(acc, x))
                outs.append(_uniform(acc))
            fx.gpu.barrier()
            return outs

        def blk_sumuniform(vals):
            for q in range_constexpr(len(vals)):
                if lane == 0:
                    _sts(smem.wred, q * NW + wave, vals[q])
            fx.gpu.barrier()
            outs = []
            for q in range_constexpr(len(vals)):
                acc = I(smem.wred.ptr[q * NW])
                for w in range_constexpr(1, NW):
                    acc = acc + I(smem.wred.ptr[q * NW + w])
                outs.append(_uniform(acc))
            fx.gpu.barrier()
            return outs

        def resample_thresholds(r_mid):
            span_blocks = (ke - ks_al + 63) // 64
            ns = fx.min(fx.max(I(2048), L // SDIV), I(NBMAX * 64))
            nblk = (ns + 63) // 64

            def sample():
                keys = []
                for t in range_constexpr(NLS):
                    j = t * (NW * 4) + wave * 4 + g
                    u0 = (j * span_blocks) // nblk
                    wb = ((j + 1) * span_blocks) // nblk - u0
                    h = row * 0x45D9F3B + j * 0x2C1B3C6D + 12345
                    h = ((h >> 16) ^ h) * 0x45D9F3B
                    h = ((h >> 16) ^ h) * 0x45D9F3B
                    h = ((h >> 16) ^ h) & 0x7FFFFFFF
                    start = ks_al + (u0 + h % fx.max(wb, I(1))) * 64
                    v = fx.Vector(buffer_ops.buffer_load(rsrc, start + l16 * 4, vec_width=4, dtype=F, mask=j < nblk))
                    for k in range_constexpr(4):
                        e = start + l16 * 4 + k
                        ok = (j < nblk) & (e >= ks) & (e < ke)
                        keys.append(ok.select(_float_to_key(v[k]), I(INT_MIN)))
                return keys

            keys = sample()
            kmax_all = keys[0]
            kmin_all = (keys[0] == INT_MIN).select(I(KMAXV), keys[0])
            for t in range_constexpr(1, NKS):
                kmax_all = fx.max(kmax_all, keys[t])
                kmin_all = fx.min(kmin_all, (keys[t] == INT_MIN).select(I(KMAXV), keys[t]))
            pmax, pmin = blk_reduce([kmax_all, kmin_all], ["max", "min"])
            dmax = pmax - pmin
            bl = I(32) - I(fx.math.ctlz(dmax | 1).to(I))
            sh = fx.max(bl - 12, I(0))
            r_a = fx.max(fx.min((r_mid * TA) // TMID, nv), I(1))
            r_c = fx.max((r_mid * TC) // TMID, I(1))
            zero(smem.hist, HSZ)
            fx.gpu.barrier()
            for t in range_constexpr(NKS):
                kk = keys[t]
                idx = (kk != INT_MIN).select(lsr(kk - pmin, sh), I(NBIN) + lane)
                fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(idx)), I(1), syncscope="workgroup")
            fx.gpu.barrier()
            b0, ab0, pp0 = scan_hist(r_mid)
            ba, aba, ppa = scan_hist(r_a)
            bc, abc, ppc = scan_hist(r_c)
            fx.gpu.barrier()
            sh1 = fx.max(sh - 12, I(0))
            zero(smem.hist, HSZ)
            fx.gpu.barrier()
            for t in range_constexpr(NKS):
                kk = keys[t]
                mem = (kk != INT_MIN) & (lsr(kk - pmin, sh) == b0)
                fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(mem.select(lsr(kk - pmin, sh1) - (b0 << (sh - sh1)), I(NBIN) + lane))), I(1), syncscope="workgroup")
            fx.gpu.barrier()
            b1, ab1, pp1 = scan_hist(r_mid - ab0)
            fx.gpu.barrier()
            return (_key_to_float(pmin + (((b0 << (sh - sh1)) + b1) << sh1)), _key_to_float(pmin + (ba << sh)), _key_to_float(pmin + (bc << sh)))

        wlen = L.bitcast(fx.Uint32)
        r0 = fx.max(fx.min((nv * TMID + (L - 1)) // fx.max(L, I(1)), nv), I(1))
        state0 = [r0, I(0), I(0), tm0, ta0, tc0, I(1)] + list(first)
        for att, st in range(0, 3, 1, init=state0):
            r_c = _uniform(st[0])
            done_c = _uniform(st[1])
            cnt_c = _uniform(st[2])
            tm_c = _uniform(st[3])
            ta_c = _uniform(st[4])
            tc_c = _uniform(st[5])
            have_c = _uniform(st[6])
            first_c = [st[7 + q] for q in range_constexpr(depth * V)]
            r_n = r_c
            done_n = done_c
            cnt_n = cnt_c
            tm_n = tm_c
            ta_n = ta_c
            tc_n = tc_c
            have_n = have_c
            nf = list(first_c)
            if done_c == 0:
                thr = tm_c.bitcast(F)
                thr_a = ta_c.bitcast(F)
                thr_c = tc_c.bitcast(F)
                if have_c == 0:
                    thr, thr_a, thr_c = resample_thresholds(r_c)
                for i, state in range(0, nit, 1, init=list(first_c) + [I(0), I(0), I(0)]):
                    cur = [fx.Vector(state[v]) for v in range_constexpr(V)]
                    run0 = _uniform(state[depth * V])
                    na0 = _uniform(state[depth * V + 1])
                    nc0 = _uniform(state[depth * V + 2])
                    nxt = load_chunk(I(i) + depth, (I(i) + depth) < nit)
                    base = ks_al + I(i) * CH + wave * SW
                    hv, incls, tots = [], [], []
                    wtot = I(0)
                    ia = I(0)
                    ic = I(0)
                    for v in range_constexpr(V):
                        hk = []
                        cl = I(0)
                        for k in range_constexpr(4):
                            e = base + (v * 64 + lane) * 4 + k
                            inr = (e - ks).bitcast(fx.Uint32) < wlen
                            xv = F(cur[v][k])
                            hh = (xv >= thr) & inr
                            hk.append(hh)
                            cl = cl + hh.select(I(1), I(0))
                            if fx.const_expr(ac):
                                ia = ia + _popc(_ballot((xv >= thr_a) & inr))
                                ic = ic + _popc(_ballot((xv >= thr_c) & inr))
                        incl = _wave_scan(cl, "add")
                        tv = _readlane(incl, 63)
                        hv.append(hk)
                        incls.append(incl - cl)
                        tots.append(tv)
                        wtot = wtot + tv
                    par = (I(i) & 1) * 16
                    if lane == 0:
                        _sts(smem.cbuf, par + wave, wtot)
                    fx.gpu.barrier()
                    pref = I(0)
                    itot = I(0)
                    for w in range_constexpr(NW):
                        cw = _uniform(smem.cbuf.ptr[par + w])
                        pref = pref + (wave > w).select(cw, I(0))
                        itot = itot + cw
                    runv = run0 + pref
                    for v in range_constexpr(V):
                        pos = runv + incls[v]
                        for k in range_constexpr(4):
                            wp = (hv[v][k] & (pos < CAP)).select(pos, I(CAP) + lane)
                            e2 = base + (v * 64 + lane) * 4 + k
                            _sts(smem.lidx, wp, e2)
                            _sts(smem.lkey, wp, F(cur[v][k]).bitcast(I))
                            pos = pos + hv[v][k].select(I(1), I(0))
                        runv = runv + tots[v]
                    res = yield [state[V + q] for q in range_constexpr((depth - 1) * V)] + [nxt[v] for v in range_constexpr(V)] + [run0 + itot, na0 + ia, nc0 + ic]
                fx.gpu.barrier()
                cnt_n = _uniform(res[depth * V])
                na_w, nc_w = res[depth * V + 1], res[depth * V + 2]
                n_a, n_c = blk_sumuniform([na_w, nc_w])
                ok_n = exact | ((cnt_n >= K) & (cnt_n <= CAP))
                done_n = ok_n.select(I(1), I(0))
                # next pass when this one failed: a threshold whose exact count lies in [K, CAP] if one of the three does,
                # else the sample is re-ranked by the observed ratio
                use_c = (cnt_n > CAP) & (n_c >= K) & (n_c <= CAP)
                use_a = (cnt_n < K) & (n_a >= K) & (n_a <= CAP)
                have_n = (use_c | use_a).select(I(1), I(0))
                tm_n = use_c.select(tc_c, use_a.select(ta_c, tm_c))
                ta_n = tm_n
                tc_n = tm_n
                r_n = fx.max(fx.min((r_c * TMID) // fx.max(cnt_n, I(1)), nv), I(1))
                nf = []
                for d_ in range_constexpr(depth):
                    nf = nf + load_chunk(I(d_), (~ok_n) & (nit > d_))
                fx.gpu.barrier()
            att_res = yield [r_n, done_n, cnt_n, tm_n, ta_n, tc_n, have_n] + list(nf)
        count = _uniform(att_res[2])
        done_f = _uniform(att_res[1])
        if done_f != 0:
            ncopy = fx.min(fx.min(count, I(CAP)), i32_cstride)
            for c in range(0, (ncopy + (NT - 1)) // NT, 1):
                j = tx + NT * I(c)
                a = I(smem.lidx.ptr[j])
                b = I(smem.lkey.ptr[j])
                buffer_ops.buffer_store(fx.Vector.from_elements([a, b], dtype=I), cand_r, (row * i32_cstride + j) * 2, mask=j < ncopy)
        if tx == 0:
            buffer_ops.buffer_store(count, cnt_r, row)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_meta: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64,
               i64_stride: fx.Int64, i32_cstride: fx.Int32, arg_order: fx.Int64, rows: fx.Int32, stream: fx.Stream):
        k2(arg_lg, arg_ks, arg_ke, arg_meta, arg_cnt, arg_cand, i64_stride, i32_cstride, arg_order).launch(
            grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


# =====================================================================================================================
# K3: exact select over the candidate list (registers), exact fallback for rows whose list is unusable
# =====================================================================================================================
@launcher
def compile_select(NT=256, CAP=6144, POPMAX=64):
    NW = NT // 64
    PER = CAP // NT
    NCH = NT * 4

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, HSZ]
        tmp: fx.Array[fx.Int32, POPMAX + 64]
        wred: fx.Array[fx.Int32, 64]
        wsum: fx.Array[fx.Int32, 16]
        ctl: fx.Array[fx.Int32, 16]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k3(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64, arg_out: fx.Int64,
           i64_stride: fx.Int64, i32_cstride: fx.Int32, i32_out_stride: fx.Int32):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        row = I(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()

        ks_r = buffer_ops.create_buffer_resource_from_addr(arg_ks)
        ke_r = buffer_ops.create_buffer_resource_from_addr(arg_ke)
        cnt_r = buffer_ops.create_buffer_resource_from_addr(arg_cnt)
        cand_r = buffer_ops.create_buffer_resource_from_addr(arg_cand, num_records_bytes=NUM_REC)
        out_r = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=NUM_REC)
        ks = _uniform(buffer_ops.buffer_load(ks_r, row, vec_width=1, dtype=I))
        ke = _uniform(buffer_ops.buffer_load(ke_r, row, vec_width=1, dtype=I))
        count = _uniform(buffer_ops.buffer_load(cnt_r, row, vec_width=1, dtype=I))
        L = ke - ks
        ks_al = ks & I(-64)
        exact = L <= CAP
        good = exact | ((count >= K) & (count <= CAP))

        def zero(field, n):
            for i in range_constexpr(n // NT + 1):
                p = tx + NT * i
                if p < n:
                    _sts(field, p, I(0))

        def lsr(x, s):
            if fx.const_expr(isinstance(s, int)):
                if fx.const_expr(s == 0):
                    res_ = x
                else:
                    res_ = (x >> s) & ((1 << (32 - s)) - 1)
            else:
                res_ = (s == 0).select(x, (x >> s) & ((I(1) << (I(32) - s)) - 1))
            return res_

        def pad(b):
            return b + (b >> 6)

        def hist_inc(idx):
            fx.atomic_add(fx.add_offset(smem.hist.ptr, pad(idx)), I(1), syncscope="workgroup")

        def scan_hist(need_):
            ssum = I(smem.hist.ptr[(63 - lane) * 65])
            for t in range_constexpr(1, 64):
                ssum = ssum + I(smem.hist.ptr[(63 - lane) * 65 + t])
            incl = _wave_scan(ssum, "add")
            hit = _ballot(incl >= need_)
            lstar = I(fx.math.cttz(hit).to(I))
            excl = _readlane(incl - ssum, lstar)
            cbase = (63 - lstar) * 64
            c = I(smem.hist.ptr[cbase + (63 - lane) + (63 - lstar)])
            incl2 = _wave_scan(c, "add")
            cum = excl + incl2
            hit2 = _ballot(cum >= need_)
            l2 = I(fx.math.cttz(hit2).to(I))
            return cbase + 63 - l2, _readlane(cum - c, l2), _readlane(c, l2)

        def blk_reduce(vals, ops):
            outs = []
            for q in range_constexpr(len(vals)):
                w_ = _wave_scan(vals[q], ops[q])
                if lane == 63:
                    _sts(smem.wred, q * NW + wave, w_)
            fx.gpu.barrier()
            for q in range_constexpr(len(vals)):
                acc = I(smem.wred.ptr[q * NW])
                for w in range_constexpr(1, NW):
                    x = I(smem.wred.ptr[q * NW + w])
                    acc = acc + x if ops[q] == "add" else (fx.max(acc, x) if ops[q] == "max" else fx.min(acc, x))
                outs.append(_uniform(acc))
            fx.gpu.barrier()
            return outs

        if good:
            N = fx.min(fx.min(count, I(CAP)), i32_cstride)
            need0 = fx.min(N, I(K))
            per = (N + (NT - 1)) // NT
            keys = []
            for q in range_constexpr(PER):
                jq = tx * per + q
                okq = (q < per) & (jq < N)
                pr = fx.Vector(buffer_ops.buffer_load(cand_r, (row * i32_cstride + jq) * 2, vec_width=2, dtype=I, mask=okq))
                keys.append(okq.select(_float_to_key(I(pr[1]).bitcast(F)), I(INT_MIN)))

            def load_idx(q):
                jq = tx * per + q
                okq = (q < per) & (jq < N)
                return I(buffer_ops.buffer_load(cand_r, (row * i32_cstride + jq) * 2, vec_width=1, dtype=I, mask=okq)) - ks

            all_sel = N <= K
            kt = I(INT_MIN)
            need_eq = I(0)
            if ~all_sel:
                lmax = keys[0]
                lmin = (keys[0] == INT_MIN).select(I(KMAXV), keys[0])
                for q in range_constexpr(1, PER):
                    lmax = fx.max(lmax, keys[q])
                    lmin = fx.min(lmin, (keys[q] == INT_MIN).select(I(KMAXV), keys[q]))
                kmax, kmin = blk_reduce([lmax, lmin], ["max", "min"])
                dmax = kmax - kmin
                bl = I(32) - I(fx.math.ctlz(dmax | 1).to(I))

                def do_round(sh, P, Rprev, need_r, first_round):
                    zero(smem.hist, HSZ)
                    fx.gpu.barrier()
                    for q in range_constexpr(PER):
                        dd = keys[q] - kmin
                        member = keys[q] != INT_MIN
                        if not first_round:
                            member = member & (lsr(dd, P) == Rprev)
                        off_ = lsr(dd, sh) - (Rprev << (P - sh))
                        hist_inc(member.select(off_, I(NBIN) + lane))
                    fx.gpu.barrier()
                    bq, aq, pq = scan_hist(need_r)
                    fx.gpu.barrier()
                    return bq, aq, pq

                def brute(Rv, sh, need_r, pop):
                    if tx == 0:
                        _sts(smem.ctl, 2, I(0))
                    fx.gpu.barrier()
                    for q in range_constexpr(PER):
                        mem3 = (keys[q] != INT_MIN) & (lsr(keys[q] - kmin, sh) == Rv)
                        if mem3:
                            pp = I(fx.atomic_add(fx.add_offset(smem.ctl.ptr, I(2)), I(1), syncscope="workgroup"))
                            _sts(smem.tmp, pp, keys[q])
                    fx.gpu.barrier()
                    if tx < pop:
                        mine = I(smem.tmp.ptr[tx])
                        gcnt = I(0)
                        ecnt = I(0)
                        for qq, sq in range(0, pop, 1, init=[gcnt, ecnt]):
                            tq = I(smem.tmp.ptr[I(qq)])
                            rq = yield [sq[0] + (tq > mine).select(I(1), I(0)), sq[1] + (tq == mine).select(I(1), I(0))]
                        gq = rq[0]
                        eq_ = rq[1]
                        if (gq < need_r) & (need_r <= gq + eq_):
                            _sts(smem.ctl, 3, mine)
                            _sts(smem.ctl, 4, need_r - gq)
                            _sts(smem.ctl, 5, eq_)
                    fx.gpu.barrier()
                    return _uniform(smem.ctl.ptr[3]), _uniform(smem.ctl.ptr[4]), _uniform(smem.ctl.ptr[5])

                sh0 = fx.max(bl - 12, I(0))
                b0, a0, p0 = do_round(sh0, bl, I(0), need0, True)
                n0 = need0 - a0
                fin = I(0)
                if (sh0 == 0) | (p0 <= POPMAX):
                    if sh0 == 0:
                        kt = kmin + b0
                        need_eq = n0
                    else:
                        kt, need_eq, eq_total = brute(b0, sh0, n0, p0)
                    fin = I(1)
                if fin == 0:
                    sh1 = fx.max(sh0 - 12, I(0))
                    b1, a1, p1 = do_round(sh1, sh0, b0, n0, False)
                    r1 = (b0 << (sh0 - sh1)) + b1
                    n1 = n0 - a1
                    if (sh1 == 0) | (p1 <= POPMAX):
                        if sh1 == 0:
                            kt = kmin + r1
                            need_eq = n1
                        else:
                            kt, need_eq, eq_total = brute(r1, sh1, n1, p1)
                        fin = I(1)
                    if fin == 0:
                        b2, a2, p2 = do_round(I(0), sh1, r1, n1, False)
                        kt = kmin + ((r1 << sh1) + b2)
                        need_eq = n1 - a2
            if all_sel:
                for q in range_constexpr(PER):
                    jq = tx * per + q
                    buffer_ops.buffer_store(load_idx(q), out_r, row * i32_out_stride + jq, mask=(q < per) & (jq < N))
            else:
                gtc = I(0)
                eqc = I(0)
                for q in range_constexpr(PER):
                    gtc = gtc + (keys[q] > kt).select(I(1), I(0))
                    eqc = eqc + (keys[q] == kt).select(I(1), I(0))
                packed = gtc + (eqc << 16)
                incl5 = _wave_scan(packed, "add")
                if lane == 63:
                    _sts(smem.wsum, wave, incl5)
                fx.gpu.barrier()
                prior5 = I(0)
                for w in range_constexpr(NW):
                    prior5 = prior5 + (wave > w).select(I(smem.wsum.ptr[w]), I(0))
                pk = incl5 - packed + prior5
                gb = pk & 0xFFFF
                eb = (pk >> 16) & 0xFFFF
                for q in range_constexpr(PER):
                    is_gt = keys[q] > kt
                    is_eq = keys[q] == kt
                    pos6 = gb + fx.min(eb, need_eq)
                    selv = is_gt | (is_eq & (eb < need_eq))
                    buffer_ops.buffer_store(load_idx(q), out_r, row * i32_out_stride + pos6, mask=selv)
                    gb = gb + is_gt.select(I(1), I(0))
                    eb = eb + is_eq.select(I(1), I(0))
            for i7 in range_constexpr(K // NT + 1):
                p7 = need0 + tx + NT * i7
                buffer_ops.buffer_store(I(-1), out_r, row * i32_out_stride + p7, mask=p7 < K)
        else:
            rsrc = buffer_ops.create_buffer_resource_from_addr(arg_lg + I64(row) * i64_stride * 4, num_records_bytes=i64_stride * 4)

            VF = 4
            CHF = NCH * VF

            def load_f(ci, ok):
                return [buffer_ops.buffer_load(rsrc, ks_al + ci * CHF + (v * NT + tx) * 4, vec_width=4, dtype=F, cache_modifier=2, mask=ok)
                        for v in range_constexpr(VF)]

            def keys_of(vec, ci, v):
                e0 = ks_al + ci * CHF + (v * NT + tx) * 4
                out = []
                for k in range_constexpr(4):
                    e = e0 + k
                    ok = (e >= ks) & (e < ke)
                    out.append(ok.select(_float_to_key(fx.Vector(vec)[k]), I(INT_MIN)))
                return e0, out

            def block_scan(v):
                incl = _wave_scan(v, "add")
                if lane == 63:
                    _sts(smem.wsum, wave, incl)
                fx.gpu.barrier()
                prior = I(0)
                total = I(0)
                for w in range_constexpr(NW):
                    wv = I(smem.wsum.ptr[w])
                    prior = prior + (wave > w).select(wv, I(0))
                    total = total + wv
                fx.gpu.barrier()
                return incl - v + prior, total

            nch = (ke - ks_al + (CHF - 1)) // CHF
            need = I(K)
            ds = []
            pref = I(0)
            for rnd in range_constexpr(3):
                zero(smem.hist, HSZ)
                fx.gpu.barrier()
                for ci, st in range(0, nch, 1, init=load_f(I(0), nch > 0)):
                    nxt = load_f(I(ci) + 1, (I(ci) + 1) < nch)
                    for v in range_constexpr(VF):
                        e0, kk = keys_of(st[v], I(ci), v)
                        for key in kk:
                            u = key ^ I(INT_MIN)
                            valid = key != I(INT_MIN)
                            if fx.const_expr(rnd == 0):
                                idx = valid.select(lsr(u, 20), I(NBIN) + lane)
                            elif fx.const_expr(rnd == 1):
                                idx = (valid & (lsr(u, 20) == ds[0])).select(lsr(u, 8) & 0xFFF, I(NBIN) + lane)
                            else:
                                idx = (valid & (lsr(u, 8) == pref)).select(u & 0xFF, I(NBIN) + lane)
                            hist_inc(idx)
                    rr = yield [nxt[v] for v in range_constexpr(VF)]
                fx.gpu.barrier()
                d, above, pop = scan_hist(need)
                fx.gpu.barrier()
                need = need - above
                ds.append(d)
                if fx.const_expr(rnd == 1):
                    pref = (ds[0] << 12) | d
            kt = (((ds[0] << 20) | (ds[1] << 8) | ds[2])) ^ I(INT_MIN)
            need_eq = need
            for ci, state in range(0, nch, 1, init=[I(0), I(0)] + load_f(I(0), nch > 0)):
                sr = state[0]
                er = state[1]
                nxt = load_f(I(ci) + 1, (I(ci) + 1) < nch)
                for v in range_constexpr(VF):
                    e0, kk = keys_of(state[2 + v], I(ci), v)
                    eqs = [kk[k] == kt for k in range_constexpr(4)]
                    gts = [kk[k] > kt for k in range_constexpr(4)]
                    ce = eqs[0].select(I(1), I(0)) + eqs[1].select(I(1), I(0)) + eqs[2].select(I(1), I(0)) + eqs[3].select(I(1), I(0))
                    pe, te = block_scan(ce)
                    sels = []
                    run_e = er + pe
                    for k in range_constexpr(4):
                        sels.append(gts[k] | (eqs[k] & (run_e < need_eq)))
                        run_e = run_e + eqs[k].select(I(1), I(0))
                    cs4 = sels[0].select(I(1), I(0)) + sels[1].select(I(1), I(0)) + sels[2].select(I(1), I(0)) + sels[3].select(I(1), I(0))
                    ps, ts = block_scan(cs4)
                    run_s = sr + ps
                    for k in range_constexpr(4):
                        buffer_ops.buffer_store(e0 + k - ks, out_r, row * i32_out_stride + run_s, mask=sels[k] & (run_s < K))
                        run_s = run_s + sels[k].select(I(1), I(0))
                    sr = sr + ts
                    er = er + te
                resf = yield [sr, er] + [nxt[v] for v in range_constexpr(VF)]
            for i in range_constexpr(K // NT + 1):
                p = resf[0] + tx + NT * i
                buffer_ops.buffer_store(I(-1), out_r, row * i32_out_stride + p, mask=p < K)

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64, arg_out: fx.Int64,
               i64_stride: fx.Int64, i32_cstride: fx.Int32, i32_out_stride: fx.Int32, rows: fx.Int32,
               stream: fx.Stream):
        k3(arg_lg, arg_ks, arg_ke, arg_cnt, arg_cand, arg_out, i64_stride, i32_cstride, i32_out_stride).launch(
            grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch
