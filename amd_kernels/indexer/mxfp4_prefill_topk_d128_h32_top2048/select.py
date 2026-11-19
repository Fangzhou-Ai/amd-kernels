# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Exact select of the prefill indexer's top-2048 from the candidate lists of ``kernel.py`` (gfx950, FlyDSL).

``select``: one workgroup per row loads its list (ascending in position) into registers, finds the exact 2048th largest key (a
12-bit radix round over the value range of the list, then a brute-force rank of the few entries sharing the final bin) and writes
the selected positions in ascending order by one ordered compaction, ties going to the smaller position.
``select_exact``: the rows whose passes all failed (all-equal keys, pathological structure) take three radix passes over the window
itself and an ordered emit: slow, rare, never wrong.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.primitive import range_constexpr

from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher

from .kernel import (
    HSZA, INT_MIN, K, KMAXV, NBIN, NUM_REC, F, I, I64,
    _ballot, _blk_reduce, _float_to_key, _hist_add, _lsr, _readlane, _scan_hist, _sts, _uniform, _wave_scan, _zero_hist,
)  # fmt: skip


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
        hist: fx.Array[fx.Int32, HSZA]
        lidx: fx.Array[fx.Int32, CAP + 64]
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
        hist_add1 = lambda idx: _hist_add(smem, idx, I(1))
        scan_hist_ = lambda need: _scan_hist(smem, lane, need)
        blk_reduce_ = lambda vals, ops: _blk_reduce(smem, wave, NW, vals, ops)

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
                if okq:
                    _sts(smem.lidx, jq, I(pr[0]) - ks)

            def load_idx(q):
                return I(smem.lidx.ptr[tx * per + q])

            all_sel = N <= K
            kt = I(INT_MIN)
            need_eq = I(0)
            if ~all_sel:
                lmax = keys[0]
                lmin = (keys[0] == INT_MIN).select(I(KMAXV), keys[0])
                for q in range_constexpr(1, PER):
                    lmax = fx.max(lmax, keys[q])
                    lmin = fx.min(lmin, (keys[q] == INT_MIN).select(I(KMAXV), keys[q]))
                kmax, kmin = blk_reduce_([lmax, lmin], ["max", "min"])
                dmax = kmax - kmin
                bl = I(32) - I(fx.math.ctlz(dmax | 1).to(I))

                def do_round(sh, P, Rprev, need_r, first_round):
                    _zero_hist(smem, tx, NT)
                    fx.gpu.barrier()
                    for q in range_constexpr(PER):
                        dd = keys[q] - kmin
                        member = keys[q] != INT_MIN
                        if not first_round:
                            member = member & (_lsr(dd, P) == Rprev)
                        off_ = _lsr(dd, sh) - (Rprev << (P - sh))
                        hist_add1(member.select(off_, I(NBIN) + lane))
                    fx.gpu.barrier()
                    bq, aq, pq = scan_hist_(need_r)
                    fx.gpu.barrier()
                    return bq, aq, pq

                def brute(Rv, sh, need_r, pop):
                    if tx == 0:
                        _sts(smem.ctl, 2, I(0))
                    fx.gpu.barrier()
                    for q in range_constexpr(PER):
                        mem3 = (keys[q] != INT_MIN) & (_lsr(keys[q] - kmin, sh) == Rv)
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

    @flyc.jit
    def launch(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64, arg_out: fx.Int64,
               i64_stride: fx.Int64, i32_cstride: fx.Int32, i32_out_stride: fx.Int32, rows: fx.Int32,
               stream: fx.Stream):
        k3(arg_lg, arg_ks, arg_ke, arg_cnt, arg_cand, arg_out, i64_stride, i32_cstride, i32_out_stride).launch(
            grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch


@launcher
def compile_select_exact(NT=256, CAP=6144, POPMAX=64):
    NW = NT // 64
    PER = CAP // NT
    NCH = NT * 4

    @fx.struct
    class Shared:
        hist: fx.Array[fx.Int32, HSZA]
        lidx: fx.Array[fx.Int32, CAP + 64]
        tmp: fx.Array[fx.Int32, POPMAX + 64]
        wred: fx.Array[fx.Int32, 64]
        wsum: fx.Array[fx.Int32, 16]
        ctl: fx.Array[fx.Int32, 16]

    @flyc.kernel(known_block_size=[NT, 1, 1])
    def k4(arg_lg: fx.Int64, arg_ks: fx.Int64, arg_ke: fx.Int64, arg_cnt: fx.Int64, arg_cand: fx.Int64, arg_out: fx.Int64,
           i64_stride: fx.Int64, i32_cstride: fx.Int32, i32_out_stride: fx.Int32):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = _uniform(tx // 64)
        row = I(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        hist_add1 = lambda idx: _hist_add(smem, idx, I(1))
        scan_hist_ = lambda need: _scan_hist(smem, lane, need)
        blk_reduce_ = lambda vals, ops: _blk_reduce(smem, wave, NW, vals, ops)

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
        bad = (~exact) & ((count < K) | (count > CAP))

        if bad:
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
                _zero_hist(smem, tx, NT)
                fx.gpu.barrier()
                for ci, st in range(0, nch, 1, init=load_f(I(0), nch > 0)):
                    nxt = load_f(I(ci) + 1, (I(ci) + 1) < nch)
                    for v in range_constexpr(VF):
                        e0, kk = keys_of(st[v], I(ci), v)
                        for key in kk:
                            u = key ^ I(INT_MIN)
                            valid = key != I(INT_MIN)
                            if fx.const_expr(rnd == 0):
                                idx = valid.select(_lsr(u, 20), I(NBIN) + lane)
                            elif fx.const_expr(rnd == 1):
                                idx = (valid & (_lsr(u, 20) == ds[0])).select(_lsr(u, 8) & 0xFFF, I(NBIN) + lane)
                            else:
                                idx = (valid & (_lsr(u, 8) == pref)).select(u & 0xFF, I(NBIN) + lane)
                            hist_add1(idx)
                    rr = yield [nxt[v] for v in range_constexpr(VF)]
                fx.gpu.barrier()
                d, above, pop = scan_hist_(need)
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
        k4(arg_lg, arg_ks, arg_ke, arg_cnt, arg_cand, arg_out, i64_stride, i32_cstride, i32_out_stride).launch(
            grid=(I64(rows), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch
