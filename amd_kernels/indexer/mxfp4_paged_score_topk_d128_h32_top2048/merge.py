# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
# The radix passes, block scans and tie handling follow AITER's FlyDSL DCP top-k merge (aiter/ops/flydsl/kernels/
# dcp_topk_merge.py, MIT, Copyright (C) 2024-2026, Advanced Micro Devices, Inc.).
"""Merge kernel: the top 2048 keys of a row from its chunk lists.

One workgroup (512 threads) a row. The row's pool (the chunk kernel's lists, ``geometry.py``) is read into registers
as order-preserving int32 keys, thread ``t`` holding entries ``4 t .. 4 t + 3`` of each 2048-entry block. Three radix
passes (11, 11 and 10 bits) over an LDS histogram find the 2048th key; keys above it and the first ties in position
order are compacted into an LDS stage and stored as two coalesced rows (positions, scores), ``-1`` / ``-inf`` past the
live keys.

A chunk whose list is full of keys above the threshold (its smallest listed key above the 2048th, with more live keys
than listed) may hold selected keys that were not listed. Those chunks, at most seven, are scored again for this row,
and every key of them replaces their lists in the pool before the passes run once more, so the result is exact. The
scores themselves match AITER's scorer bit for bit.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import T

from ._wave import F, I, I64, INT_MAX, INT_MIN, ballot, inclusive_scan, key_to_float, now, readlane, store_lds, uniform, vop, wave_reduce
from .geometry import CHUNK_MAX, INFO_OFFSET, NCH_MAX, OVF_KEYS, POOL, TOPK, list_size_rt, slot_size
from .scoring import Scorer, score_chunk

NT = 512
NWAVES = NT // 64
BLOCKS = POOL // (NT * 4)  # pool blocks a thread holds
OVERFLOW = OVF_KEYS // (NT * 4)  # blocks of rescored keys
HPAD = 2304  # a histogram of 2048 bins in 32-bin groups of 36 words: the scan reads a lane's group conflict-free
BUF = HPAD + NT  # + a dummy word a thread for keys outside the pass's prefix
NCAND = 128  # keys in the threshold's bin the fast path ranks exactly
PASSES = ((21, 11), (10, 11), (0, 10))
NUM_REC = 0x7FFFF000
NEG_INF_BITS = 0xFF800000 - (1 << 32)


@launcher
def compile_merge(uniform: bool = False, timing: bool = False):
    name = "mxfp4_paged_topk_merge_d128_h32" + ("_u" if uniform else "") + ("_t" if timing else "")

    @fx.struct
    class Shared:
        bins: fx.Array[fx.Int32, 2 * BUF]
        scan: fx.Array[fx.Int32, NWAVES + 1]
        acc: fx.Array[fx.Int32, 32]
        sidx: fx.Array[fx.Int32, TOPK + NT + 4]
        sval: fx.Array[fx.Int32, TOPK + NT + 4]
        ckey: fx.Array[fx.Int32, NCAND + NT]
        cidx: fx.Array[fx.Int32, NCAND + NT]
        ovf: fx.Array[fx.Int32, OVF_KEYS]
        cof: fx.Array[fx.Int32, 32]
        tkey: fx.Array[fx.Int32, NCH_MAX]
        tval: fx.Array[fx.Int32, NCH_MAX]
        tcnt: fx.Array[fx.Int32, NCH_MAX]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(arg_q: fx.Int64, arg_qs: fx.Int64, arg_kv: fx.Int64, arg_kvs: fx.Int64, arg_bt: fx.Int64,
               arg_w: fx.Int64, arg_ends: fx.Int64, arg_qsl: fx.Int64, arg_keys: fx.Int64, arg_idx: fx.Int64,
               arg_meta: fx.Int64, arg_oidx: fx.Int64, arg_oval: fx.Int64, i32_bt_stride: fx.Int32,
               i32_page_stride: fx.Int32, i32_last_page: fx.Int32, i32_batch: fx.Int32, i32_oidx_stride: fx.Int32,
               i32_oval_stride: fx.Int32, i32_qlen: fx.Int32, arg_dbg: fx.Int64):
        tid = I(fx.thread_idx.x)
        lane = tid % 64
        wave = I(rocdl.readfirstlane(T.i32, tid // 64))
        row = I(fx.block_idx.x)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds_copy4 = fx.make_copy_atom(fx.UniversalCopy128b(), I)
        dbg = buffer_ops.create_buffer_resource_from_addr(arg_dbg, num_records_bytes=NUM_REC)

        def stamp(k):
            if fx.const_expr(timing):
                t_ = now()
                if tid == 0:
                    buffer_ops.buffer_store(t_.to(I), dbg, row * 16 + k)

        oidx = buffer_ops.create_buffer_resource_from_addr(arg_oidx + I64(row) * I64(i32_oidx_stride) * 4, num_records_bytes=TOPK * 4)
        oval = buffer_ops.create_buffer_resource_from_addr(arg_oval + I64(row) * I64(i32_oval_stride) * 4, num_records_bytes=TOPK * 4)

        def lds_load4(field, idx):
            reg = fx.make_rmem_tensor(4, I)
            fx.copy(lds_copy4, fx.make_view(fx.add_offset(field.ptr, idx), fx.make_layout(4, 1)), reg)
            return fx.Vector(reg.load())

        def lds_store4(field, idx, v):
            reg = fx.make_rmem_tensor(4, I)
            reg.store(v)
            fx.copy(lds_copy4, reg, fx.make_view(fx.add_offset(field.ptr, idx), fx.make_layout(4, 1)))

        def init_stage():
            lds_store4(smem.sidx, tid * 4, fx.Vector.from_elements([I(-1)] * 4, dtype=I))
            lds_store4(smem.sval, tid * 4, fx.Vector.from_elements([I(NEG_INF_BITS)] * 4, dtype=I))

        def store_stage():
            buffer_ops.buffer_store(lds_load4(smem.sidx, tid * 4), oidx, tid * 4)
            buffer_ops.buffer_store(lds_load4(smem.sval, tid * 4), oval, tid * 4)

        def lds_add(field, idx, v):
            fx.atomic_add(fx.add_offset(field.ptr, idx), v, syncscope="workgroup")

        def hist_scan(base, need):
            """Wave 0: the bin holding the ``need``-th largest key of the histogram at ``base`` (a lane's 32 bins: 8
            vector reads, then one 32-bin group re-read across lanes), the keys in bins above it, the total."""
            gbase = base + (63 - lane) * 36
            ssum = None
            for m in range_constexpr(8):
                v4 = lds_load4(smem.bins, gbase + 4 * m)
                part = (I(v4[0]) + I(v4[1])) + (I(v4[2]) + I(v4[3]))
                ssum = part if ssum is None else ssum + part
            incl = inclusive_scan(ssum)
            total = readlane(incl, 63)
            need = fx.min(need, total)
            lstar = I(fx.math.cttz(ballot(incl >= need)).to(I))
            above_g = readlane(incl - ssum, lstar)
            gsel = 63 - lstar
            h = I(smem.bins.ptr[base + gsel * 36 + (31 - (lane & 31))])
            incl2 = inclusive_scan(h)
            jstar = I(fx.math.cttz(ballot((incl2 >= need - above_g) & (lane < 32))).to(I))
            return gsel * 32 + 31 - jstar, above_g + (readlane(incl2, jstar) - readlane(h, jstar)), total

        def zero_bins(base, first_tid, nthreads):
            """Zero the histogram at ``base`` with threads ``first_tid ..``."""
            z = fx.Vector.from_elements([I(0)] * 4, dtype=I)
            for i in range_constexpr(-(-(HPAD // 4) // nthreads)):
                j = tid - first_tid + nthreads * i
                if (tid >= first_tid) & (j < HPAD // 4):
                    lds_store4(smem.bins, base + j * 4, z)

        def radix_init():
            zero_bins(0, 0, NT)
            if tid == 0:
                store_lds(smem.acc, 0, I(0))
                store_lds(smem.acc, 1, I(0))
                store_lds(smem.acc, 2, I(TOPK))
                store_lds(smem.acc, 3, I(0))
            fx.gpu.barrier()

        def radix(keys, active):
            """-> the 2048th largest live key (INT_MIN when there are at most 2048), the number of keys equal to it
            to take. Three passes over two alternating histograms: wave 0 scans one while the other waves clear
            the next. ``active[b]`` (uniform): block ``b``'s four keys per thread may be live."""
            for pi in range_constexpr(len(PASSES)):
                shift, width = PASSES[pi]
                nb = 1 << width
                base = (pi % 2) * BUF
                prefix = I(smem.acc.ptr[0])
                dmask = I(smem.acc.ptr[1])
                for blk in range_constexpr(len(active)):
                    if active[blk]:
                        for k in range_constexpr(4):
                            key = keys[blk * 4 + k]
                            u = key ^ I(INT_MIN)
                            hit = key != I(INT_MIN)
                            if pi > 0:
                                hit = hit & ((u & dmask) == prefix)
                            bn = (u >> shift) & (nb - 1)
                            lds_add(smem.bins, base + hit.select(bn + ((bn >> 5) << 2), I(HPAD) + tid), I(1))
                fx.gpu.barrier()
                if wave == 0:
                    rem = I(smem.acc.ptr[2])
                    d, above, total = hist_scan(base, rem)
                    if lane == 0:
                        store_lds(smem.acc, 0, prefix | (d << shift))
                        store_lds(smem.acc, 1, dmask | I(((nb - 1) << shift) - ((1 << 32) if shift + width >= 32 else 0)))
                        store_lds(smem.acc, 2, rem - above)
                        if pi == 0:
                            store_lds(smem.acc, 3, total)
                else:
                    if pi < len(PASSES) - 1:
                        zero_bins(((pi + 1) % 2) * BUF, 64, NT - 64)
                fx.gpu.barrier()
                stamp(6 + pi)
            short = I(smem.acc.ptr[3]) <= TOPK
            return short.select(I(INT_MIN), I(smem.acc.ptr[0]) ^ I(INT_MIN)), short.select(I(0), I(smem.acc.ptr[2]))

        def emit(keys, idxs, active, kt, room):
            """Compacts the keys above ``kt`` and the first ``room`` equal to it (thread, then entry order) into the
            stage and stores it."""
            ngt, neq = I(0), I(0)
            for blk in range_constexpr(len(active)):
                if active[blk]:
                    for k in range_constexpr(4):
                        key = keys[blk * 4 + k]
                        ngt = ngt + (key > kt).select(I(1), I(0))
                        neq = neq + ((key == kt) & (key != I(INT_MIN))).select(I(1), I(0))
            pack = (ngt << 16) + neq
            inc = inclusive_scan(pack)
            if lane == 63:
                store_lds(smem.scan, wave, inc)
            fx.gpu.barrier()
            stamp(9)
            before = inc - pack
            for w in range_constexpr(NWAVES - 1):
                before = before + (I(w) < wave).select(I(smem.scan.ptr[w]), I(0))
            ab, eb = before >> 16, before & 0xFFFF
            for blk in range_constexpr(len(active)):
                if active[blk]:
                    for k in range_constexpr(4):
                        key = keys[blk * 4 + k]
                        gt = key > kt
                        eq = (key == kt) & (key != I(INT_MIN))
                        pos = (gt | (eq & (eb < room))).select(ab + fx.min(eb, room), I(TOPK))
                        store_lds(smem.sidx, pos, idxs[blk * 4 + k])
                        store_lds(smem.sval, pos, key_to_float(key).bitcast(I))
                        ab = ab + gt.select(I(1), I(0))
                        eb = eb + eq.select(I(1), I(0))
            fx.gpu.barrier()
            store_stage()
            stamp(5)

        def serve(b, cb, nch, keys_raw, idx_raw, mt):
            tb = list_size_rt(nch, cb)
            slot = slot_size(tb)
            npool = (nch - 1) * slot + cb
            keys, idxv = [], []
            for blk in range_constexpr(BLOCKS):
                ok = (blk * (NT * 4) + tid * 4) < npool
                keys += [ok.select(keys_raw[blk][k], I(INT_MIN)) for k in range(4)]
                idxv += [idx_raw[blk][k] for k in range(4)]
            if tid < NCH_MAX:
                store_lds(smem.tkey, tid, I(mt[0]))
                store_lds(smem.tval, tid, I(mt[1]) >> 16)
                store_lds(smem.tcnt, tid, I(mt[1]) & 0xFFFF)
            if wave == 0:
                inr = lane < nch
                top = key_to_float(wave_reduce(inr.select(I(mt[2]), I(INT_MIN)), fx.max))
                low = key_to_float(wave_reduce(inr.select(I(mt[3]), I(INT_MAX)), fx.min))
                span = top - low
                inv = (span > F(0.0)).select(F(TOPK) / span, F(0.0))
                listed_sum = readlane(inclusive_scan(inr.select(I(mt[1]) & 0xFFFF, I(0))), 63)
                if lane == 0:
                    store_lds(smem.acc, 8, inv.bitcast(I))
                    store_lds(smem.acc, 9, (F(0.0) - low * inv).bitcast(I))
                    store_lds(smem.acc, 12, listed_sum)
            fx.gpu.barrier()
            stamp(2)
            nblk = (npool + (NT * 4 - 1)) // (NT * 4)
            active = [I(blk) < nblk for blk in range_constexpr(BLOCKS)]
            inv = I(smem.acc.ptr[8]).bitcast(F)
            offset = I(smem.acc.ptr[9]).bitcast(F)
            listed_sum = I(smem.acc.ptr[12])

            def digit(key):
                """The key's bin of 2048 equal slices of the pool's score range (monotone in the key)."""
                return fx.min(fx.max(vop("v_fma_f32 $0, $1, $2, $3", key_to_float(key), inv, offset).to(I), I(0)), I(2047))

            # pass 0 over the keys aligned past their common high bits; wave 0 finds the threshold's bin
            for blk in range_constexpr(BLOCKS):
                if active[blk]:
                    for k in range_constexpr(4):
                        key = keys[blk * 4 + k]
                        d = digit(key)
                        lds_add(smem.bins, (key != I(INT_MIN)).select(d + ((d >> 5) << 2), I(HPAD) + tid), I(1))
            fx.gpu.barrier()
            if wave == 0:
                d, above, total = hist_scan(0, I(TOPK))
                short = listed_sum <= TOPK
                if lane == 0:
                    store_lds(smem.acc, 0, short.select(I(-1), d))
                    store_lds(smem.acc, 1, short.select(listed_sum, above))
                    store_lds(smem.acc, 2, short.select(I(0), I(smem.bins.ptr[d + ((d >> 5) << 2)])))
                    store_lds(smem.acc, 3, short.select(I(1), I(0)))
            fx.gpu.barrier()
            dstar, above_n, nc = I(smem.acc.ptr[0]), I(smem.acc.ptr[1]), I(smem.acc.ptr[2])
            stamp(6)
            if fx.const_expr(timing):
                if tid == 0:
                    buffer_ops.buffer_store(nc, dbg, row * 16 + 14)
                    buffer_ops.buffer_store(above_n, dbg, row * 16 + 15)
            if nc <= NCAND:
                # count, then place the keys above the bin and the bin's keys (the candidates)
                cg, cn = I(0), I(0)
                for blk in range_constexpr(BLOCKS):
                    if active[blk]:
                        for k in range_constexpr(4):
                            key = keys[blk * 4 + k]
                            d = digit(key)
                            valid = key != I(INT_MIN)
                            cg = cg + (valid & (d > dstar)).select(I(1), I(0))
                            cn = cn + (valid & (d == dstar)).select(I(1), I(0))
                stamp(7)
                pack = (cg << 16) + cn
                inc = inclusive_scan(pack)
                if lane == 63:
                    store_lds(smem.scan, wave, inc)
                fx.gpu.barrier()
                before = inc - pack
                for w in range_constexpr(NWAVES - 1):
                    before = before + (I(w) < wave).select(I(smem.scan.ptr[w]), I(0))
                stamp(8)
                pg, pcn = before >> 16, before & 0xFFFF
                for blk in range_constexpr(BLOCKS):
                    if active[blk]:
                        for k in range_constexpr(4):
                            key = keys[blk * 4 + k]
                            d = digit(key)
                            valid = key != I(INT_MIN)
                            is_gt = valid & (d > dstar)
                            is_c = valid & (d == dstar) & (pcn < NCAND)
                            pos = is_gt.select(pg, I(TOPK) + tid)
                            store_lds(smem.sidx, pos, idxv[blk * 4 + k])
                            store_lds(smem.sval, pos, key_to_float(key).bitcast(I))
                            if is_c:
                                store_lds(smem.ckey, pcn, key)
                                store_lds(smem.cidx, pcn, idxv[blk * 4 + k])
                            pg = pg + is_gt.select(I(1), I(0))
                            pcn = pcn + (valid & (d == dstar)).select(I(1), I(0))
                stamp(9)
                fx.gpu.barrier()
                need2 = I(TOPK) - above_n
                tc = I(0)
                for w in range_constexpr(NWAVES):
                    tc = tc + (I(smem.scan.ptr[w]) & 0xFFFF)
                if tid < tc:
                    mine = I(smem.ckey.ptr[tid])
                    for cj, st in range(0, (tc + 15) // 16, 1, init=[I(0)]):
                        c0 = I(cj) * 16
                        rank = st[0]
                        for q in range_constexpr(4):
                            v4 = lds_load4(smem.ckey, c0 + q * 4)
                            for e in range_constexpr(4):
                                other = I(v4[e])
                                j = c0 + q * 4 + e
                                rank = rank + ((j < tc) & ((other > mine) | ((other == mine) & (j < tid)))).select(I(1), I(0))
                        ranks = yield [rank]
                    if ranks < need2:
                        store_lds(smem.sidx, above_n + ranks, I(smem.cidx.ptr[tid]))
                        store_lds(smem.sval, above_n + ranks, key_to_float(mine).bitcast(I))
                    if ranks == need2 - 1:
                        store_lds(smem.acc, 10, mine)
                if tid == 0:
                    if tc == 0:
                        store_lds(smem.acc, 10, I(INT_MIN))
                stamp(10)
                fx.gpu.barrier()
            else:
                radix_init()
                kt_old, room_old = radix(keys, active)
                if tid == 0:
                    store_lds(smem.acc, 10, kt_old)
                    store_lds(smem.acc, 11, room_old)
                fx.gpu.barrier()
            kt = I(smem.acc.ptr[10])
            stamp(3)
            if wave == 0:
                tk = I(smem.tkey.ptr[lane])
                tv = I(smem.tval.ptr[lane])
                tn = I(smem.tcnt.ptr[lane])
                m = fx.Int64(rocdl.ballot(T.i64, (lane < nch - 1) & (tv > tn) & (tk > kt)))
                if lane == 0:
                    store_lds(smem.acc, 4, m.to(I))
                    store_lds(smem.acc, 5, (m >> I64(32)).to(I))
            fx.gpu.barrier()
            m_lo, m_hi = I(smem.acc.ptr[4]), I(smem.acc.ptr[5])
            stamp(4)
            if (m_lo | m_hi) == 0:
                if nc <= NCAND:
                    store_stage()
                    stamp(5)
                else:
                    emit(keys, idxv, active, kt, I(smem.acc.ptr[11]))
            else:
                ends_r = buffer_ops.create_buffer_resource_from_addr(arg_ends)
                end_r = I(buffer_ops.buffer_load(ends_r, row, vec_width=1, dtype=I))
                sc = Scorer(wave, lane, arg_q, arg_qs, arg_kv, arg_kvs, arg_bt, arg_w, i32_bt_stride, i32_page_stride,
                            i32_last_page, b, row, row + 1, [end_r], end_r)
                msat = (I64(m_hi) << I64(32)) | (I64(m_lo) & I64(0xFFFFFFFF))
                if tid == 0:
                    store_lds(smem.acc, 6, I(0))
                fx.gpu.barrier()
                for ii in range(0, nch, 1):
                    i = I(ii)
                    if ((msat >> I64(i)) & I64(1)) != I64(0):
                        ov = I(smem.acc.ptr[6])
                        score_chunk(sc, 1, CHUNK_MAX, NWAVES, cb, i, smem.ovf, ov * cb)
                        if tid == 0:
                            store_lds(smem.cof, ov, i)
                            store_lds(smem.acc, 6, ov + 1)
                        fx.gpu.barrier()
                nsat = I(smem.acc.ptr[6])
                keys2, idxs2 = [], []
                for blk in range_constexpr(BLOCKS):
                    li = fx.min((blk * (NT * 4) + tid * 4) // slot, I(63))
                    is_sat = ((msat >> I64(li)) & I64(1)) != I64(0)
                    for k in range_constexpr(4):
                        keys2.append(is_sat.select(I(INT_MIN), keys[blk * 4 + k]))
                        idxs2.append(idxv[blk * 4 + k])
                for s in range_constexpr(OVERFLOW):
                    e0 = s * (NT * 4) + tid * 4
                    ok = e0 < nsat * cb
                    v4 = lds_load4(smem.ovf, e0)
                    q = fx.min(e0 // cb, I(31))
                    base_idx = I(smem.cof.ptr[q]) * cb + (e0 - q * cb)
                    for k in range_constexpr(4):
                        keys2.append(ok.select(I(v4[k]), I(INT_MIN)))
                        idxs2.append(base_idx + k)
                radix_init()
                active2 = active + [I(s * (NT * 4)) < nsat * cb for s in range_constexpr(OVERFLOW)]
                kt2, room2 = radix(keys2, active2)
                emit(keys2, idxs2, active2, kt2, room2)

        stamp(0)
        if fx.const_expr(uniform):
            qsl_end = i32_batch * i32_qlen
        else:
            qsl_end = I(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_qsl), i32_batch, vec_width=1, dtype=I, is_scalar=True))
        inf = fx.Vector(buffer_ops.buffer_load(buffer_ops.create_buffer_resource_from_addr(arg_meta + INFO_OFFSET, num_records_bytes=NUM_REC), row * 4, vec_width=4, dtype=I, is_scalar=True))
        cb, nch, seq = I(inf[0]), I(inf[1]), I(inf[2])
        tb = list_size_rt(nch, cb)
        npool = (nch - 1) * slot_size(tb) + cb
        nblk = (npool + (NT * 4 - 1)) // (NT * 4)
        stamp(11)
        lk = buffer_ops.create_buffer_resource_from_addr(arg_keys + I64(row) * (POOL * 4), num_records_bytes=POOL * 4)
        lx = buffer_ops.create_buffer_resource_from_addr(arg_idx + I64(row) * (POOL * 4), num_records_bytes=POOL * 4)
        meta = buffer_ops.create_buffer_resource_from_addr(arg_meta, num_records_bytes=NUM_REC)
        zero4 = fx.Vector.from_elements([I(INT_MIN)] * 4, dtype=I)
        keys_raw, idx_raw = [], []
        for blk in range_constexpr(BLOCKS):
            kv4, iv4 = zero4, zero4
            if blk < nblk:
                kv4 = fx.Vector(buffer_ops.buffer_load(lk, blk * (NT * 4) + tid * 4, vec_width=4, dtype=I))
                iv4 = fx.Vector(buffer_ops.buffer_load(lx, blk * (NT * 4) + tid * 4, vec_width=4, dtype=I))
            keys_raw.append(kv4)
            idx_raw.append(iv4)
        mt = fx.Vector(buffer_ops.buffer_load(meta, (row * NCH_MAX + fx.min(tid, I(NCH_MAX - 1))) * 4, vec_width=4, dtype=I))
        init_stage()
        radix_init()
        stamp(1)
        if row < qsl_end:
            serve(seq, cb, nch, keys_raw, idx_raw, mt)
        else:
            store_stage()

    @flyc.jit
    def launch_merge(arg_q: fx.Int64, arg_qs: fx.Int64, arg_kv: fx.Int64, arg_kvs: fx.Int64, arg_bt: fx.Int64,
                     arg_w: fx.Int64, arg_ends: fx.Int64, arg_qsl: fx.Int64, arg_keys: fx.Int64, arg_idx: fx.Int64,
                     arg_meta: fx.Int64, arg_oidx: fx.Int64, arg_oval: fx.Int64, i32_bt_stride: fx.Int32,
                     i32_page_stride: fx.Int32, i32_last_page: fx.Int32, i32_batch: fx.Int32,
                     i32_oidx_stride: fx.Int32, i32_oval_stride: fx.Int32, i32_qlen: fx.Int32, arg_dbg: fx.Int64,
                     grid_x: fx.Int32, stream: fx.Stream):
        kernel(arg_q, arg_qs, arg_kv, arg_kvs, arg_bt, arg_w, arg_ends, arg_qsl, arg_keys, arg_idx, arg_meta,
               arg_oidx, arg_oval, i32_bt_stride, i32_page_stride, i32_last_page, i32_batch, i32_oidx_stride,
               i32_oval_stride, i32_qlen, arg_dbg).launch(grid=(I64(grid_x), 1, 1), block=(NT, 1, 1), stream=stream)

    launch_merge.kernel_name = name
    return launch_merge
