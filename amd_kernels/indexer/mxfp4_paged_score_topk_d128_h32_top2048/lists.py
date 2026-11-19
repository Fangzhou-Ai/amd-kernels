# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Chunk kernel: FP4 paged MQA scores of one chunk of a sequence's keys for its query rows, and each row's top list.

One workgroup (8 waves) a (sequence, chunk): the waves score the chunk's 64-key steps for every row of the sequence
into LDS (``scoring.py``), then wave ``r`` selects row ``r``'s list of the chunk (``geometry.py``: the ``tb`` largest
keys, or every live key of the sequence's last chunk) with a three pass radix select over the keys in LDS (11, 11 and
10 bits of the order-preserving key, left-aligned past the bits all keys share), compacts it lane-major into an LDS
stage (the wave's histogram, dead by then) and stores keys, then positions, as coalesced rows. Entries past a row's
live keys are ``INT_MIN``; ``meta`` records the list's smallest key (the chunk's ``tb``-th largest, or ``INT_MIN`` when
it holds every live key) and the chunk's live key count, which the merge kernel needs to know whether the list may miss
keys above its threshold. The scores are never stored.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import T

from ._wave import SC1, AtomicBinOp, atomic_agent, carve, layout, I, I64, INT_MAX, INT_MIN, ballot, inclusive_scan, lane_prefix, now, popcount, readlane, store_lds, wave_reduce
from ._wave import uniform as uniform_value
from .geometry import CHUNK_MAX, COUNTER_BYTES, FULL_STRIDE, INFO_OFFSET, NCH_MAX, POOL, QMAX, chunk_size_rt, list_size_rt, slot_size
from .scoring import SPEC_SIZES, TAG, Scorer, score_chunk

NW = 8
HPAD = 2304  # the histogram a 32-bin group per 36 words: the scan reads a lane's group conflict-free
HIST = HPAD + 64  # + one dummy word a lane, for keys outside the round's prefix
NUM_REC = 0x7FFFF000
import os
SKIP = int(os.environ.get('KK_SKIP', '0'))
FENCE_SCOPE = os.environ.get('KK_FENCE', 'wavefront')  # the select's LDS exchanges stay inside one wave


def chunk_parts(rows: int, uniform: bool = False, timing: bool = False, sync: bool = False):
    """``rows``: query rows a workgroup scores of a sequence (1 - 6; a sequence of more rows takes several workgroups
    a chunk)."""
    assert 1 <= rows <= QMAX
    OFFS, WORDS = layout(keys=rows * CHUNK_MAX, hist=rows * HIST)
    CM = (SC1 if os.environ.get('KK_CM', '1') == '1' else 0) if sync else 0  # stores a merge workgroup of another XCD reads: write-through

    @flyc.jit
    def role(smem, arg_q: fx.Int64, arg_qs: fx.Int64, arg_kv: fx.Int64, arg_kvs: fx.Int64, arg_bt: fx.Int64,
               arg_w: fx.Int64, arg_ends: fx.Int64, arg_qsl: fx.Int64, arg_keys: fx.Int64, arg_idx: fx.Int64,
               arg_meta: fx.Int64, arg_full: fx.Int64, i32_bt_stride: fx.Int32, i32_page_stride: fx.Int32,
               i32_last_page: fx.Int32, i32_batch: fx.Int32, i32_nch_grid: fx.Int32, i32_qlen: fx.Int32, i32_target: fx.Int32, arg_cnt: fx.Int64, i32_tiles: fx.Int32, arg_dbg: fx.Int64):
        tx = fx.thread_idx.x
        lane = tx % 64
        wave = I(rocdl.readfirstlane(T.i32, tx // 64))
        bid = I(fx.block_idx.x)
        dbg = buffer_ops.create_buffer_resource_from_addr(arg_dbg, num_records_bytes=NUM_REC)

        def stamp(k):
            if fx.const_expr(timing):
                t_ = now()
                if tx == 0:
                    buffer_ops.buffer_store(t_.to(I), dbg, bid * 16 + k)

        stamp(13)
        b = bid % i32_batch
        c = (bid // i32_batch) % i32_nch_grid
        tile = bid // (i32_batch * i32_nch_grid)
        qsl = buffer_ops.create_buffer_resource_from_addr(arg_qsl)
        bt_head = buffer_ops.create_buffer_resource_from_addr(arg_bt, num_records_bytes=I64(i32_batch * i32_bt_stride) * 4)
        spec = [I(buffer_ops.buffer_load(bt_head, b * i32_bt_stride + c * (size // 64) + wave, vec_width=1, dtype=I, is_scalar=True))
                for size in SPEC_SIZES]
        if fx.const_expr(uniform):
            first = b * i32_qlen
            last = first + i32_qlen
        else:
            first = uniform_value(buffer_ops.buffer_load(qsl, b, vec_width=1, dtype=I))
            last = uniform_value(buffer_ops.buffer_load(qsl, b + 1, vec_width=1, dtype=I))
        count = last - first
        ends_r = buffer_ops.create_buffer_resource_from_addr(arg_ends)

        def row_bound(j):
            return ((first + j) < last).select(uniform_value(buffer_ops.buffer_load(ends_r, fx.min(first + j, last - 1), vec_width=1, dtype=I)), I(0))

        max_end = I(0)
        for j in range_constexpr(QMAX):
            max_end = fx.max(max_end, row_bound(j))
        row0 = first + tile * rows
        ends = [row_bound(tile * rows + r) for r in range_constexpr(rows)]
        cb = chunk_size_rt(max_end, i32_target)
        nch = fx.max((max_end + cb - 1) // cb, I(1))
        tb = list_size_rt(nch, cb)
        slot = slot_size(tb)
        lkeys = buffer_ops.create_buffer_resource_from_addr(arg_keys, num_records_bytes=NUM_REC)
        lidx = buffer_ops.create_buffer_resource_from_addr(arg_idx, num_records_bytes=NUM_REC)
        meta = buffer_ops.create_buffer_resource_from_addr(arg_meta, num_records_bytes=NUM_REC)
        info = buffer_ops.create_buffer_resource_from_addr(arg_meta + INFO_OFFSET, num_records_bytes=NUM_REC)
        full = buffer_ops.create_buffer_resource_from_addr(arg_full, num_records_bytes=NUM_REC)
        sc = Scorer(wave, lane, arg_q, arg_qs, arg_kv, arg_kvs, arg_bt, arg_w, i32_bt_stride, i32_page_stride,
                    i32_last_page, b, row0, last, ends, max_end, spec)
        lds_copy4 = fx.make_copy_atom(fx.UniversalCopy128b(), I)
        def lds_load4(field, idx):
            reg = fx.make_rmem_tensor(4, I)
            fx.copy(lds_copy4, fx.make_view(fx.add_offset(field.ptr, idx), fx.make_layout(4, 1)), reg)
            return fx.Vector(reg.load())

        def lds_store4(field, idx, v):
            reg = fx.make_rmem_tensor(4, I)
            reg.store(v)
            fx.copy(lds_copy4, reg, fx.make_view(fx.add_offset(field.ptr, idx), fx.make_layout(4, 1)))

        def fence():
            fx.memory_fence(ordering=fx.AtomicOrdering.AcqRel, syncscope=FENCE_SCOPE)

        def hist_scan(wb, need):
            """The bin holding the ``need``-th largest key of the histogram at ``wb`` (a lane's 32 bins: 8 vector reads,
            then one 32-bin group re-read across lanes) and the number of keys in bins above it."""
            base = wb + (63 - lane) * 36
            ssum = None
            for m in range_constexpr(8):
                v4 = lds_load4(smem.hist, base + 4 * m)
                part = (I(v4[0]) + I(v4[1])) + (I(v4[2]) + I(v4[3]))
                ssum = part if ssum is None else ssum + part
            incl = inclusive_scan(ssum)
            lstar = I(fx.math.cttz(ballot(incl >= need)).to(I))
            above_g = readlane(incl - ssum, lstar)
            gsel = 63 - lstar
            h = I(smem.hist.ptr[wb + gsel * 36 + (31 - (lane & 31))])
            incl2 = inclusive_scan(h)
            jstar = I(fx.math.cttz(ballot((incl2 >= need - above_g) & (lane < 32))).to(I))
            return gsel * 32 + 31 - jstar, above_g + (readlane(incl2, jstar) - readlane(h, jstar))

        def zero_hist(wb):
            z = fx.Vector.from_elements([I(0)] * 4, dtype=I)
            for i in range_constexpr(HPAD // 256):
                lds_store4(smem.hist, wb + (lane + 64 * i) * 4, z)
            fence()

        def hist_add(wb, key, rnd, pref, sh):
            u = (key ^ I(INT_MIN)) << sh
            ok = key != I(INT_MIN)
            if fx.const_expr(rnd == 0):
                bn = (u >> 21) & 0x7FF
            elif fx.const_expr(rnd == 1):
                bn = (u >> 10) & 0x7FF
                ok = ok & (((u >> 21) & 0x7FF) == pref)
            else:
                bn = u & 0x3FF
                ok = ok & (((u >> 10) & 0x3FFFFF) == pref)
            idx = ok.select(bn + ((bn >> 5) << 2), I(HPAD) + lane)
            fx.atomic_add(fx.add_offset(smem.hist.ptr, wb + idx), I(1), syncscope="workgroup")

        def threshold(wb, region, nblk, need0, cap, kmax_l, kmin_l):
            """Selects the row's keys in LDS at ``region`` (``nblk`` 256-key blocks, more than ``cap`` live): the keys
            above ``kt`` and the first ``ties`` equal to it, at least ``need0`` and at most ``cap`` of them, the
            unlisted keys all below ``tk``. A pass whose threshold bin brings the list to at most ``cap`` keys ends
            the search; the last pass is exact (``need0`` keys). -> kt, ties, tk, listed."""
            umax = wave_reduce(kmax_l, fx.max) ^ I(INT_MIN)
            umin = wave_reduce(kmin_l, fx.min) ^ I(INT_MIN)
            nb = I(32) - I(fx.math.ctlz(umax ^ umin).to(I))
            sh = (nb == 0).select(I(0), I(32) - nb)
            full = (nb == 0) | (nb == 32)
            lowmask = full.select(I(-1), ~(I(-1) << nb))
            hi = full.select(I(0), umin & (I(-1) << nb))

            def key_of(aligned):
                return (((aligned >> sh) & lowmask) | hi) ^ I(INT_MIN)

            need, pref, fl = need0, I(0), I(0)
            kt, ties, tk, listed = I(INT_MIN), I(0), I(INT_MIN), I(0)
            for rnd in range_constexpr(3):
                if fl == 0:
                    zero_hist(wb)
                    for jj in range(0, nblk, 1):
                        v4 = lds_load4(smem.keys, region + I(jj) * 256 + lane * 4)
                        for k in range_constexpr(4):
                            hist_add(wb, I(v4[k]), rnd, pref, sh)
                    fence()
                    d, above = hist_scan(wb, need)
                    count_d = I(smem.hist.ptr[wb + d + ((d >> 5) << 2)])
                    newpref = d if rnd == 0 else (pref << (11 if rnd == 1 else 10)) | d
                    if fx.const_expr(rnd < 2):
                        bits = 11 if rnd == 0 else 22
                        low_key = key_of(newpref << (32 - bits))
                        listed_here = need0 - need + above + count_d
                        fin = listed_here <= cap
                        kt = fin.select((low_key == I(INT_MIN)).select(I(INT_MIN), low_key - 1), kt)
                        ties = fin.select(I(0), ties)
                        tk = fin.select(low_key, tk)
                        listed = fin.select(listed_here, listed)
                        fl = fin.select(I(1), fl)
                        need = fin.select(need, need - above)
                        pref = fin.select(pref, newpref)
                    else:
                        exact = key_of(newpref)
                        kt, ties, tk, listed, fl = exact, need - above, exact, need0, I(1)
            return kt, ties, tk, listed

        def select_row(r, cc):
            row = row0 + r
            end_r = ends[0]
            for rr in range_constexpr(1, rows):
                end_r = (r == rr).select(ends[rr], end_r)
            wb = r * HIST
            region = r * CHUNK_MAX
            nblk = cb // 256
            last_chunk = cc == nch - 1
            cap = last_chunk.select(cb, slot)
            nvalid = fx.max(fx.min(end_r - cc * cb, cb), I(0))
            stamp(3)
            lbase = row * POOL + cc * slot
            both = cap * 2 + 64 <= HIST

            def stats_pass():
                """The extremes of the live keys of the row's chunk."""
                for jj, st in range(0, nblk, 1, init=[I(INT_MIN), I(INT_MAX)]):
                    v4 = lds_load4(smem.keys, region + I(jj) * 256 + lane * 4)
                    kmax, kmin = st[0], st[1]
                    for k in range_constexpr(4):
                        key = I(v4[k])
                        kmax = fx.max(kmax, key)
                        kmin = fx.min(kmin, (key != I(INT_MIN)).select(key, I(INT_MAX)))
                    stats = yield [kmax, kmin]
                return wave_reduce(stats[0], fx.max), wave_reduce(stats[1], fx.min), stats[0], stats[1]

            def compact(kt, ties, keys_at, idx_at, dummy_at):
                """The keys above ``kt`` (at most ``cap``), then the first ``ties`` equal to it, in block / key / lane
                order, into the stage: keys at ``keys_at`` and positions at ``idx_at`` (None: not written). -> the keys
                above ``kt``."""
                for jj, st in range(0, nblk, 1, init=[I(0)]):
                    j = I(jj)
                    rg = st[0]
                    v4 = lds_load4(smem.keys, region + j * 256 + lane * 4)
                    for k in range_constexpr(4):
                        key = I(v4[k])
                        gt = key > kt
                        m = ballot(gt)
                        rank = rg + lane_prefix(m)
                        put = gt & (rank < cap)
                        if keys_at is not None:
                            store_lds(smem.hist, wb + put.select(keys_at + rank, dummy_at + lane), key)
                        if idx_at is not None:
                            store_lds(smem.hist, wb + put.select(idx_at + rank, dummy_at + lane), cc * cb + j * 256 + lane * 4 + k)
                        rg = rg + popcount(m)
                    above_total = yield [rg]
                if ties > 0:
                    for jj, st in range(0, nblk, 1, init=[I(0)]):
                        j = I(jj)
                        re = st[0]
                        v4 = lds_load4(smem.keys, region + j * 256 + lane * 4)
                        for k in range_constexpr(4):
                            key = I(v4[k])
                            eq = (key == kt) & (key != I(INT_MIN))
                            m = ballot(eq)
                            p = re + lane_prefix(m)
                            take = eq & (p < ties)
                            if keys_at is not None:
                                store_lds(smem.hist, wb + take.select(keys_at + above_total + p, dummy_at + lane), key)
                            if idx_at is not None:
                                store_lds(smem.hist, wb + take.select(idx_at + above_total + p, dummy_at + lane), cc * cb + j * 256 + lane * 4 + k)
                            re = re + popcount(m)
                        tied = yield [re]
                return above_total

            def pad_stage(at):
                pad = fx.Vector.from_elements([I(INT_MIN)] * 4, dtype=I)
                for ii in range(0, (cap + 255) // 256, 1):
                    lds_store4(smem.hist, wb + at + (I(ii) * 64 + lane) * 4, pad)

            def store_plane(plane, at):
                for ii in range(0, (cap + 255) // 256, 1):
                    off = (I(ii) * 64 + lane) * 4
                    buffer_ops.buffer_store(lds_load4(smem.hist, wb + at + off), plane, lbase + off, mask=off < cap, cache_modifier=CM)

            def emit_list(kt, ties):
                """Builds the list of the keys above ``kt`` (and ``ties`` equal to it) and stores it. -> the keys above
                ``kt`` (more than ``cap``: the list is cut and the caller redoes it), the largest key."""
                total = I(0)
                if both:
                    pad_stage(0)
                    fence()
                    total = compact(kt, ties, 0, cap, cap * 2)
                    fence()
                    stamp(6)
                    store_plane(lkeys, 0)
                    store_plane(lidx, cap)
                else:
                    pad_stage(0)
                    fence()
                    total = compact(kt, ties, 0, None, cap)
                    fence()
                    store_plane(lkeys, 0)
                    fence()
                    compact(kt, ties, None, 0, cap)
                    fence()
                    store_plane(lidx, 0)
                return total

            tk, listed, chunk_max, chunk_min, total = I(INT_MIN), nvalid, I(INT_MIN), I(INT_MAX), I(0)
            if nvalid > cap:
                chunk_max, chunk_min, smax_, smin_ = stats_pass()
                kt, ties, tk, listed = threshold(wb, region, nblk, tb, cap, smax_, smin_)
                stamp(4)
                total = emit_list(kt, ties)
            else:
                chunk_max, chunk_min, smax_, smin_ = stats_pass()
                stamp(4)
                total = emit_list(I(INT_MIN), I(0))
            stamp(7)
            if lane == 0:
                buffer_ops.buffer_store(fx.Vector.from_elements([tk, (nvalid << 16) | listed, chunk_max, chunk_min], dtype=I), meta, (row * NCH_MAX + cc) * 4, cache_modifier=CM)
                if cc == 0:
                    buffer_ops.buffer_store(fx.Vector.from_elements([cb, nch, b, I(0)], dtype=I), info, row * 4, cache_modifier=CM)
            if nvalid > cap:
                for jj in range(0, nblk, 1):
                    buffer_ops.buffer_store(lds_load4(smem.keys, region + I(jj) * 256 + lane * 4), full,
                                            row * FULL_STRIDE + cc * cb + I(jj) * 256 + lane * 4, cache_modifier=CM)

        if (tile * rows < count) & ((c * cb < max_end) | (c == 0)):
            stamp(0)
            if fx.const_expr(SKIP != 2):
                score_chunk(sc, rows, CHUNK_MAX, NW, cb, c, smem.keys, 0, stamp if timing else None)
            fx.gpu.barrier()
            stamp(1)
            if fx.const_expr(SKIP == 1):
                if wave == 0:
                    buffer_ops.buffer_store(I(smem.keys.ptr[lane]), meta, c)
            if fx.const_expr(SKIP != 1) and (wave < rows) & (tile * rows + wave < count):
                select_row(wave, c)
            stamp(2)
        if fx.const_expr(sync):
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            if tx == 0:
                atomic_agent(AtomicBinOp.add, arg_cnt + I64((b * i32_tiles + tile) * COUNTER_BYTES), 1)

    return role, OFFS, WORDS

@launcher
def compile_chunk_lists(rows: int, uniform: bool = False, timing: bool = False):
    """``rows``: query rows a workgroup scores of a sequence (1 - 6; a sequence of more rows takes several workgroups
    a chunk)."""
    assert 1 <= rows <= QMAX
    name = f"mxfp4_paged_chunk_lists_d128_h32_r{rows}_{TAG}" + ("_u" if uniform else "") + ("_t" if timing else "")
    role, OFFS, WORDS = chunk_parts(rows, uniform, timing)

    @fx.struct
    class Raw:
        u: fx.Array[fx.Int32, WORDS]

    @flyc.kernel(name=name, known_block_size=[NW * 64, 1, 1])
    def kernel(arg_q: fx.Int64, arg_qs: fx.Int64, arg_kv: fx.Int64, arg_kvs: fx.Int64, arg_bt: fx.Int64,
               arg_w: fx.Int64, arg_ends: fx.Int64, arg_qsl: fx.Int64, arg_keys: fx.Int64, arg_idx: fx.Int64,
               arg_meta: fx.Int64, arg_full: fx.Int64, i32_bt_stride: fx.Int32, i32_page_stride: fx.Int32,
               i32_last_page: fx.Int32, i32_batch: fx.Int32, i32_nch_grid: fx.Int32, i32_qlen: fx.Int32, i32_target: fx.Int32, arg_dbg: fx.Int64):
        smem = carve(fx.SharedAllocator().allocate(Raw).peek().u, **OFFS)
        role(smem, arg_q, arg_qs, arg_kv, arg_kvs, arg_bt, arg_w, arg_ends, arg_qsl, arg_keys, arg_idx, arg_meta, arg_full,
             i32_bt_stride, i32_page_stride, i32_last_page, i32_batch, i32_nch_grid, i32_qlen, i32_target, I64(0), I(0), arg_dbg)

    @flyc.jit
    def launch_chunk_lists(arg_q: fx.Int64, arg_qs: fx.Int64, arg_kv: fx.Int64, arg_kvs: fx.Int64, arg_bt: fx.Int64,
                           arg_w: fx.Int64, arg_ends: fx.Int64, arg_qsl: fx.Int64, arg_keys: fx.Int64,
                           arg_idx: fx.Int64, arg_meta: fx.Int64, arg_full: fx.Int64, i32_bt_stride: fx.Int32,
                           i32_page_stride: fx.Int32, i32_last_page: fx.Int32, i32_batch: fx.Int32, i32_nch_grid: fx.Int32, i32_qlen: fx.Int32,
                           i32_target: fx.Int32, arg_dbg: fx.Int64, grid_x: fx.Int32, stream: fx.Stream):
        kernel(arg_q, arg_qs, arg_kv, arg_kvs, arg_bt, arg_w, arg_ends, arg_qsl, arg_keys, arg_idx, arg_meta, arg_full,
               i32_bt_stride, i32_page_stride, i32_last_page, i32_batch, i32_nch_grid, i32_qlen, i32_target, arg_dbg).launch(
            grid=(I64(grid_x), 1, 1), block=(NW * 64, 1, 1), stream=stream)

    launch_chunk_lists.kernel_name = name
    return launch_chunk_lists
