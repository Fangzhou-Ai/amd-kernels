# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""q_b projection, push all-gather of q and W_UK absorption of all 96 heads in one launch.

Workgroups [0, g1) are producers: a (16 mt1 tokens x 16 nt1 columns) tile of q = q_a x q_b^T (four waves split K, an LDS
reduce, one bf16 rounding) stored into the gather buffers of all eight ranks, rows [M, 96, 192]. Workgroups [g1, g1 +
96 ns2) are consumers: one head and a slice of the latent columns, the W_UK slice in registers, the head's q of every
token staged in LDS, ``[bf16(q_nope @ W_UK) | q_pe]`` stored to out [M, 96, 576]. A consumer loads its weight, waits
for the producer tiles that wrote its head (flags.py); the producers have lower ids, so they are all dispatched before a
consumer holds a slot. ``timing`` stamps ``s_memrealtime`` (100 MHz) per workgroup into the debug buffer (the benchmark
timeline), ``skeleton`` replaces the MFMAs by XORs of their operands (the empirical floor of the dependence chain),
``c_delay`` holds the consumers' weight loads back (x 1.1 us), ``nph`` stages the consumers' q rows in that many token
phases (LDS), ``pf_cap`` bounds the q_b steps whose loads are in flight."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from .flags import (
    DATA_OFF,
    FREE_OFF,
    LINE,
    Q_AREA,
    READY_OFF,
    WORDS,
    flag_read,
    flag_store,
    sel8,
    spin,
    spin_from,
    ticket_add,
)
from .kernel_qb import K, KSTEPS, N, NT, PAD
from .kernel_wuk import GROUPS, KS, L, OW, P, QW, SYS
from .ops import mfma, mfma_stub, now, pk_bf16

NTHR = 256
PF_CAP = 4
QB = 6
ROW16 = 25
PRELOAD_HINTS = {"llvm_options": {"amdgpu-kernarg-preload": True, "amdgpu-kernarg-preload-count": 16}}


def grid_fused(rows, mt1, nt1, ns2):
    g1 = ((rows + 16 * mt1 - 1) // (16 * mt1)) * (N // (16 * nt1))
    return g1, 96 * ns2


@launcher
def compile_q_gather_wuk(mt1, nt1, ns2, mmax, timing=False, skeleton=False, c_delay=0, nph=1, pf_cap=PF_CAP, wpe=1):
    """mt1, nt1: q_b tile (16-token and 16-column tiles); ns2: consumer workgroups per head; mmax: the row bound of the
    variant (a multiple of 16)."""
    nw = 4
    mtok, cols, wsteps = 16 * mt1, 16 * nt1, KSTEPS // nw
    pf = max(2, min(pf_cap, 36 // (nt1 + mt1)))
    ncb = N // cols
    ncbh = QW // cols
    assert QW % cols == 0 and mmax % 16 == 0 and ns2 in (1, 2, 4, 8)
    ntb_max = (mmax + mtok - 1) // mtok
    assert ntb_max * ncb <= WORDS
    items1 = (mtok * (cols // 8) + NTHR - 1) // NTHR
    row4a = (cols + PAD) // 4
    gper = GROUPS // ns2
    gpw = max(1, gper // nw)
    wpg = max(1, nw // gper)
    mph = (((mmax + nph - 1) // nph) + 15) // 16 * 16
    qitems = (mph * 24 + NTHR - 1) // NTHR
    wait_iters = (ntb_max * ncbh + NTHR - 1) // NTHR
    name = (
        f"bf16_q_b_gather_w_uk_h96_mt{mt1}_nt{nt1}_ns{ns2}_m{mmax}"
        + ("_timing" if timing else "")
        + ("_skeleton" if skeleton else "")
        + (f"_cd{c_delay}" if c_delay else "")
        + (f"_ph{nph}" if nph > 1 else "")
        + (f"_pf{pf_cap}" if pf_cap != PF_CAP else "")
    )
    mac = mfma_stub if skeleton else mfma

    @fx.struct
    class Shared:
        m: fx.Array[fx.Int32, max(nw * mtok * (cols + PAD), mph * ROW16 * 4), 16]

    @flyc.kernel(name=name, known_block_size=[NTHR, 1, 1])
    def kernel(
        x: fx.Int64,
        sx: fx.Int32,
        w_qb: fx.Int64,
        w_uk: fx.Int64,
        out: fx.Int64,
        rows: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        me: fx.Int32,
        state: fx.Int64,
        g1: fx.Int32,
        dbg: fx.Int64,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        bases = (b0, b1, b2, b3, b4, b5, b6, b7)
        mine = sel8(me, bases)
        sr = buffer_ops.create_buffer_resource_from_addr(state, num_records_bytes=16)
        epoch = fx.Int32(buffer_ops.buffer_load(sr, 0, vec_width=1, dtype=fx.Int32, is_scalar=True))
        par = DATA_OFF + fx.Int64(epoch & 1) * Q_AREA
        atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        sm = fx.SharedAllocator().allocate(Shared).peek()
        dr = buffer_ops.create_buffer_resource_from_addr(dbg, num_records_bytes=65536 * 64)

        def stamp(k):
            if timing:
                t_ = now()
                if tx == 0:
                    buffer_ops.buffer_store(
                        Vec.from_elements([t_.to(fx.Int32), (t_ >> fx.Int64(32)).to(fx.Int32)], fx.Int32),
                        dr,
                        bid * 16 + 2 * k,
                    )

        stamp(0)
        if bid < g1:
            if (bid == 0) & (tx < 8):
                flag_store(sel8(tx, bases) + FREE_OFF + fx.Int64(me * (LINE * 4)), epoch + 1)
            ffree = flag_read(mine + FREE_OFF + fx.Int64((tx % 8) * (LINE * 4)))
            cb = bid % ncb
            tb = bid // ncb
            vp = fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, sm.m.ptr), fx.make_layout(nw * mtok * (cols + PAD), 1)),
                fx.make_layout(4, 1),
            )
            wr = buffer_ops.create_buffer_resource_from_addr(w_qb, num_records_bytes=N * K * 2)
            xr = buffer_ops.create_buffer_resource_from_addr(x, num_records_bytes=fx.Int64(rows) * fx.Int64(sx * 2))

            def operands(s):
                step = wave * wsteps + s
                av = [
                    Vec(
                        buffer_ops.buffer_load(
                            wr, ((cb * nt1 + j) * KSTEPS + step) * 256 + lane * 4, vec_width=4, dtype=fx.Int32
                        )
                    )
                    for j in range_constexpr(nt1)
                ]
                bv = []
                for i in range_constexpr(mt1):
                    tok = tb * mtok + i * 16 + lane % 16
                    bv.append(
                        Vec(
                            buffer_ops.buffer_load(
                                xr,
                                tok * (sx // 2) + step * 16 + (lane // 16) * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=tok < rows,
                            )
                        )
                    )
                return av + bv

            acc = [Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(mt1 * nt1)]
            ops = [operands(s) for s in range_constexpr(min(pf, wsteps))]
            fx.rocdl.sched_barrier(0)
            if timing:
                fx.rocdl.s_waitcnt(vmcnt=0)
                stamp(7)
            for s in range_constexpr(wsteps):
                if s + pf < wsteps:
                    ops.append(operands(s + pf))
                cur = ops[s]
                for i in range_constexpr(mt1):
                    for j in range_constexpr(nt1):
                        acc[i * nt1 + j] = mac(cur[j], cur[nt1 + i], acc[i * nt1 + j])
            for i in range_constexpr(mt1):
                for j in range_constexpr(nt1):
                    tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    tmp.store(acc[i * nt1 + j].bitcast(fx.Int32))
                    fx.copy(
                        atom, tmp, fx.slice(vp, (None, (wave * mtok + i * 16 + lane % 16) * row4a + j * 4 + lane // 16))
                    )
            stamp(1)
            if tx < 8:
                spin_from(mine + FREE_OFF + fx.Int64(tx * (LINE * 4)), epoch, ffree)
            fx.gpu.barrier()
            stamp(2)
            for r in range_constexpr(items1):
                item = tx + r * NTHR
                tk = item // (cols // 8)
                ch = item % (cols // 8)
                tok = tb * mtok + tk
                live = (tk < mtok) & (tok < rows)
                sums = [None] * 8
                for w in range_constexpr(nw):
                    for h in range_constexpr(2):
                        tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(atom, fx.slice(vp, (None, (w * mtok + tk) * row4a + ch * 2 + h)), tmp)
                        v = Vec(tmp.load()).bitcast(fx.Float32)
                        for e in range_constexpr(4):
                            a = fx.Float32(v[e])
                            sums[h * 4 + e] = a if w == 0 else sums[h * 4 + e] + a
                packed = Vec.from_elements(
                    [pk_bf16(sums[2 * q], sums[2 * q + 1]) for q in range_constexpr(4)], fx.Int32
                )
                for p in range_constexpr(8):
                    dstp = buffer_ops.create_buffer_resource_from_addr(
                        bases[p] + par, num_records_bytes=fx.Int64(rows) * (96 * QW * 2)
                    )
                    buffer_ops.buffer_store(
                        packed, dstp, tok * (96 * 96) + me * 1152 + cb * (cols // 2) + ch * 4, mask=live
                    )
            stamp(3)
            fx.rocdl.s_waitcnt(vmcnt=0)
            fx.gpu.barrier()
            stamp(4)
            if tx < 8:
                flag_store(sel8(tx, bases) + READY_OFF + fx.Int64((me * WORDS + tb * ncb + cb) * 4), epoch + 1)
            stamp(5)
        else:
            cid = bid - g1
            sl = cid % ns2
            h = cid // ns2
            v16 = fx.logical_divide(
                fx.make_view(fx.recast_iter(fx.Int32, sm.m.ptr), fx.make_layout(mph * ROW16 * 4, 1)),
                fx.make_layout(4, 1),
            )
            for _ in range_constexpr(c_delay):
                fx.rocdl.s_sleep(40)
            wr2 = buffer_ops.create_buffer_resource_from_addr(w_uk, num_records_bytes=96 * L * P * 2)
            gq = buffer_ops.create_buffer_resource_from_addr(
                mine + par, num_records_bytes=fx.Int64(rows) * (96 * QW * 2)
            )
            dst = buffer_ops.create_buffer_resource_from_addr(out, num_records_bytes=fx.Int64(rows) * (96 * OW * 2))
            gw = wave % gper if gper < 4 else wave
            aw = []
            for gi in range_constexpr(gpw):
                grp = sl * gper + gw * gpw + gi
                for jj in range_constexpr(4):
                    for s in range_constexpr(KS):
                        aw.append(
                            Vec(
                                buffer_ops.buffer_load(
                                    wr2,
                                    (((h * GROUPS + grp) * 4 + jj) * KS + s) * 256 + lane * 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    cache_modifier=NT,
                                )
                            )
                        )
            stamp(1)
            ntb = (rows + mtok - 1) // mtok
            nwait = ntb * ncbh
            for jw in range_constexpr(wait_iters):
                w = tx + jw * NTHR
                if w < nwait:
                    spin(
                        mine
                        + READY_OFF
                        + fx.Int64(((h // 12) * WORDS + (w // ncbh) * ncb + (h % 12) * ncbh + w % ncbh) * 4),
                        epoch + 1,
                    )
            fx.gpu.barrier()
            stamp(2)
            part = wave // gper if gper < 4 else wave * 0
            for ph in range_constexpr(nph):
                t0 = ph * mph
                for r0 in range_constexpr((qitems + QB - 1) // QB):
                    loaded = []
                    for r in range_constexpr(min(QB, qitems - r0 * QB)):
                        item = tx + (r0 * QB + r) * NTHR
                        tk = item // 24
                        c = item % 24
                        loaded.append(
                            Vec(
                                buffer_ops.buffer_load(
                                    gq,
                                    (t0 + tk) * (96 * QW // 2) + h * (QW // 2) + c * 4,
                                    vec_width=4,
                                    dtype=fx.Int32,
                                    mask=(tk < mph) & (t0 + tk < rows),
                                    cache_modifier=SYS,
                                )
                            )
                        )
                    for r in range_constexpr(min(QB, qitems - r0 * QB)):
                        item = tx + (r0 * QB + r) * NTHR
                        tk = item // 24
                        c = item % 24
                        if (tk < mph) & (t0 + tk < rows):
                            tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            tmp.store(loaded[r])
                            fx.copy(atom, tmp, fx.slice(v16, (None, tk * ROW16 + c)))
                fx.gpu.barrier()
                for ti in range_constexpr(mph // 16):
                    if (t0 + ti * 16 < rows) & ((ti % wpg) == part):
                        bq = []
                        for s in range_constexpr(KS):
                            tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(
                                atom, fx.slice(v16, (None, (ti * 16 + lane % 16) * ROW16 + s * 4 + lane // 16)), tmp
                            )
                            bq.append(Vec(tmp.load()))
                        tok = t0 + ti * 16 + lane % 16
                        for gi in range_constexpr(gpw):
                            grp = sl * gper + gw * gpw + gi
                            packed = []
                            for jj in range_constexpr(4):
                                ac = Vec.filled(4, 0.0, fx.Float32)
                                for s in range_constexpr(KS):
                                    ac = mac(aw[(gi * 4 + jj) * KS + s], bq[s], ac)
                                packed += [pk_bf16(ac[2 * z], ac[2 * z + 1]) for z in range_constexpr(2)]
                            for z in range_constexpr(2):
                                buffer_ops.buffer_store(
                                    Vec.from_elements(packed[4 * z : 4 * z + 4], fx.Int32),
                                    dst,
                                    tok * (96 * OW // 2) + h * (OW // 2) + grp * 32 + z * 16 + (lane // 16) * 4,
                                    mask=tok < rows,
                                )
                for r in range_constexpr((mph * 8 + NTHR - 1) // NTHR):
                    item = tx + r * NTHR
                    tk = item // 8
                    ch = item % 8
                    gt = t0 + tk
                    live = (tk < mph) & (gt < rows) & ((gt % ns2) == sl)
                    if live:
                        tmp = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(atom, fx.slice(v16, (None, tk * ROW16 + 16 + ch)), tmp)
                        buffer_ops.buffer_store(
                            Vec(tmp.load()), dst, gt * (96 * OW // 2) + h * (OW // 2) + 256 + ch * 4, mask=live
                        )
                if ph + 1 < nph:
                    fx.gpu.barrier()
            stamp(3)
        stamp(6)
        fx.gpu.barrier()
        if tx == 0:
            old = ticket_add(state + 4)
            if old + 1 == g1 + 96 * ns2:
                flag_store(state + 4, 0)
                flag_store(state, epoch + 1)

    @flyc.jit
    def launch_fused(
        x: fx.Int64,
        sx: fx.Int32,
        w_qb: fx.Int64,
        w_uk: fx.Int64,
        out: fx.Int64,
        rows: fx.Int32,
        b0: fx.Int64,
        b1: fx.Int64,
        b2: fx.Int64,
        b3: fx.Int64,
        b4: fx.Int64,
        b5: fx.Int64,
        b6: fx.Int64,
        b7: fx.Int64,
        me: fx.Int32,
        state: fx.Int64,
        g1: fx.Int32,
        dbg: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            x,
            sx,
            w_qb,
            w_uk,
            out,
            rows,
            b0,
            b1,
            b2,
            b3,
            b4,
            b5,
            b6,
            b7,
            me,
            state,
            g1,
            dbg,
            value_attrs={"rocdl.waves_per_eu": wpe, "rocdl.flat_work_group_size": f"{NTHR},{NTHR}"},
        ).launch(grid=(fx.Int64(g1 + 96 * ns2), 1, 1), block=(NTHR, 1, 1), stream=stream)

    launch_fused.compile_hints = PRELOAD_HINTS
    launch_fused.kernel_name = name
    return launch_fused
