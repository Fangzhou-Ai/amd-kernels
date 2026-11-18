# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA spec-verify decode, one workgroup (4 waves) per (sequence, head): f_b gate GEMV, causal conv + SiLU, delta rule
over the sequence's tokens with the 128 x 128 fp32 state in registers, per-token state snapshots, gated RMSNorm.

Prologue (all tokens of the sequence, independent of the state): the state loads are issued first; thread (d, kh) of the
gate GEMV holds 64 bf16 of f_b row d in registers and dots them with f_a (packed bf16 dot2); the q / k / v conv + SiLU
and the gate a go to LDS as [token][q, k, v, a][128]. Token loop: lane (g, s) of wave w owns state rows
v = 32 w + 8 g + i (i < 8), columns 4 s + {0..3, 64..67}; the row reductions are 16-lane DPP butterflies; the sum of
squares of o crosses waves through LDS (one barrier per token).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import (
    bf16_scalar_to_f32,
    bf16x2_hi,
    bf16x2_lo,
    dpp,
    fdot2_bf16,
    pk_bf16,
    rsqrt,
    round_bf16,
    sigmoid,
    exp2,
    sum_group,
    wave_sum_lane63,
    mfma_bf16,
    fma,
)

NH = 12
D = 128
LP = NH * D
COL_GATE = 3 * LP
COL_FA = 4 * LP
COL_BETA = COL_FA + D
LOG2E = 1.4426950408889634
SCALE = D ** -0.5
EPS_QK = 1e-6


def compile_kda_decode(max_tok, nw_f32=False, waves=4, lpr=8, store_cm=2, snap=False):
    block = 64 * waves
    rg = 64 // lpr  # state rows a wave covers per instruction
    rpl = D // (waves * rg)  # state rows per lane
    cpl = D // lpr  # columns per lane and row
    nvec = cpl // 4
    nt_n = D // 16 // waves  # gate column tiles (16 f_b rows) per wave
    name = f"kda_decode_fb_gate_spec_h12_d128_t{max_tok}_w{waves}_l{lpr}_cm{store_cm}" + ("_snap" if snap else "") + ("_nwf32" if nw_f32 else "")

    @fx.struct
    class Shared:
        qkva: fx.Array[fx.Float32, max_tok * 5 * D]  # [token][q, k, v, a, scale][128]
        scal: fx.Array[fx.Float32, 2 * max_tok]  # [token][qk, beta]
        part: fx.Array[fx.Float32, 2 * waves]

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def kernel(
        arg_proj: fx.Int64,  # [M, >= 6288] bf16
        i32_pstride: fx.Int32,  # proj row stride, elements (even)
        i32_proj_bytes: fx.Int32,
        arg_fbw: fx.Int64,  # [1536, 128] bf16
        arg_alog: fx.Int64,  # [12] fp32
        arg_dtb: fx.Int64,  # [1536] fp32
        arg_cw: fx.Int64,  # [3, 4, 1536] fp32
        arg_cs: fx.Int64,  # conv state [slots, 4608, P] bf16
        i64_cs_slot_bytes: fx.Int64,
        i32_cs_dim: fx.Int32,  # elements
        i32_cs_pos: fx.Int32,
        arg_state: fx.Int64,  # [slots, 12, 128, 128] fp32
        i64_state_slot_bytes: fx.Int64,
        arg_cu: fx.Int64,  # [N + 1] int32
        arg_idx: fx.Int64,  # [N, >= max_tok] int32
        i32_idx_stride: fx.Int32,
        arg_acc: fx.Int64,  # [N] int32
        arg_snap: fx.Int64,  # [N] int32 token whose state is written (snap)
        arg_nw: fx.Int64,  # [128] bf16 / fp32
        arg_out: fx.Int64,  # [M, 12, 128] bf16
        i32_out_stride: fx.Int32,  # elements per token
        i32_out_bytes: fx.Int32,
        f32_lbc: fx.Float32,  # lower_bound * log2(e)
        f32_eps: fx.Float32,
    ):
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lds = smem.qkva.ptr
        lds_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        lds_v = fx.logical_divide(fx.make_view(smem.qkva.ptr, fx.make_layout(max_tok * 5 * D, 1)), fx.make_layout(4, 1))

        def ld4(idx):
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
            fx.copy(lds_atom, fx.slice(lds_v, (None, idx)), r)
            return Vec(r.load())

        scal = smem.scal.ptr
        part = smem.part.ptr
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        i_n = bid // NH
        hv = bid % NH
        lane = tx % 64
        w = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        s = lane % lpr
        rgi = lane // lpr

        def loadn(rsrc, off, n, dtype):
            x = buffer_ops.buffer_load(rsrc, off, vec_width=n, dtype=dtype)
            out = [x]
            if const_expr(n > 1):
                xv = Vec(x)
                out = [xv[k] for k in range_constexpr(n)]
            return out

        def put(ptr, idx, val):
            ptr[idx] = val

        def sload(rsrc, off):
            return fx.Int32(buffer_ops.buffer_load(rsrc, off, vec_width=1, is_scalar=True))

        cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
        idx_r = buffer_ops.create_buffer_resource_from_addr(arg_idx)
        acc_r = buffer_ops.create_buffer_resource_from_addr(arg_acc)
        n0 = sload(cu_r, i_n)
        n_tok = sload(cu_r, i_n + 1) - n0
        n_acc = sload(acc_r, i_n)
        snap_pos = sload(buffer_ops.create_buffer_resource_from_addr(arg_snap), i_n) if snap else fx.Int32(0)
        slots = []
        for h in range_constexpr((max_tok + 3) // 4):
            sv = buffer_ops.buffer_load(idx_r, i_n * i32_idx_stride + 4 * h, vec_width=4, is_scalar=True)
            sv = Vec(sv)
            slots += [fx.Int32(sv[k]) for k in range_constexpr(4)]

        def pick(vals, i):
            r = vals[len(vals) - 1]
            for k in range_constexpr(len(vals) - 2, -1, -1):
                r = (i == k).select(vals[k], r)
            return r

        out_r = buffer_ops.create_buffer_resource_from_addr(
            arg_out, num_records_bytes=fx.Int64(i32_out_bytes)
        )

        # loads that need only the sequence's first row: issued before the slot lookups return
        pr = buffer_ops.create_buffer_resource_from_addr(arg_proj, num_records_bytes=i32_proj_bytes)
        a_exp = fx.math.exp(fx.Float32(buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(arg_alog, num_records_bytes=NH * 4), hv, vec_width=1)))
        # gate = f_a @ f_b^T on the matrix cores: lane (i, q) holds A[token i][32 kb + 8 q ..] and B[32 kb + 8 q ..][row 16 nt + i]
        fbr = buffer_ops.create_buffer_resource_from_addr(arg_fbw, num_records_bytes=LP * D * 2)
        l16 = lane % 16
        qd = lane // 16
        nts = [w * nt_n + j for j in range_constexpr(nt_n)]
        afr = [
            Vec(buffer_ops.buffer_load(pr, ((n0 + l16) * i32_pstride + COL_FA + 32 * kb + 8 * qd) // 2, vec_width=4, dtype=fx.Int32))
            for kb in range_constexpr(4)
        ]
        bfr = [
            [Vec(buffer_ops.buffer_load(fbr, ((hv * D + 16 * nt + l16) * D + 32 * kb + 8 * qd) // 2, vec_width=4, dtype=fx.Int32))
             for kb in range_constexpr(4)]
            for nt in nts
        ]
        dtbr = buffer_ops.create_buffer_resource_from_addr(arg_dtb, num_records_bytes=LP * 4)
        dtbs = [fx.Float32(buffer_ops.buffer_load(dtbr, hv * D + 16 * nt + l16, vec_width=1)) for nt in nts]
        hf = w // 2
        c = tx % D
        cwr = buffer_ops.create_buffer_resource_from_addr(arg_cw, num_records_bytes=12 * LP * 4)
        sets = [(hf, True), (fx.Int32(2), hf == 0)] if waves == 4 else [((hf > 2).select(fx.Int32(2), hf), hf < 3)]
        conv = []
        for si, wr in sets:
            dim = si * LP + hv * D + c
            wt = [
                fx.Float32(buffer_ops.buffer_load(cwr, (si * 4 + j) * LP + hv * D + c, vec_width=1))
                for j in range_constexpr(4)
            ]
            raw_x = [
                buffer_ops.buffer_load(pr, (n0 + t) * i32_pstride + dim, vec_width=1, dtype=fx.Int16)
                for t in range_constexpr(max_tok)
            ]
            conv.append((si, wr, dim, wt, raw_x))
        bl = [buffer_ops.buffer_load(pr, (n0 + t) * i32_pstride + COL_BETA + hv, vec_width=1, dtype=fx.Int16)
              for t in range_constexpr(max_tok)]
        # stage 2 (one token per wave): the gate row pair of this lane and its norm weight pair
        nwr = buffer_ops.create_buffer_resource_from_addr(arg_nw, num_records_bytes=D * 4)
        if const_expr(nw_f32):
            nw2 = [fx.Float32(x) for x in loadn(nwr, 2 * lane, 2, fx.Float32)]
        else:
            nwp = loadn(nwr, lane, 1, fx.Int32)[0]
            nw2 = [bf16x2_lo(nwp), bf16x2_hi(nwp)]
        n_s2 = (max_tok + waves - 1) // waves
        ogs = [loadn(pr, ((n0 + w + tj * waves) * i32_pstride + COL_GATE + hv * D) // 2 + lane, 1, fx.Int32)[0]
               for tj in range_constexpr(n_s2)]
        v0 = w * (rg * rpl) + rgi * rpl
        fx.rocdl.sched_barrier(0)

        seed = (n_acc > 1).select(n_acc - 1, fx.Int32(0))
        slot0 = pick(slots, seed)
        cslot = slots[0]
        live = (n_tok > 0).select(slot0, fx.Int32(0)) > 0

        if live:
            hv64 = fx.Int64(hv)
            csr = buffer_ops.create_buffer_resource_from_addr(
                arg_cs + fx.Int64(cslot) * i64_cs_slot_bytes, num_records_bytes=fx.Int64(i64_cs_slot_bytes)
            )
            raw_hs = [
                [buffer_ops.buffer_load(csr, dim * i32_cs_dim + (seed + j) * i32_cs_pos, vec_width=1, dtype=fx.Int16)
                 for j in range_constexpr(3)]
                for _, _, dim, _, _ in conv
            ]
            srcr = buffer_ops.create_buffer_resource_from_addr(
                arg_state + fx.Int64(slot0) * i64_state_slot_bytes + hv64 * (D * D * 4),
                num_records_bytes=D * D * 4,
            )
            soff = [[(v0 + i) * D + 4 * (s + lpr * m) for m in range_constexpr(nvec)] for i in range_constexpr(rpl)]
            st0 = []
            for i in range_constexpr(rpl):
                for m in range_constexpr(nvec):
                    ld = Vec(buffer_ops.buffer_load(srcr, soff[i][m], vec_width=4, dtype=fx.Float32))
                    st0 += [ld[j] for j in range_constexpr(4)]
            fx.rocdl.sched_barrier(0)

            zero4 = Vec.filled(4, 0.0, fx.Float32)
            for j in range_constexpr(nt_n):
                acc4 = zero4
                for kb in range_constexpr(4):
                    acc4 = mfma_bf16(afr[kb], bfr[j][kb], acc4)
                for e in range_constexpr(4):
                    tq = 4 * qd + e
                    gate = exp2(f32_lbc * sigmoid(a_exp * (round_bf16(acc4[e]) + dtbs[j])))
                    if tq < max_tok:
                        put(lds, (tq * 5 + 3) * D + 16 * nts[j] + l16, gate)

            for (si, wr, dim, wt, raw_x), raw_h in zip(conv, raw_hs):
                hist = [bf16_scalar_to_f32(r) for r in raw_h] + [bf16_scalar_to_f32(r) for r in raw_x]
                ys = []
                for t in range_constexpr(max_tok):
                    acc = hist[t] * wt[0] + hist[t + 1] * wt[1]
                    acc = acc + (hist[t + 2] * wt[2] + hist[t + 3] * wt[3])
                    ys.append(acc * sigmoid(acc))
                if wr:
                    for t in range_constexpr(max_tok):
                        put(lds, (t * 5 + si) * D + c, ys[t])
                    for t in range_constexpr(max_tok):
                        buffer_ops.buffer_store(raw_x[t], csr, dim * i32_cs_dim + (2 + t) * i32_cs_pos, mask=t < n_tok)
                    buffer_ops.buffer_store(raw_h[1], csr, dim * i32_cs_dim)
                    buffer_ops.buffer_store(raw_h[2], csr, dim * i32_cs_dim + i32_cs_pos)
            if tx == 0:
                for t in range_constexpr(max_tok):
                    put(scal, 2 * t + 1, sigmoid(bf16_scalar_to_f32(bl[t])))
            fx.gpu.barrier()

            # stage 2: wave w normalizes token w (q, k), and builds the output scale norm_weight * sigmoid(gate)
            for tj in range_constexpr(n_s2):
                tw = w + tj * waves
                if tw < max_tok:
                    base = tw * 5 * D
                    yq = [lds[base + lane + 64 * h] for h in range_constexpr(2)]
                    yk = [lds[base + D + lane + 64 * h] for h in range_constexpr(2)]
                    qq = wave_sum_lane63(yq[0] * yq[0] + yq[1] * yq[1])
                    kk = wave_sum_lane63(yk[0] * yk[0] + yk[1] * yk[1])
                    qk = wave_sum_lane63(yq[0] * yk[0] + yq[1] * yk[1])
                    rl = lambda x: fx.Float32(fx.Int32(fx.rocdl.readlane(
                        T.i32, fx.as_ir_value(x.bitcast(fx.Int32)), fx.as_ir_value(fx.Int32(63)))).bitcast(fx.Float32))
                    qq, kk, qk = rl(qq), rl(kk), rl(qk)
                    rq = rsqrt(qq + EPS_QK) * SCALE
                    rk = rsqrt(kk + EPS_QK)
                    for h in range_constexpr(2):
                        put(lds, base + lane + 64 * h, yq[h] * rq)
                        put(lds, base + D + lane + 64 * h, yk[h] * rk)
                    ogp = ogs[tj]
                    put(lds, base + 4 * D + 2 * lane, nw2[0] * sigmoid(bf16x2_lo(ogp)))
                    put(lds, base + 4 * D + 2 * lane + 1, nw2[1] * sigmoid(bf16x2_hi(ogp)))
                    if lane == 0:
                        put(scal, 2 * tw, qk * (rq * rk))
            fx.gpu.barrier()

            def finish(os_prev, sc_prev, tprev, parts_base):
                """Token ``tprev``'s gated norm and output store (its ss partials are in ``part``)."""
                tsum = part[parts_base]
                for k in range_constexpr(1, waves):
                    tsum = tsum + part[parts_base + k]
                rstd = rsqrt(tsum * fx.Float32(1.0 / D) + f32_eps)
                ow = [os_prev[i] * rstd * sc_prev[i] for i in range_constexpr(rpl)]
                packed = [pk_bf16(ow[2 * k], ow[2 * k + 1]) for k in range_constexpr(rpl // 2)]
                buffer_ops.buffer_store(
                    Vec.from_elements(packed, fx.Int32) if rpl > 2 else packed[0],
                    out_r,
                    ((n0 + tprev) * i32_out_stride + hv * D + v0) // 2,
                    mask=(s == 0) & (tprev >= 0),
                )

            def tok_vec(tt, si, off, n):
                """``n`` floats of token ``tt``'s set ``si`` from column ``off`` (a multiple of n; n 2 or 4 or 8)."""
                base4 = (tt * 5 + si) * (D // 4) + off // 4
                out = []
                if const_expr(n >= 4):
                    for h in range_constexpr(n // 4):
                        v4 = ld4(base4 + h)
                        out += [v4[j] for j in range_constexpr(4)]
                else:
                    for i in range_constexpr(n):
                        out.append(lds[(tt * 5 + si) * D + off + i])
                return out

            init = [fx.as_ir_value(x) for x in st0] + [fx.as_ir_value(fx.Float32(0.0)) for _ in range(2 * rpl)]
            for t, stt in range(fx.Index(0), fx.Index(n_tok), fx.Index(1), init=init):
                t32 = fx.Int32(t)
                S = [[Vec.from_elements([fx.Float32(stt[(i * nvec + m) * 4 + j]) for j in range_constexpr(4)], fx.Float32)
                      for m in range_constexpr(nvec)] for i in range_constexpr(rpl)]
                os_prev = [fx.Float32(stt[rpl * cpl + i]) for i in range_constexpr(rpl)]
                sc_prev = [fx.Float32(stt[rpl * cpl + rpl + i]) for i in range_constexpr(rpl)]
                pbp = ((t32 + 1) % 2) * waves
                b4 = t32 * (5 * D // 4)
                qn = [ld4(b4 + s + lpr * m) for m in range_constexpr(nvec)]
                kn = [ld4(b4 + 32 + s + lpr * m) for m in range_constexpr(nvec)]
                aa = [ld4(b4 + 96 + s + lpr * m) for m in range_constexpr(nvec)]
                vv = tok_vec(t32, 2, v0, rpl)
                scn = tok_vec(t32, 4, v0, rpl)
                qkn = scal[2 * t32]
                bt = scal[2 * t32 + 1]
                new = []
                os = []
                ss = fx.Float32(0.0)
                for i in range_constexpr(rpl):
                    dd = [S[i][m] * aa[m] for m in range_constexpr(nvec)]
                    sk4 = dd[0] * kn[0]
                    sq4 = dd[0] * qn[0]
                    for m in range_constexpr(1, nvec):
                        sk4 = fma(dd[m], kn[m], sk4)
                        sq4 = fma(dd[m], qn[m], sq4)
                    sk = sum_group((sk4[0] + sk4[1]) + (sk4[2] + sk4[3]), lpr)
                    sq = sum_group((sq4[0] + sq4[1]) + (sq4[2] + sq4[3]), lpr)
                    err = (vv[i] - sk) * bt
                    o = sq + err * qkn
                    e4 = Vec.from_elements([err] * 4, fx.Float32)
                    new.append([fma(e4, kn[m], dd[m]) for m in range_constexpr(nvec)])
                    os.append(o)
                    ss = ss + o * o
                slot_t = pick(slots, t32)
                keep = slot_t > 0
                if const_expr(snap):
                    keep = keep & (t32 == snap_pos)
                dstr = buffer_ops.create_buffer_resource_from_addr(
                    arg_state + fx.Int64(slot_t) * i64_state_slot_bytes + hv64 * (D * D * 4),
                    num_records_bytes=keep.select(fx.Int32(D * D * 4), fx.Int32(0)),
                )
                for i in range_constexpr(rpl):
                    for m in range_constexpr(nvec):
                        buffer_ops.buffer_store(new[i][m], dstr, soff[i][m], cache_modifier=store_cm)
                finish(os_prev, sc_prev, t32 - 1, pbp)
                tot = fx.Int32(fx.rocdl.readlane(
                    T.i32, fx.as_ir_value(wave_sum_lane63(ss * fx.Float32(1.0 / lpr)).bitcast(fx.Int32)),
                    fx.as_ir_value(fx.Int32(63)))).bitcast(fx.Float32)
                if lane == 0:
                    put(part, (t32 % 2) * waves + w, tot)
                fx.gpu.barrier()
                results = yield [fx.as_ir_value(x) for i in range_constexpr(rpl) for m in range_constexpr(nvec) for x in [new[i][m][j] for j in range_constexpr(4)]] + [fx.as_ir_value(x) for x in os] + [fx.as_ir_value(x) for x in scn]
            last = fx.Int32(n_tok) - 1
            finish([fx.Float32(results[rpl * cpl + i]) for i in range_constexpr(rpl)],
                   [fx.Float32(results[rpl * cpl + rpl + i]) for i in range_constexpr(rpl)], last, (last % 2) * waves)
        else:
            for t in range(fx.Index(0), fx.Index(n_tok), fx.Index(1)):
                if tx < 16:
                    buffer_ops.buffer_store(
                        Vec.from_elements([fx.Int32(0)] * 4, fx.Int32),
                        out_r,
                        ((n0 + fx.Int32(t)) * i32_out_stride + hv * D) // 2 + 4 * tx,
                    )

    @flyc.jit
    def launch(
        arg_proj: fx.Int64,
        i32_pstride: fx.Int32,
        i32_proj_bytes: fx.Int32,
        arg_fbw: fx.Int64,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i64_cs_slot_bytes: fx.Int64,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_state: fx.Int64,
        i64_state_slot_bytes: fx.Int64,
        arg_cu: fx.Int64,
        arg_idx: fx.Int64,
        i32_idx_stride: fx.Int32,
        arg_acc: fx.Int64,
        arg_snap: fx.Int64,
        arg_nw: fx.Int64,
        arg_out: fx.Int64,
        i32_out_stride: fx.Int32,
        i32_out_bytes: fx.Int32,
        f32_lbc: fx.Float32,
        f32_eps: fx.Float32,
        i32_nseq: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_proj, i32_pstride, i32_proj_bytes, arg_fbw, arg_alog, arg_dtb, arg_cw, arg_cs, i64_cs_slot_bytes,
            i32_cs_dim, i32_cs_pos, arg_state, i64_state_slot_bytes, arg_cu, arg_idx, i32_idx_stride, arg_acc,
            arg_snap, arg_nw, arg_out, i32_out_stride, i32_out_bytes, f32_lbc, f32_eps,
        ).launch(grid=(fx.Int64(i32_nseq) * NH, 1, 1), block=(block, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
