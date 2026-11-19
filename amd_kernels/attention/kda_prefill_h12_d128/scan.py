# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunk scan (gfx950, FlyDSL): per (sequence, head, value group) the chunks of the sequence in order, the [K, V] state in
MFMA accumulators, one wave per 16 value columns.

Per chunk, with S^T [K, V16] in registers (``Vn = u - w S^T; O = qg S^T + Aqk Vn; S^T = diag(decay) S^T + kg^T Vn``):
every MFMA takes a workspace fragment (reference.py, fragment-major: one fragment is one 1 KB linear read) as one operand and
a register-resident operand as the other. ``Vn`` leaves the first GEMM as accumulators (lane (v, q) holds tokens 4 q + e of tile mt) and
is the B operand of the state update and the A operand of the Aqk product with no data movement; the state accumulators (lane
(v, q) holds k = 16 kt + 4 q + e) feed the next chunk's GEMM the same way, the reduction order being permuted identically on
both operands (columns 32 ks + 16 (s // 4) + 4 q + s % 4 at slot s). O is computed transposed (O^T = S qg^T), so a lane
holds four consecutive value columns of one token and stores 8 bytes.

Data path: a chunk's record (w | qg | kgt | aqk | decay, 58 KB) moves HBM / L2 -> LDS with asynchronous ``buffer_load ... lds``
(linear 1 KB pieces split over the waves) into a two-stage ring: before chunk c every wave waits for its own DMAs and the
workgroup barriers, which also proves the other stage free, then the DMAs of chunk c + 1 are issued. The ``u`` slices of a wave
are loaded straight into registers one chunk ahead.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir import ir
from flydsl._mlir.dialects import rocdl as _rocdl_ods
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import barrier, bf16x2_hi, bf16x2_lo, lds_load4, mfma, sched_barrier
from .reference import (BT, D, NH, RECORD_AQK, RECORD_BYTES, RECORD_DEC, RECORD_KGT, RECORD_QG, RECORD_U,
                        RECORD_W)

STAGE = 60 * 1024  # 58 KB of tiles + the DMA padding of the last wave pieces
STATE_BYTES = D * D * 4
NDMA_BLOCKS = RECORD_U // 1024  # 57 1 KB pieces of w | qg | kgt | aqk | decay


@launcher
def compile_scan(waves: int, nvt: int):
    vcta = 16 * waves * nvt
    assert D % vcta == 0
    ngrp = D // vcta
    nthreads = 64 * waves
    ndma = -(-NDMA_BLOCKS // waves)
    name = f"kda_prefill_scan_h12_d128_w{waves}_v{nvt}"

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, 2 * STAGE, 16]

    @flyc.kernel(name=name, known_block_size=[nthreads, 1, 1])
    def kernel(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        i64_out_bytes: fx.Int64,
        i32_npairs: fx.Int32,
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        l16 = lane % 16
        qg = lane // 16
        pair = (bid // 8 // ngrp) * 8 + bid % 8
        vg = bid // 8 % ngrp
        if pair < i32_npairs:
            n = pair // NH
            h = pair % NH
            cu_r = buffer_ops.create_buffer_resource_from_addr(arg_cu)
            tok0 = fx.Int32(buffer_ops.buffer_load(cu_r, n, vec_width=1, is_scalar=True))
            ntok = fx.Int32(buffer_ops.buffer_load(cu_r, n + 1, vec_width=1, is_scalar=True)) - tok0
            nchunks = (ntok + (BT - 1)) // BT
            choff_r = buffer_ops.create_buffer_resource_from_addr(arg_choff)
            c0 = fx.Int32(buffer_ops.buffer_load(choff_r, n * 2, vec_width=1, is_scalar=True))
            slot = fx.Int32(buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource_from_addr(arg_sidx), n, vec_width=1, is_scalar=True))
            has = fx.Int32(buffer_ops.buffer_load(
                buffer_ops.create_buffer_resource_from_addr(arg_has), n // 4, vec_width=1, is_scalar=True))
            has = ((has >> ((n % 4) * 8)) & 0xFF) != 0
            if nchunks > 0:
                lds_base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
                lane16 = lane * 16
                vs0 = (vg * waves + wave) * nvt
                st_res = buffer_ops.create_buffer_resource_from_addr(
                    arg_state + fx.Int64(slot) * i64_slot_bytes + fx.Int64(h) * STATE_BYTES, num_records_bytes=STATE_BYTES)
                s0 = []
                for vt in range_constexpr(nvt):
                    for kt in range_constexpr(8):
                        s0.append(buffer_ops.buffer_load(
                            st_res, (16 * (vs0 + vt) + l16) * D + 16 * kt + 4 * qg, vec_width=4, dtype=fx.Float32,
                            mask=has))
                out_r = buffer_ops.create_buffer_resource_from_addr(arg_out, num_records_bytes=i64_out_bytes)

                def rec_addr(c):
                    cc = fx.min(c, nchunks - 1)
                    return arg_ws + fx.Int64(h * i32_nc + c0 + cc) * RECORD_BYTES

                def dma(c, stage):
                    rs = buffer_ops.create_buffer_resource_from_addr(rec_addr(c), num_records_bytes=RECORD_BYTES)
                    for i in range_constexpr(ndma):
                        j = wave + i * waves
                        dst = fx.Int32(fx.rocdl.readfirstlane(T.i32, lds_base + stage * STAGE + j * 1024))
                        lp = fx.to_llvm_ptr(fx.inttoptr(fx.PointerType.get(T.i8, fx.AddressSpace.Shared), fx.Int64(dst)))
                        _rocdl_ods.raw_ptr_buffer_load_async_lds(
                            rs, lp, fx.Int32(16).ir_value(), fx.Int32(j * 1024 + lane16).ir_value(),
                            fx.Int32(0).ir_value(), fx.Int32(0).ir_value(),
                            aux=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 0))

                def load_u(c):
                    out = []
                    for vt in range_constexpr(nvt):
                        rs = buffer_ops.create_buffer_resource_from_addr(
                            rec_addr(c) + (RECORD_U + 2048 * (vs0 + vt)), num_records_bytes=2048)
                        for mt in range_constexpr(4):
                            out.append(buffer_ops.buffer_load(rs, (mt * 64 + lane) * 2, vec_width=2, dtype=fx.Int32))
                    return out

                dma(fx.Int32(0), fx.Int32(0))
                u0 = load_u(fx.Int32(0))
                init = [fx.as_ir_value(x) for x in s0 + u0]
                for c, st in range(0, fx.Int64(nchunks), 1, init=init):
                    ci = fx.Int32(c)
                    S = [[Vec(st[vt * 8 + kt]) for kt in range_constexpr(8)] for vt in range_constexpr(nvt)]
                    U = [[Vec(st[nvt * 8 + vt * 4 + mt]) for mt in range_constexpr(4)] for vt in range_constexpr(nvt)]
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                    barrier()
                    base = lds_base + (ci % 2) * STAGE + lane16
                    dma(ci + 1, (ci + 1) % 2)
                    u_next = load_u(ci + 1)
                    sched_barrier()

                    vn = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)] for _ in range_constexpr(nvt)]
                    ob = [[Vec.filled(4, 0.0, fx.Float32) for _ in range_constexpr(4)] for _ in range_constexpr(nvt)]
                    for ks in range_constexpr(4):
                        sf = []
                        for vt in range_constexpr(nvt):
                            lo, hi = S[vt][2 * ks], S[vt][2 * ks + 1]
                            sf.append(Vec.from_elements([lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]],
                                                        fx.Float32).to(fx.BFloat16).bitcast(fx.Int32))
                        for mt in range_constexpr(4):
                            wf = lds_load4(base + RECORD_W + (mt * 4 + ks) * 1024)
                            for vt in range_constexpr(nvt):
                                vn[vt][mt] = mfma(wf, sf[vt], vn[vt][mt])
                        for nt in range_constexpr(4):
                            qf = lds_load4(base + RECORD_QG + (nt * 4 + ks) * 1024)
                            for vt in range_constexpr(nvt):
                                ob[vt][nt] = mfma(sf[vt], qf, ob[vt][nt])
                    vnf = []
                    for vt in range_constexpr(nvt):
                        row = []
                        vv = []
                        for mt in range_constexpr(4):
                            u2 = U[vt][mt]
                            ulo, uhi = fx.Int32(u2[0]), fx.Int32(u2[1])
                            uf = [bf16x2_lo(ulo), bf16x2_hi(ulo), bf16x2_lo(uhi), bf16x2_hi(uhi)]
                            vv.append([vn[vt][mt][e] + uf[e] for e in range_constexpr(4)])
                        for ksp in range_constexpr(2):
                            a, b = vv[2 * ksp], vv[2 * ksp + 1]
                            row.append(Vec.from_elements(a + b, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32))
                        vnf.append(row)
                    for ksp in range_constexpr(2):
                        for nt in range_constexpr(2 * ksp, 4):
                            af = lds_load4(base + RECORD_AQK + (nt * 2 + ksp) * 1024)
                            for vt in range_constexpr(nvt):
                                ob[vt][nt] = mfma(vnf[vt][ksp], af, ob[vt][nt])
                    rows = fx.min(ntok - ci * BT, fx.Int32(BT))
                    for vt in range_constexpr(nvt):
                        for nt in range_constexpr(4):
                            t_loc = nt * 16 + l16
                            o = ob[vt][nt]
                            buffer_ops.buffer_store(
                                o.to(fx.BFloat16).bitcast(fx.Int32), out_r,
                                ((tok0 + ci * BT + t_loc) * i32_ostride + h * D + 16 * (vs0 + vt) + 4 * qg) // 2,
                                mask=t_loc < rows)
                    s_new = [[None] * 8 for _ in range_constexpr(nvt)]
                    for kt in range_constexpr(8):
                        d4 = lds_load4(base - lane16 + RECORD_DEC + kt * 64 + qg * 16).bitcast(fx.Float32)
                        for vt in range_constexpr(nvt):
                            s_new[vt][kt] = S[vt][kt] * d4
                    for ksp in range_constexpr(2):
                        for kt in range_constexpr(8):
                            kf = lds_load4(base + RECORD_KGT + (kt * 2 + ksp) * 1024)
                            for vt in range_constexpr(nvt):
                                s_new[vt][kt] = mfma(kf, vnf[vt][ksp], s_new[vt][kt])
                    results = yield [fx.as_ir_value(s_new[vt][kt]) for vt in range_constexpr(nvt) for kt in range_constexpr(8)] + [
                        fx.as_ir_value(x) for x in u_next]
                fx.rocdl.s_waitcnt(vmcnt=0)
                for vt in range_constexpr(nvt):
                    for kt in range_constexpr(8):
                        buffer_ops.buffer_store(
                            Vec(results[vt * 8 + kt]), st_res, (16 * (vs0 + vt) + l16) * D + 16 * kt + 4 * qg)

    @flyc.jit
    def launch(
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        i64_out_bytes: fx.Int64,
        i32_npairs: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_out, i32_ostride,
               i64_out_bytes, i32_npairs).launch(grid=(((i32_npairs + 7) // 8) * 8 * ngrp, 1, 1), block=(nthreads, 1, 1),
                                               stream=stream)

    return launch
