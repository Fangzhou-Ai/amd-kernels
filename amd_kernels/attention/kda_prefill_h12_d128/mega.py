# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA prefill in one launch (gfx950, FlyDSL): the prologue workgroups (conv, gate, intra-chunk matrices -> workspace records), the
walk workgroups (one per (sequence, head, slice group): the serial state recurrence) and the output workgroups (``O`` and the gated
RMSNorm) run side by side; records and state records are handed over through flags (walk.py).

Workgroups are dispatched in block order: the walk workgroups first (so they are resident before anything waits for them), then
the prologue workgroups in chunk order (head fastest), then the output workgroups in the same order. A prologue workgroup never
waits, a walk workgroup waits for records of chunks whose prologue workgroups precede it, an output workgroup for the state
record of its chunk: no wait depends on a workgroup that is dispatched later.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .common import WALK_GROUPS, touch_args
from .outk import OUT_LDS, out_body
from .prologue import LDS_BYTES, prologue_body
from .reference import NH
from .scan import SC1
from .walk import UNIT, WALK_LDS, walk_body

NGRP = WALK_GROUPS
LDS = max(LDS_BYTES, WALK_LDS, OUT_LDS)


@launcher
def compile_mega(w_f32: bool, ckpt: bool, ts: bool = False, knobs: tuple = ()):
    kn = dict(knobs)
    name = "kda_prefill_mega_h12_d128" + ("_wf32" if w_f32 else "") + ("_ck" if ckpt else "") + ("_ts" if ts else "") + "".join(f"_{k}{v}" for k, v in knobs)

    @fx.struct
    class Shared:
        buf: fx.Array[fx.Uint8, LDS, 16]

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(
        arg_mixed: fx.Int64,
        i32_ldx: fx.Int32,
        i32_mixed_bytes: fx.Int32,
        arg_rawg: fx.Int64,
        i32_ldg: fx.Int32,
        arg_beta: fx.Int64,
        i32_ldb: fx.Int32,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i32_cs_slot: fx.Int32,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_tab: fx.Int64,
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        f32_lb: fx.Float32,
        f32_scale: fx.Float32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_sb: fx.Int64,
        i32_npairs: fx.Int32,
        arg_ck: fx.Int64,
        i64_ck_row_bytes: fx.Int64,
        arg_ckoff: fx.Int64,
        arg_ckrow: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        arg_gate: fx.Int64,
        i32_gstride: fx.Int32,
        arg_nw: fx.Int64,
        f32_eps: fx.Float32,
        arg_flags: fx.Int64,
        i32_nwg: fx.Int32,
        i32_lag: fx.Int32,
        arg_ts: fx.Int64,
    ):
        touch_args(arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_sb, i32_npairs, arg_ck,
                   i64_ck_row_bytes, arg_ckoff, arg_ckrow, arg_tab, arg_out, i32_ostride, arg_gate, i32_gstride, arg_nw, f32_eps, arg_flags, i32_nwg, i32_lag, arg_ts)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        base = fx.Int32(fx.ptrtoint(smem.buf.ptr))
        bid = fx.Int32(fx.block_idx.x)
        tx = fx.thread_idx.x
        if const_expr(ts):
            t_start = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])).to(fx.Int32)
            ts_r = buffer_ops.create_buffer_resource_from_addr(arg_ts, num_records_bytes=fx.Int32(0x7FFFFFF0))

        def stamp(role, idx):
            t_end = fx.Int64(_llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])).to(fx.Int32)
            if tx == 0:
                buffer_ops.buffer_store(Vec.from_elements([fx.Int32(role), idx, t_start, t_end], fx.Int32), ts_r, bid * 4)

        if bid < i32_nwg:
            for u in range(bid, ((i32_npairs + 7) // 8) * UNIT, i32_nwg):
                walk_body(base, u, ckpt, arg_ws, i32_nc, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu, arg_choff, arg_sb,
                          i32_npairs, arg_ck, i64_ck_row_bytes, arg_ckoff, arg_ckrow, arg_flags, arg_ts if ts else None, kn)
            if const_expr(ts):
                stamp(0, bid)
        else:
            idp = bid - i32_nwg
            lagc = fx.min(i32_lag, i32_nc)
            lag12 = lagc * NH
            j = idp - lag12
            inter = (idp >= lag12) & (j < (i32_nc - lagc) * (2 * NH))
            k = j // (2 * NH)
            r = j % (2 * NH)
            is_p = (idp < lag12) | (inter & (r < NH))
            p_idx = (idp < lag12).select(idp, lag12 + k * NH + r)
            o_idx = inter.select(k * NH + r - NH, (i32_nc - lagc) * NH + j - (i32_nc - lagc) * (2 * NH))
            if is_p:
                if const_expr(kn.get('pp', 0)):
                    cp = p_idx // NH
                    if cp < 6:
                        fx.rocdl.s_setprio(3)
                    elif cp < 16:
                        fx.rocdl.s_setprio(2)
                    elif cp < 40:
                        fx.rocdl.s_setprio(1)
                prologue_body(base, p_idx // NH, p_idx % NH, bid, arg_mixed, i32_ldx, i32_mixed_bytes, arg_rawg, i32_ldg, arg_beta, i32_ldb,
                              arg_alog, arg_dtb, arg_cw, arg_cs, i32_cs_slot, i32_cs_dim, i32_cs_pos, arg_tab, arg_ws, i32_nc, f32_lb,
                              f32_scale, SC1, arg_flags)
                if const_expr(ts):
                    stamp(1, p_idx)
            else:
                out_body(base, o_idx // NH, o_idx % NH, w_f32, 0, arg_ws, i32_nc, arg_sb, arg_tab, arg_out, i32_ostride, arg_gate,
                         i32_gstride, arg_nw, f32_eps, arg_flags)
                if const_expr(ts):
                    stamp(2, o_idx)

    @flyc.jit
    def launch(
        arg_mixed: fx.Int64,
        i32_ldx: fx.Int32,
        i32_mixed_bytes: fx.Int32,
        arg_rawg: fx.Int64,
        i32_ldg: fx.Int32,
        arg_beta: fx.Int64,
        i32_ldb: fx.Int32,
        arg_alog: fx.Int64,
        arg_dtb: fx.Int64,
        arg_cw: fx.Int64,
        arg_cs: fx.Int64,
        i32_cs_slot: fx.Int32,
        i32_cs_dim: fx.Int32,
        i32_cs_pos: fx.Int32,
        arg_tab: fx.Int64,
        arg_ws: fx.Int64,
        i32_nc: fx.Int32,
        f32_lb: fx.Float32,
        f32_scale: fx.Float32,
        arg_state: fx.Int64,
        i64_slot_bytes: fx.Int64,
        arg_sidx: fx.Int64,
        arg_has: fx.Int64,
        arg_cu: fx.Int64,
        arg_choff: fx.Int64,
        arg_sb: fx.Int64,
        i32_npairs: fx.Int32,
        arg_ck: fx.Int64,
        i64_ck_row_bytes: fx.Int64,
        arg_ckoff: fx.Int64,
        arg_ckrow: fx.Int64,
        arg_out: fx.Int64,
        i32_ostride: fx.Int32,
        arg_gate: fx.Int64,
        i32_gstride: fx.Int32,
        arg_nw: fx.Int64,
        f32_eps: fx.Float32,
        arg_flags: fx.Int64,
        i32_nwg: fx.Int32,
        i32_lag: fx.Int32,
        arg_ts: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(arg_mixed, i32_ldx, i32_mixed_bytes, arg_rawg, i32_ldg, arg_beta, i32_ldb, arg_alog, arg_dtb, arg_cw, arg_cs, i32_cs_slot,
               i32_cs_dim, i32_cs_pos, arg_tab, arg_ws, i32_nc, f32_lb, f32_scale, arg_state, i64_slot_bytes, arg_sidx, arg_has, arg_cu,
               arg_choff, arg_sb, i32_npairs, arg_ck, i64_ck_row_bytes, arg_ckoff, arg_ckrow, arg_out, i32_ostride, arg_gate, i32_gstride,
               arg_nw, f32_eps, arg_flags, i32_nwg, i32_lag, arg_ts).launch(
            grid=(i32_nwg + i32_nc * NH * 2, 1, 1), block=(256, 1, 1), stream=stream)

    return launch
