# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSA decode prologue, FlyDSL (gfx950): the q_a and kv_a RMS norms of the qkv_a projection row and the fp8_ds_mla
KV-cache insert, one launch.

One workgroup of 4 waves per token. A row is [q 1536 | kv 512 | k_pe 64] bf16 and the first 2048 columns map one to
one onto 256 lanes of 8 columns: waves 0-2 hold q, wave 3 holds kv. The squares are summed in fp32 (a DPP fold per
wave, the three q waves added through LDS); ``y = RN_bf16(x * rsqrt(sum / n + eps) * w)``, one rounding as AITER's
norm does. Wave 3 then takes the max of ``|y|`` over each 16-lane group (one 128-column tile), the tile scale is the
smallest power of two >= max(amax / 448, 1e-4), and its 8 columns go into the cache record as e4m3 of ``y / scale``:
[512 e4m3 | 4 fp32 scales | 64 bf16 k_pe] = 656 bytes at ``slot``. A slot < 0 (a token this DCP rank does not
own, or padding) writes no cache byte; the norm outputs are written for every row.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..common import PRELOAD_HINTS, bf16x8, f32, maxnum, pk_bf16, pow2_exponent, pow2_pair, row_max, wave_sum

Q = 1536
KV = 512
PE = 64
ROW = Q + KV + PE
RECORD = 656
NT = 256


def _cvt_pk_fp8(old, a, b, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32 (scale 1): a, b -> two e4m3 bytes in the low / high half of ``old``."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def compile_qkv_a_norm_kv_insert(insert: bool):
    name = "dsa_qkv_a_norm_kv_insert_h7168" + ("" if insert else "_noinsert")

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, 4]

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_n: fx.Int32,
        i32_nsl: fx.Int32,
        i32_row_st: fx.Int32,
        i32_tok_b: fx.Int32,
        i32_eps: fx.Int32,
        i64_page_b: fx.Int64,
        arg_x: fx.Int64,  # [n, row stride] bf16, rows of ROW
        arg_qw: fx.Int64,  # [1536] bf16
        arg_kw: fx.Int64,  # [512] bf16
        arg_q: fx.Int64,  # q_out [n, 1536] bf16
        arg_kv: fx.Int64,  # kv_out [n, 512] bf16
        arg_cache: fx.Int64,  # fp8_ds_mla pages, 64-token blocks of 656-byte records
        arg_slot: fx.Int64,  # slots [nsl] i64, < 0 skips
    ):
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        is_q = tx < (Q // 8)
        smem = fx.SharedAllocator().allocate(Shared).peek()
        part = smem.part.ptr
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x + fx.Int64(row) * fx.Int64(i32_row_st * 2), num_records_bytes=ROW * 2
        )
        wr = buffer_ops.create_buffer_resource_from_addr(
            is_q.select(arg_qw, arg_kw), num_records_bytes=Q * 2
        )
        wcol = is_q.select(tx, tx - (Q // 8)) * 4
        if const_expr(insert):
            sl_r = buffer_ops.create_buffer_resource_from_addr(
                arg_slot, num_records_bytes=fx.Int64(i32_nsl) * 8
            )
            slot = fx.Int32(
                buffer_ops.buffer_load(sl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True)
            )
        xv = bf16x8(Vec(buffer_ops.buffer_load(xr, tx * 4, vec_width=4, dtype=fx.Int32)))
        wv = bf16x8(Vec(buffer_ops.buffer_load(wr, wcol, vec_width=4, dtype=fx.Int32)))
        sq = fx.math.fma(xv[0], xv[0], xv[1] * xv[1])
        for i in range_constexpr(1, 4):
            sq = sq + fx.math.fma(xv[2 * i], xv[2 * i], xv[2 * i + 1] * xv[2 * i + 1])
        tot = wave_sum(sq)
        if lane == 0:
            part[wave] = tot
        fx.gpu.barrier()
        p = [part[i] for i in range_constexpr(4)]
        ss = is_q.select((p[0] + p[1]) + p[2], p[3])
        inv_n = is_q.select(fx.Float32(1.0 / Q), fx.Float32(1.0 / KV))
        eps = i32_eps.bitcast(fx.Float32)
        rr = fx.Float32(fx.rocdl.rsq(T.f32, fx.as_ir_value(ss * inv_n + eps)))
        y = [(xv[i] * rr) * wv[i] for i in range_constexpr(8)]
        packed = [pk_bf16(y[2 * i], y[2 * i + 1]) for i in range_constexpr(4)]
        out = buffer_ops.create_buffer_resource_from_addr(
            is_q.select(arg_q + fx.Int64(row) * (Q * 2), arg_kv + fx.Int64(row) * (KV * 2)),
            num_records_bytes=Q * 2,
        )
        buffer_ops.buffer_store(
            Vec.from_elements(packed, fx.Int32), out, is_q.select(tx, tx - (Q // 8)) * 4
        )
        if const_expr(insert):
            yb = bf16x8(Vec.from_elements(packed, fx.Int32))
            amax = fx.math.absf(yb[0])
            for i in range_constexpr(1, 8):
                amax = maxnum(amax, fx.math.absf(yb[i]))
            amax = row_max(amax)
            s = maxnum(amax / fx.Float32(448.0), f32(0x38D1B717))  # 1e-4
            scale, rescale = pow2_pair(pow2_exponent(s))
            q8 = [v * rescale for v in yb]
            lo = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q8[0], q8[1], False), q8[2], q8[3], True)
            hi = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), q8[4], q8[5], False), q8[6], q8[7], True)
            kl = tx - (Q // 8)  # 0..63 in wave 3
            ok = (slot >= 0) & (row < i32_nsl) & (kl >= 0)
            rec = (
                arg_cache
                + fx.Int64(slot // 64) * i64_page_b
                + fx.Int64((slot % 64) * i32_tok_b)
            )
            cr = buffer_ops.create_buffer_resource_from_addr(rec, num_records_bytes=RECORD)
            buffer_ops.buffer_store(Vec.from_elements([lo, hi], fx.Int32), cr, kl * 2, mask=ok)
            buffer_ops.buffer_store(scale, cr, KV // 4 + kl // 16, mask=ok & ((kl % 16) == 0))
            pe_ok = ok & (kl < 8)
            pe = buffer_ops.buffer_load(xr, (Q + KV) // 2 + kl * 4, vec_width=4, dtype=fx.Int32, mask=pe_ok)
            buffer_ops.buffer_store(pe, cr, (KV + 16) // 4 + kl * 4, mask=pe_ok)

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_cache: fx.Int64,
        arg_slot: fx.Int64,
        i32_n: fx.Int32,
        i32_nsl: fx.Int32,
        i32_row_st: fx.Int32,
        i32_tok_b: fx.Int32,
        i32_eps: fx.Int32,
        i64_page_b: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            i32_n, i32_nsl, i32_row_st, i32_tok_b, i32_eps, i64_page_b,
            arg_x, arg_qw, arg_kw, arg_q, arg_kv, arg_cache, arg_slot,
        ).launch(grid=(i32_n, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
