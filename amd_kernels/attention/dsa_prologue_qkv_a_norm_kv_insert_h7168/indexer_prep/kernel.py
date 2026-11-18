# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""DSA indexer decode prep, FlyDSL (gfx950): the key LayerNorm and its MXFP4 insert into the indexer cache, the query
MXFP4 quantization, the head weights and the top-k buffer reset, one launch.

One workgroup of 4 waves per query row. Waves 0-1: a lane per (head, 32-column group) of the query: its 32 bf16 are
the amax, the power-of-two scale (8 bits, ``reference.py``) and 16 payload bytes, so the group needs no cross-lane
step. Wave 2: the key row, two columns a lane: LayerNorm statistics through wave folds, the 16 lanes of a group fold
the amax, each lane stores its payload byte and each group's first lane its scale byte into the key's page slot
(slot < 0: no store). Wave 3: lane l < 32 scales head weight l, and the other 32 lanes zero the scale entries of the
heads 32..63 (the AITER row-group scale tile holds 64 heads). The workgroups past the query rows each reset one row
of the top-k buffer to -1 (a separate workgroup, so the reset's stores do not sit behind a row's loads).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..common import PRELOAD_HINTS, bf16x8, f32, maxnum, pk_bf16, pow2_exponent, pow2_pair, row_max, wave_sum

D = 128
HEADS = 32
NT = 256
PAGE_BYTES = 4352  # 64 slots x 68 bytes: 4096 payload, 256 scales


def _e2m1x2(a, b, old, sel):
    """E2M1 codes of a, b (v_cvt_scalef32_pk_fp4_f32, scale 1) into byte ``sel`` of ``old``, a the low nibble."""
    return fx.Int32(
        _rocdl_ops.cvt_scalef32_pk_fp4_f32(
            T.i32,
            fx.as_ir_value(fx.Int32(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            sel,
        )
    )


def _unsign_zero(w):
    """Codes 0b1000 (-0) of the packed nibbles of ``w`` -> 0, as the reference keeps no sign on a zero code."""
    low = w & 0x77777777
    nonzero = (low + 0x77777777) & 0x88888888
    return w & (low | nonzero)


def _group_scale(amax):
    """(biased exponent, 2^(127 - e)) of the group scale from its amax."""
    s = maxnum(amax / fx.Float32(6.0), f32(0x38D1B717))  # 1e-4
    e = pow2_exponent(s)
    return e, pow2_pair(e)[1]


def _ld16(rsrc, off):
    v = buffer_ops.buffer_load(rsrc, off, vec_width=1, dtype=T.i16)
    return fx.Int16(v).to(fx.Int32)


def compile_indexer_prep(topk: int):
    assert topk % (4 * NT) == 0
    name = f"dsa_indexer_prep_h{HEADS}_d{D}_top{topk}"

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        i32_nq: fx.Int32,
        i32_nout: fx.Int32,
        i32_nsl: fx.Int32,
        i32_qst: fx.Int32,
        i32_kst: fx.Int32,
        i32_wst: fx.Int32,
        i32_ost: fx.Int32,
        i32_eps: fx.Int32,
        i32_wscale: fx.Int32,
        i64_page_b: fx.Int64,
        arg_q: fx.Int64,  # [nq, 32, 128] bf16, row stride i32_qst
        arg_k: fx.Int64,  # [nsl, 128] bf16, row stride i32_kst
        arg_w: fx.Int64,  # [nq, 32] bf16, row stride i32_wst
        arg_lw: fx.Int64,  # LayerNorm weight [128] bf16
        arg_lb: fx.Int64,  # LayerNorm bias [128] bf16
        arg_slot: fx.Int64,  # [nsl] i64, < 0 skips
        arg_cache: fx.Int64,  # indexer pages
        arg_qp: fx.Int64,  # q payload [nq, 32, 64] u8
        arg_qs: fx.Int64,  # q scales [nq, 1, 4, 16, 4] u8
        arg_wo: fx.Int64,  # weights [nq, 32] fp32
        arg_out: fx.Int64,  # top-k buffer [nout, topk] i32, row stride i32_ost
    ):
        tx = fx.thread_idx.x
        row = fx.block_idx.x
        lane = tx % 64
        if row < i32_nq:
            if tx < 128:
                head = tx // 4
                g = tx % 4
                qr = buffer_ops.create_buffer_resource_from_addr(
                    arg_q + fx.Int64(row) * fx.Int64(i32_qst * 2), num_records_bytes=HEADS * D * 2
                )
                base = head * (D // 2) + g * 16
                x = []
                for i in range_constexpr(4):
                    x += bf16x8(Vec(buffer_ops.buffer_load(qr, base + i * 4, vec_width=4, dtype=fx.Int32)))
                amax = fx.math.absf(x[0])
                for i in range_constexpr(1, 32):
                    amax = maxnum(amax, fx.math.absf(x[i]))
                e, rescale = _group_scale(amax)
                words = []
                for i in range_constexpr(4):
                    d = fx.Int32(0)
                    for j in range_constexpr(4):
                        d = _e2m1x2(x[8 * i + 2 * j] * rescale, x[8 * i + 2 * j + 1] * rescale, d, j)
                    words.append(_unsign_zero(d))
                qp = buffer_ops.create_buffer_resource_from_addr(
                    arg_qp, num_records_bytes=fx.Int64(i32_nq) * (HEADS * 64)
                )
                buffer_ops.buffer_store(
                    Vec.from_elements(words, fx.Int32), qp, (row * HEADS + head) * 16 + g * 4
                )
                qs = buffer_ops.create_buffer_resource_from_addr(
                    arg_qs, num_records_bytes=fx.Int64(i32_nq) * 256
                )
                buffer_ops.buffer_store(
                    e.to(fx.Int8),
                    qs,
                    row * 256 + g * 64 + (head % 16) * 4 + head // 16,
                    offset_is_bytes=True,
                )
            if tx >= 192:
                if lane < 32:
                    wr = buffer_ops.create_buffer_resource_from_addr(
                        arg_w + fx.Int64(row) * fx.Int64(i32_wst * 2), num_records_bytes=HEADS * 2
                    )
                    wv = (_ld16(wr, lane) << 16).bitcast(fx.Float32)
                    wo = buffer_ops.create_buffer_resource_from_addr(
                        arg_wo, num_records_bytes=fx.Int64(i32_nq) * (HEADS * 4)
                    )
                    buffer_ops.buffer_store(wv * i32_wscale.bitcast(fx.Float32), wo, row * HEADS + lane)
                if lane >= 32:
                    qs = buffer_ops.create_buffer_resource_from_addr(
                        arg_qs, num_records_bytes=fx.Int64(i32_nq) * 256
                    )
                    for i in range_constexpr(2):
                        s = (lane - 32) * 2 + i
                        buffer_ops.buffer_store(
                            fx.Int16(0),
                            qs,
                            row * 256 + (s // 16) * 64 + (s % 16) * 4 + 2,
                            offset_is_bytes=True,
                        )
            if tx >= 128:
                if tx < 192:
                    sl_r = buffer_ops.create_buffer_resource_from_addr(
                        arg_slot, num_records_bytes=fx.Int64(i32_nsl) * 8
                    )
                    slot = fx.Int32(
                        buffer_ops.buffer_load(sl_r, row * 2, vec_width=1, dtype=fx.Int32, is_scalar=True)
                    )
                    kr = buffer_ops.create_buffer_resource_from_addr(
                        arg_k + fx.Int64(row) * fx.Int64(i32_kst * 2), num_records_bytes=D * 2
                    )
                    lw = buffer_ops.create_buffer_resource_from_addr(arg_lw, num_records_bytes=D * 2)
                    lb = buffer_ops.create_buffer_resource_from_addr(arg_lb, num_records_bytes=D * 2)
                    xw = fx.Int32(buffer_ops.buffer_load(kr, lane, vec_width=1, dtype=fx.Int32))
                    ww = fx.Int32(buffer_ops.buffer_load(lw, lane, vec_width=1, dtype=fx.Int32))
                    bw = fx.Int32(buffer_ops.buffer_load(lb, lane, vec_width=1, dtype=fx.Int32))
                    x0, x1 = (xw << 16).bitcast(fx.Float32), (xw & -65536).bitcast(fx.Float32)
                    mean = wave_sum(x0 + x1) * fx.Float32(1.0 / D)
                    c0, c1 = x0 - mean, x1 - mean
                    var = wave_sum(fx.math.fma(c0, c0, c1 * c1)) * fx.Float32(1.0 / D)
                    rstd = fx.Float32(_rocdl_ops.rsq(T.f32, fx.as_ir_value(var + i32_eps.bitcast(fx.Float32))))
                    w0, w1 = (ww << 16).bitcast(fx.Float32), (ww & -65536).bitcast(fx.Float32)
                    b0, b1 = (bw << 16).bitcast(fx.Float32), (bw & -65536).bitcast(fx.Float32)
                    yw = pk_bf16(fx.math.fma(w0, rstd * c0, b0), fx.math.fma(w1, rstd * c1, b1))
                    y0, y1 = (yw << 16).bitcast(fx.Float32), (yw & -65536).bitcast(fx.Float32)
                    amax = row_max(maxnum(fx.math.absf(y0), fx.math.absf(y1)))
                    e, rescale = _group_scale(amax)
                    byte = _unsign_zero(_e2m1x2(y0 * rescale, y1 * rescale, fx.Int32(0), 0) & 0xFF)
                    ok = (slot >= 0) & (row < i32_nsl)
                    off = slot % 64
                    cr = buffer_ops.create_buffer_resource_from_addr(
                        arg_cache + fx.Int64(slot // 64) * i64_page_b, num_records_bytes=PAGE_BYTES
                    )
                    grp = lane // 16
                    buffer_ops.buffer_store(
                        byte.to(fx.Int8), cr, grp * 1024 + off * 16 + lane % 16, mask=ok, offset_is_bytes=True
                    )
                    buffer_ops.buffer_store(
                        e.to(fx.Int8),
                        cr,
                        4096 + grp * 64 + (off % 16) * 4 + off // 16,
                        mask=ok & ((lane % 16) == 0),
                        offset_is_bytes=True,
                    )
        r = row - i32_nq
        if (r >= 0) & (r < i32_nout):
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out + fx.Int64(r) * fx.Int64(i32_ost * 4), num_records_bytes=topk * 4
            )
            neg = Vec.from_elements([fx.Int32(-1)] * 4, fx.Int32)
            for i in range_constexpr(topk // (4 * NT)):
                buffer_ops.buffer_store(neg, outr, (i * NT + tx) * 4)

    @flyc.jit
    def launch(
        arg_q: fx.Int64,
        arg_k: fx.Int64,
        arg_w: fx.Int64,
        arg_lw: fx.Int64,
        arg_lb: fx.Int64,
        arg_slot: fx.Int64,
        arg_cache: fx.Int64,
        arg_qp: fx.Int64,
        arg_qs: fx.Int64,
        arg_wo: fx.Int64,
        arg_out: fx.Int64,
        i32_nq: fx.Int32,
        i32_nout: fx.Int32,
        i32_nsl: fx.Int32,
        i32_qst: fx.Int32,
        i32_kst: fx.Int32,
        i32_wst: fx.Int32,
        i32_ost: fx.Int32,
        i32_eps: fx.Int32,
        i32_wscale: fx.Int32,
        i64_page_b: fx.Int64,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_nq, i32_nout, i32_nsl, i32_qst, i32_kst, i32_wst, i32_ost, i32_eps, i32_wscale, i64_page_b,
            arg_q, arg_k, arg_w, arg_lw, arg_lb, arg_slot, arg_cache, arg_qp, arg_qs, arg_wo, arg_out,
        ).launch(grid=(i32_grid, 1, 1), block=(NT, 1, 1), stream=stream)

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
