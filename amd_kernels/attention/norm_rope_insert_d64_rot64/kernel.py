# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Per-head RMS norm + NeoX RoPE (all 64 dims) + K / V cache insert of 64-dim heads, FlyDSL (gfx950), one launch.

A token row is a run of 64-dim heads: ``[q (nq) | k (nkv) | v (nkv)]`` (one layer's qkv projection) or
``nl x [k (nkv) | v (nkv)]`` (the context K / V projection of every layer). Eight lanes take one head, lane j of the
group holding dims 8j..8j+7 (one 16-byte load); a wave holds 8 heads of any tokens and roles, so every lane runs the
same code and the role only picks the store.

Norm: the sum of squares in the order of the op it replaces (DPP butterflies; every lane of the group gets the same
bits), ``r = v_rsq_f32(sum / 64 + eps)`` and ``n = (x * r) * w`` in fp32. ``round_norm`` (vLLM's eager ``rms_norm`` +
``rotary_embedding``): lane j adds its 8 squares in order, the lanes add ((t0 + t1) + (t2 + t3)) + ((t4 + t5) +
(t6 + t7)), and ``n`` is rounded to bf16 before RoPE. Otherwise (the Inductor-compiled norm + RoPE of a draft layer,
its persistent-reduction layout of 8 lanes x 8 values): lane j adds ((s0 + s2) + fma(x4, x4, s6)) + ((s1 + s3) +
fma(x5, x5, s7)), the lanes add by xor 4, 2, 1, and RoPE takes the fp32 ``n`` (Inductor elides that rounding). RoPE: dim d < 32 pairs with d + 32 in lane j ^ 4 (DPP row_shl / row_shr 4);
``o = fma(n, cos, -/+ (n' * sin))``, rounded to bf16 once. V is stored as given.

Insert at ``slot`` (< 0, or a token at or past the slots given: skipped) into ``[pages, nkv, page, 128]`` (K 64 |
V 64): bf16 as is, or e4m3 of ``value / scale`` (the bf16 K or V; IEEE fp32 divide, ``v_cvt_scalef32_pk_fp8_f32``
under MODE.fp16_ovfl, as ``triton_reshape_and_cache_flash``). The cache stores use 64-bit addresses (a layer's
cache can be a view of a pool larger than 4 GB).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from amd_kernels.common import buffer_ops
from amd_kernels.warm_up import launcher

NW = 4  # waves per workgroup
D = 64
ROT = 64
HALF = ROT // 2
GL = 8  # lanes per head
HPB = NW * 64 // GL  # heads per workgroup
HWREG_MODE_FP16_OVFL = 1 | (23 << 6)  # hwreg(HW_REG_MODE, 23, 1)
PRELOAD_HINTS = {
    "llvm_options": {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 16,
    }
}
CACHES = (None, "bf16", "fp8")


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _bf16x8(v):
    """8 fp32 of 4 dwords of bf16 pairs (low half first)."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pk_bf16(a, b):
    """Two fp32 -> two RNE bf16 in a dword (v_cvt_pk_bf16_f32), ``a`` low."""
    return fx.Int32(fx.rocdl.cvt_pk_bf16_f32(fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))))


def _dpp(src, ctrl, old=None, bank_mask=0xF):
    o = _llvm.mlir_undef(T.i32) if old is None else fx.as_ir_value(old)
    full = old is None and bank_mask == 0xF
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, o, fx.as_ir_value(fx.Int32(src)), ctrl, 0xF, bank_mask, full))


def _xor_lane(v, m):
    """Lane ``l ^ m`` (m = 1, 2, 4) of an i32."""
    if m == 1:
        return _dpp(v, 0xB1)  # quad_perm [1, 0, 3, 2]
    if m == 2:
        return _dpp(v, 0x4E)  # quad_perm [2, 3, 0, 1]
    assert m == 4
    return _dpp(v, 0x104, old=_dpp(v, 0x114, bank_mask=0xA), bank_mask=0x5)  # row_shl 4 | row_shr 4


def _xor_f(x, m):
    return _xor_lane(_bits(x), m).bitcast(fx.Float32)


def _sum_sq(x, eager: bool):
    """Sum of squares of a head's 64 values (8 a lane), every lane of the group the same bits."""
    sq = [v * v for v in x]
    if eager:
        a = sq[0]
        for e in range(1, 8):
            a = a + sq[e]
        lanes = (1, 2, 4)
    else:
        a = ((sq[0] + sq[2]) + fx.math.fma(x[4], x[4], sq[6])) + ((sq[1] + sq[3]) + fx.math.fma(x[5], x[5], sq[7]))
        lanes = (4, 2, 1)
    for m in lanes:
        a = a + _xor_f(a, m)
    return a


def _set_fp16_ovfl():
    _llvm.call_intrinsic(
        None, "llvm.amdgcn.s.setreg",
        [fx.Int32(HWREG_MODE_FP16_OVFL).ir_value(), fx.Int32(1).ir_value()], [], [],
    )  # fmt: skip


def _cvt_pk_fp8(old, a, b, hi: bool):
    """v_cvt_scalef32_pk_fp8_f32 (scale 1): a, b -> two e4m3 bytes in the low / high half of ``old``."""
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(fx.Int32(old))),
            fx.as_ir_value(fx.Float32(a)),
            fx.as_ir_value(fx.Float32(b)),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _e4m3x8(v):
    """8 fp32 -> 8 e4m3 bytes in 2 dwords."""
    w0 = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), v[0], v[1], False), v[2], v[3], True)
    w1 = _cvt_pk_fp8(_cvt_pk_fp8(fx.Int32(0), v[4], v[5], False), v[6], v[7], True)
    return [w0, w1]


def _gptr(addr, ty, align):
    return fx.inttoptr(fx.PointerType.get(ty, address_space=fx.AddressSpace.Global, alignment=align), fx.Int64(addr))


def _gstore(addr, words):
    nb = 4 * len(words)
    _llvm.StoreOp(
        fx.as_ir_value(Vec.from_elements(words, fx.Int32)), _gptr(addr, T.i32, nb).llvm_ptr, alignment=nb
    )


@launcher
def compile_norm_rope_insert(nq: int, nkv: int, nl: int, block_size: int, cache, round_norm: bool):
    """``nq`` q heads then ``nl`` layers of ``nkv`` K + ``nkv`` V heads a row (``nl == 1`` or ``nq == 0``).
    ``cache``: None (q only, no K / V heads), "bf16" or "fp8"; ``nl > 1`` reads the caches and scales from a
    pointer table ``[nl caches | nl k scales | nl v scales]``."""
    assert cache in CACHES and (nl == 1 or nq == 0) and (nq > 0 or cache is not None)
    insert = cache is not None
    S = nq + (nl * 2 * nkv if insert else 0)  # heads a row
    BS = block_size
    ELT = 2 if cache == "bf16" else 1
    TABLE = nl > 1
    name = f"norm_rope_insert_d64_rot64_q{nq}_kv{nkv}_l{nl}_p{BS}_{cache or 'noinsert'}" + (
        "" if round_norm else "_f32rope"
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        i32_n: fx.Int32,  # rows
        i32_row_st: fx.Int32,  # input row stride (elements, even)
        arg_x: fx.Int64,  # [n, row stride] bf16
        arg_pos: fx.Int64,  # [n] i64
        arg_cs: fx.Int64,  # cos_sin_cache [i32_cs_rows, 64] bf16 (cos 32 | sin 32)
        arg_qw: fx.Int64,  # q norm weight [64] bf16
        arg_kw: fx.Int64,  # K norm weights [nl, 64] bf16
        arg_q: fx.Int64,  # q_out [n, nq * 64] bf16
        arg_kv: fx.Int64,  # nl == 1: the cache; else the pointer table (i64)
        arg_ks: fx.Int64,  # nl == 1: fp32 scalars (fp8)
        arg_vs: fx.Int64,
        arg_slot: fx.Int64,  # i64 slots; layer l at + l * i32_slot_st
        i32_slot_st: fx.Int32,
        i32_nsl: fx.Int32,  # slots given: tokens >= i32_nsl are not inserted
        i32_eps: fx.Int32,  # fp32 bits
        i32_cs_rows: fx.Int32,
        i64_page_b: fx.Int64,  # cache strides (bytes)
        i32_head_b: fx.Int32,
        i32_tok_b: fx.Int32,
    ):
        tx = fx.thread_idx.x
        j = tx % GL
        g = fx.block_idx.x * HPB + tx // GL
        tok = g // S
        s = g % S
        live = tok < i32_n
        x_r = buffer_ops.create_buffer_resource_from_addr(
            arg_x, num_records_bytes=fx.Int64(i32_n) * fx.Int64(i32_row_st) * 2
        )
        xv = Vec(
            buffer_ops.buffer_load(
                x_r, tok * (i32_row_st >> 1) + s * (D // 2) + j * 4, vec_width=4, dtype=fx.Int32, mask=live
            )
        )
        pos_r = buffer_ops.create_buffer_resource_from_addr(arg_pos, num_records_bytes=fx.Int64(i32_n) * 8)
        pos = fx.Int32(buffer_ops.buffer_load(pos_r, tok * 2, vec_width=1, dtype=fx.Int32, mask=live))
        pos = (pos < 0).select(pos + i32_cs_rows, pos)
        cs_r = buffer_ops.create_buffer_resource_from_addr(
            arg_cs, num_records_bytes=fx.Int64(i32_cs_rows) * (ROT * 2)
        )
        cs_off = pos * (ROT // 2) + (j % 4) * 4
        cosv = Vec(buffer_ops.buffer_load(cs_r, cs_off, vec_width=4, dtype=fx.Int32))
        sinv = Vec(buffer_ops.buffer_load(cs_r, cs_off + HALF // 2, vec_width=4, dtype=fx.Int32))

        t = s - nq
        if const_expr(nq > 0):
            is_q = s < nq
            lay = fx.Int32(0)
        else:
            lay = t // (2 * nkv)
        r = t % (2 * nkv)
        is_k = r < nkv
        h = r % nkv
        kw_r = buffer_ops.create_buffer_resource_from_addr(arg_kw, num_records_bytes=nl * D * 2)
        if const_expr(nq > 0):
            qw_r = buffer_ops.create_buffer_resource_from_addr(arg_qw, num_records_bytes=D * 2)
            wq = Vec(buffer_ops.buffer_load(qw_r, j * 4, vec_width=4, dtype=fx.Int32))
            wk = Vec(buffer_ops.buffer_load(kw_r, j * 4, vec_width=4, dtype=fx.Int32))
            wv = [is_q.select(fx.Int32(wq[i]), fx.Int32(wk[i])) for i in range_constexpr(4)]
        else:
            wv = Vec(buffer_ops.buffer_load(kw_r, lay * (D // 2) + j * 4, vec_width=4, dtype=fx.Int32))

        xs = _bf16x8(xv)
        ws = _bf16x8(wv)
        cs = _bf16x8(cosv)
        sn = _bf16x8(sinv)
        ss = _sum_sq(xs, round_norm)
        rr = fx.Float32(
            fx.rocdl.rsq(T.f32, fx.as_ir_value(ss * fx.Float32(1.0 / D) + i32_eps.bitcast(fx.Float32)))
        )
        nv = [(xs[e] * rr) * ws[e] for e in range_constexpr(8)]
        if const_expr(round_norm):
            yw = [_pk_bf16(nv[2 * e], nv[2 * e + 1]) for e in range_constexpr(4)]
            y = _bf16x8(yw)
            p = _bf16x8([_xor_lane(w, 4) for w in yw])
        else:
            y = nv
            p = [_xor_f(v, 4) for v in nv]
        first = j < 4
        o = []
        for e in range_constexpr(8):
            ps = p[e] * sn[e]
            o.append(fx.math.fma(y[e], cs[e], first.select(-ps, ps)))
        ow = [_pk_bf16(o[2 * e], o[2 * e + 1]) for e in range_constexpr(4)]

        if const_expr(nq > 0):
            q_r = buffer_ops.create_buffer_resource_from_addr(
                arg_q, num_records_bytes=fx.Int64(i32_n) * (nq * D * 2)
            )
            buffer_ops.buffer_store(
                Vec.from_elements(ow, fx.Int32), q_r, tok * (nq * D // 2) + s * (D // 2) + j * 4, mask=live & is_q
            )
        if const_expr(insert):
            kv_side = live & (tok < i32_nsl)
            if const_expr(nq > 0):
                kv_side = kv_side & (s >= nq)
            sl_r = buffer_ops.create_buffer_resource_from_addr(
                arg_slot, num_records_bytes=(fx.Int64((nl - 1) * i32_slot_st) + fx.Int64(i32_nsl)) * 8
            )
            slot = fx.Int32(
                buffer_ops.buffer_load(sl_r, (lay * i32_slot_st + tok) * 2, vec_width=1, dtype=fx.Int32, mask=kv_side)
            )
            ok = kv_side & (slot >= 0)
            slot = ok.select(slot, fx.Int32(0))
            if const_expr(TABLE):
                ptrs = _gptr(arg_kv, T.i64, 8)
                base = fx.Int64(ptrs[lay])
            else:
                base = fx.Int64(arg_kv)
            addr = (
                base
                + fx.Int64(slot // BS) * i64_page_b
                + fx.Int64(h * i32_head_b + (slot % BS) * i32_tok_b + is_k.select(fx.Int32(0), fx.Int32(D * ELT)))
                + fx.Int64(j * (8 * ELT))
            )
            if const_expr(cache == "bf16"):
                data = [is_k.select(ow[e], fx.Int32(xv[e])) for e in range_constexpr(4)]
            else:
                if const_expr(TABLE):
                    ks_a = fx.Int64(ptrs[nl + lay])
                    vs_a = fx.Int64(ptrs[2 * nl + lay])
                else:
                    ks_a = fx.Int64(arg_ks)
                    vs_a = fx.Int64(arg_vs)
                sc = is_k.select(fx.Float32(_gptr(ks_a, T.f32, 4)[0]), fx.Float32(_gptr(vs_a, T.f32, 4)[0]))
                src = _bf16x8([is_k.select(ow[e], fx.Int32(xv[e])) for e in range_constexpr(4)])
                vals = [v / sc for v in src]
                _set_fp16_ovfl()
                data = _e4m3x8(vals)
            if ok:
                _gstore(addr, data)

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_pos: fx.Int64,
        arg_cs: fx.Int64,
        arg_qw: fx.Int64,
        arg_kw: fx.Int64,
        arg_q: fx.Int64,
        arg_kv: fx.Int64,
        arg_ks: fx.Int64,
        arg_vs: fx.Int64,
        arg_slot: fx.Int64,
        i32_n: fx.Int32,
        i32_row_st: fx.Int32,
        i32_slot_st: fx.Int32,
        i32_nsl: fx.Int32,
        i32_eps: fx.Int32,
        i32_cs_rows: fx.Int32,
        i64_page_b: fx.Int64,
        i32_head_b: fx.Int32,
        i32_tok_b: fx.Int32,
        stream: fx.Stream,
    ):
        heads = i32_n * S
        kernel(
            i32_n, i32_row_st, arg_x, arg_pos, arg_cs, arg_qw, arg_kw, arg_q, arg_kv, arg_ks, arg_vs, arg_slot,
            i32_slot_st, i32_nsl, i32_eps, i32_cs_rows, i64_page_b, i32_head_b, i32_tok_b,
        ).launch(grid=(fx.Int64((heads + (HPB - 1)) // HPB), 1, 1), block=(64 * NW, 1, 1), stream=stream)  # fmt: skip

    launch.kernel_name = name
    launch.compile_hints = PRELOAD_HINTS
    return launch
