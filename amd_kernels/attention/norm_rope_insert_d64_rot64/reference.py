# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract of ``qkv_norm_rope_insert`` and ``kv_norm_rope_insert_multilayer`` (root README, rule 3): torch
only.

Per 64-dim head ``x`` (bf16), both kernels compute

1. RMS norm in fp32: the sum of squares ``s = x * x`` (rounded) of 8 groups of 8 consecutive dims, in the order of
   the op replaced. Eager (vLLM's ``rms_norm``): each group ``((s0 + s1) + s2) + ... + s7``, then the groups
   ((t0 + t1) + (t2 + t3)) + ((t4 + t5) + (t6 + t7)). Compiled (Inductor's persistent reduction, 8 lanes x 8):
   each group ((s0 + s2) + fma(x4, x4, s6)) + ((s1 + s3) + fma(x5, x5, s7)), then ((t0 + t4) + (t2 + t6)) +
   ((t1 + t5) + (t3 + t7)). Then ``r = rsqrt(sum / 64 + eps)`` (the hardware ``v_rsq_f32``, not correctly
   rounded: here ``torch.rsqrt`` on the tensor's device) and ``n = (x * r) * w`` (fp32 products).
2. NeoX RoPE over all 64 dims (halves of 32): ``o1 = fma(n1, cos, -(n2 sin))``, ``o2 = fma(n2, cos, n1 sin)`` in
   fp32, rounded to bf16 once. ``cos_sin_cache [rows, 64]`` is cos | sin per position; a negative position wraps
   (``pos + rows``). ``kv_norm_rope_insert_multilayer`` first rounds ``n`` to bf16 (vLLM's eager ``rms_norm`` then
   ``rotary_embedding``: the products of two bf16 are exact, so the fma form does not matter there);
   ``qkv_norm_rope_insert`` rotates the fp32 ``n`` (Inductor's fused norm + RoPE of the compiled draft layer).
3. Insert into ``[pages, kv_heads, page, 128]`` at ``[slot // page, head, slot % page]``: K (64) then V (64), V as
   given. A bf16 cache takes the bf16 values; an e4m3 cache takes e4m3 of ``value / scale`` (IEEE fp32 divide,
   RNE, saturated at +-448). A slot < 0, or a token at or past the slots given, is skipped.

Tolerances (``check_bf16``, ``check_cache``): ``torch.rsqrt`` may differ from ``v_rsq_f32`` by an ulp, so a normed
value may sit one ulp from this reference, and RoPE can move that ulp into a smaller output or cancel it near zero;
an Inductor build that picks another reduction layout sums in another order. Every output is within one step (bf16 ulp / e4m3 code) of the reference or within ``ROW_ATOL`` (2^-6)
of its 64-value row's largest reference magnitude; at least ``EXACT_MIN`` bit-equal; every cache byte the kernel
must not write is unchanged.
"""

import torch

D = 64
ROT = 64
HALF = ROT // 2
FP8_MAX = 448.0
EXACT_MIN = 0.999
ROW_ATOL = 2.0**-6


def _ro_f32(s: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """fp32 of the exact value ``s + e`` (``s = RN64(s + e)``, ``e`` its error), correctly rounded: round ``s`` to
    odd at fp64 by ``e``, then RNE to fp32 (fp64 keeps more than 2 extra bits, so no double rounding)."""
    bits = s.view(torch.int64)
    toward_zero = (e != 0) & ((e > 0) != (s > 0))
    bits = torch.where(toward_zero & (s != 0), bits - 1, bits)
    bits = torch.where(e != 0, bits | 1, bits)
    return bits.view(torch.float64).float()


def fma32(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """fp32 ``a * b + c`` with one rounding (the product of two fp32 is exact in fp64; two-sum keeps the error)."""
    p = a.double() * b.double()
    cd = c.double()
    s = p + cd
    bb = s - p
    e = (p - (s - bb)) + (cd - bb)
    return _ro_f32(s, e)


def sum_sq(x: torch.Tensor, eager: bool) -> torch.Tensor:
    """``x [..., 64]`` fp32 -> ``[..., 1]``, step 1's order."""
    g = x.reshape(*x.shape[:-1], 8, 8)
    sq = g * g
    if eager:
        t = sq[..., 0]
        for e in range(1, 8):
            t = t + sq[..., e]
        a = [t[..., i] for i in range(8)]
        s = ((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))
    else:
        t = ((sq[..., 0] + sq[..., 2]) + fma32(g[..., 4], g[..., 4], sq[..., 6])) + (
            (sq[..., 1] + sq[..., 3]) + fma32(g[..., 5], g[..., 5], sq[..., 7])
        )
        a = [t[..., i] for i in range(8)]
        s = ((a[0] + a[4]) + (a[2] + a[6])) + ((a[1] + a[5]) + (a[3] + a[7]))
    return s.unsqueeze(-1)


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float, eager: bool) -> torch.Tensor:
    """``x [..., 64]`` bf16, ``w [64]`` bf16 -> fp32 ``n`` (step 1)."""
    x32 = x.float()
    r = torch.rsqrt(sum_sq(x32, eager) * (1.0 / D) + torch.tensor(eps, dtype=torch.float32))
    return (x32 * r) * w.float()


def rope_neox(n: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor) -> torch.Tensor:
    """``n [tokens, heads, 64]`` fp32 -> bf16 (step 2)."""
    rows = cos_sin_cache.shape[0]
    pos = torch.where(positions < 0, positions + rows, positions).long()
    cos = cos_sin_cache[pos, :HALF].float()[:, None, :]
    sin = cos_sin_cache[pos, HALF:].float()[:, None, :]
    n1, n2 = n[..., :HALF], n[..., HALF:]
    o1 = fma32(n1, cos, -(n2 * sin))
    o2 = fma32(n2, cos, n1 * sin)
    return torch.cat([o1, o2], dim=-1).bfloat16()


def e4m3_codes(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    v = (x.float() / scale.float()).clamp(-FP8_MAX, FP8_MAX)
    return v.to(torch.float8_e4m3fn).view(torch.uint8)


def insert_reference(cache, k, v, slots, k_scale=None, v_scale=None):
    """Write ``k``, ``v [n, heads, 64]`` bf16 into a copy of ``cache [pages, heads, page, 128]`` (bf16, or uint8
    e4m3) at ``slots [n]`` (< 0 skipped). Returns (the copy, the mask of the elements written)."""
    out = cache.clone()
    mask = torch.zeros(cache.shape, dtype=torch.bool, device=cache.device)
    page = cache.shape[2]
    keep = slots >= 0
    s = slots[keep].long()
    if cache.dtype == torch.bfloat16:
        kk, vv = k[keep], v[keep]
    else:
        kk, vv = e4m3_codes(k[keep], k_scale), e4m3_codes(v[keep], v_scale)
        out = out.view(torch.uint8)
    out[s // page, :, s % page, :D] = kk
    out[s // page, :, s % page, D:] = vv
    mask[s // page, :, s % page] = True
    return out.view(cache.dtype), mask


def qkv_norm_rope_insert_reference(
    qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads,
    num_kv_heads, kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None,
):  # fmt: skip
    """-> (q_out ``[n, num_heads * 64]`` bf16, (the updated cache copy, the mask of the elements written) or
    None)."""
    n = qkv.shape[0]
    q, k, v = qkv.reshape(n, -1, D).split([num_heads, num_kv_heads, num_kv_heads], dim=1)
    q = rope_neox(rms_norm(q, q_norm_weight, eps, False), positions, cos_sin_cache)
    if kv_cache is None:
        return q.reshape(n, num_heads * D), None
    k = rope_neox(rms_norm(k, k_norm_weight, eps, False), positions, cos_sin_cache)
    m = min(n, slot_mapping.shape[0])
    cache = insert_reference(kv_cache, k[:m], v[:m], slot_mapping[:m], k_scale, v_scale)
    return q.reshape(n, num_heads * D), cache


def kv_norm_rope_insert_multilayer_reference(
    kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales,
    v_scales, eps, n_insert=None,
):  # fmt: skip
    """``kv [n, layers * 2 * kv_heads * 64]`` (per layer: K heads then V heads) -> per layer (the updated copy of
    its cache, the mask of the elements written); rows >= ``n_insert`` are not inserted."""
    n = kv.shape[0]
    nl = len(kv_caches)
    nkv = kv_caches[0].shape[1]
    m = n if n_insert is None else min(n_insert, n)
    x = kv.reshape(n, nl, 2, nkv, D)
    out = []
    for lay in range(nl):
        normed = rms_norm(x[:, lay, 0], k_norm_weight[lay], eps, True).bfloat16().float()
        k = rope_neox(normed, positions, cos_sin_cache)
        ks = None if k_scales is None else k_scales[lay]
        vs = None if v_scales is None else v_scales[lay]
        out.append(insert_reference(kv_caches[lay], k[:m], x[:m, lay, 1], slot_mappings[lay][:m], ks, vs))
    return out


def _near(g_bits, r_bits, g_val, r_val, sign_bit, mag_mask):
    """Rows of 64: within one step (same sign, adjacent magnitude bits; both formats are sign-magnitude) or within
    ROW_ATOL of the row's largest reference magnitude."""
    step = ((g_bits ^ r_bits) & sign_bit == 0) & ((g_bits & mag_mask) - (r_bits & mag_mask)).abs().le(1)
    row_max = r_val.abs().amax(-1, keepdim=True)
    return step | ((g_val - r_val).abs() <= row_max * ROW_ATOL)


def check_bf16(got: torch.Tensor, ref: torch.Tensor, name: str) -> float:
    """bf16 heads: every value near the reference (``_near``), at least EXACT_MIN bit-equal; returns the bit-equal
    fraction."""
    g, r = got.reshape(-1, D), ref.reshape(-1, D)
    gb, rb = g.view(torch.int16).int(), r.view(torch.int16).int()
    near = _near(gb, rb, g.float(), r.float(), 0x8000, 0x7FFF)
    frac = (gb == rb).float().mean().item()
    assert bool(near.all()), (name, int((~near).sum()))
    assert frac >= EXACT_MIN, (name, frac)
    return frac


def check_cache(got, ref, before, written, name: str) -> float:
    """Cache pages (bf16 or uint8 e4m3): near the reference (``_near``, per 64-element K or V row) where the kernel
    writes (``written``), at least EXACT_MIN of those elements bit-equal, every other element unchanged; returns the
    bit-equal fraction of the written elements."""
    if got.dtype != torch.bfloat16:
        got, ref, before = got.view(torch.uint8), ref.view(torch.uint8), before.view(torch.uint8)
    assert torch.equal(got[~written], before[~written]), (name, "an element the kernel must not write changed")
    rows = written.reshape(-1, D).all(-1)
    g, r = got.reshape(-1, D)[rows], ref.reshape(-1, D)[rows]
    if got.dtype == torch.bfloat16:
        gb, rb = g.view(torch.int16).int(), r.view(torch.int16).int()
        near = _near(gb, rb, g.float(), r.float(), 0x8000, 0x7FFF)
        eq = gb == rb
    else:
        gv, rv = g.view(torch.float8_e4m3fn).float(), r.view(torch.float8_e4m3fn).float()
        near = _near(g.int(), r.int(), gv, rv, 0x80, 0x7F)
        eq = g == r
    assert bool(near.all()), (name, int((~near).sum()))
    frac = eq.float().mean().item() if g.numel() else 1.0
    assert frac >= EXACT_MIN, (name, frac)
    return frac
