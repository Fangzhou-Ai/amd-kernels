# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""One layer's q / k norm + RoPE and K / V insert of its fused qkv rows (``kernel.py``, the compiled-graph form:
RoPE on the fp32 normed values)."""

import functools
import struct

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel
from .device import on_gfx950

D = _kernel.D


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(nq: int, nkv: int, block_size: int, cache):
    return _kernel.compile_norm_rope_insert(nq, nkv, 1, block_size, cache, round_norm=False)


def cache_kind(kv_cache: torch.Tensor) -> str | None:
    """"bf16" or "fp8" for a ``[pages, kv_heads, page, 128]`` cache with unit last stride, else None."""
    if kv_cache.dim() != 4 or kv_cache.shape[3] != 2 * D or kv_cache.stride(3) != 1:
        return None
    if kv_cache.dtype == torch.bfloat16:
        return "bf16"
    if kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn):
        return "fp8"
    return None


def supports_qkv_norm_rope_insert(
    qkv: torch.Tensor, cos_sin_cache: torch.Tensor, kv_cache: torch.Tensor | None
) -> bool:
    if not on_gfx950() or qkv.dtype != torch.bfloat16 or qkv.stride(1) != 1 or qkv.stride(0) % 8:
        return False
    if cos_sin_cache.dtype != torch.bfloat16 or cos_sin_cache.shape[1] != _kernel.ROT:
        return False
    return kv_cache is None or cache_kind(kv_cache) is not None


def _qkv_norm_rope_insert(
    qkv: torch.Tensor,  # [n, (nq + 2 nkv) * 64] bf16, unit column stride, row stride % 8 == 0
    positions: torch.Tensor,  # [n] int64
    q_norm_weight: torch.Tensor,  # [64] bf16
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,  # [max_pos, 64] bf16 (cos | sin)
    eps: float,
    num_heads: int,
    num_kv_heads: int,
    q_out: torch.Tensor,  # [n, nq * 64] bf16
    kv_cache: torch.Tensor | None = None,  # [pages, nkv, page, 128] bf16 or e4m3 (K 64 | V 64)
    slot_mapping: torch.Tensor | None = None,  # int64, < 0 skips the token
    k_scale: torch.Tensor | None = None,  # fp32 scalars (e4m3 cache)
    v_scale: torch.Tensor | None = None,
) -> None:
    """q_out = RoPE(norm(q)); with ``kv_cache``, RoPE(norm(k)) and v of the first len(slot_mapping) tokens go into
    the pages at ``slot_mapping``."""
    n = qkv.shape[0]
    if n == 0:
        return
    nq, nkv = num_heads, num_kv_heads
    assert qkv.shape[1] == (nq + 2 * nkv) * D and qkv.stride(1) == 1 and qkv.stride(0) % 8 == 0
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert cos_sin_cache.is_contiguous() and cos_sin_cache.shape[1] == _kernel.ROT
    assert cos_sin_cache.dtype == torch.bfloat16
    for w in (q_norm_weight, k_norm_weight):
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == D
    assert q_out.dtype == torch.bfloat16 and q_out.is_contiguous() and q_out.shape[1] == nq * D
    insert = kv_cache is not None and slot_mapping is not None
    if insert:
        cache = cache_kind(kv_cache)
        assert cache is not None and kv_cache.shape[1] == nkv
        assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
        n_slots = min(n, slot_mapping.shape[0])
        block_size = kv_cache.shape[2]
        es = kv_cache.element_size()
        page_b, head_b, tok_b = kv_cache.stride(0) * es, kv_cache.stride(1) * es, kv_cache.stride(2) * es
        kv_ptr, sl_ptr = kv_cache.data_ptr(), slot_mapping.data_ptr()
        if cache == "fp8":
            ks_ptr, vs_ptr = k_scale.data_ptr(), v_scale.data_ptr()
        else:
            ks_ptr = vs_ptr = 0
    else:
        cache, block_size, page_b, head_b, tok_b, n_slots = None, 1, 0, 0, 0, 0
        kv_ptr = sl_ptr = ks_ptr = vs_ptr = 0
    _run_compiled(
        _get_kernel(nq, nkv, block_size, cache),
        qkv.data_ptr(),
        positions.data_ptr(),
        cos_sin_cache.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        q_out.data_ptr(),
        kv_ptr,
        ks_ptr,
        vs_ptr,
        sl_ptr,
        n,
        qkv.stride(0),
        0,
        n_slots,
        struct.unpack("<I", struct.pack("<f", eps))[0],
        cos_sin_cache.shape[0],
        page_b,
        head_b,
        tok_b,
        torch.cuda.current_stream(qkv.device),
    )


def _qkv_mode(qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads, q_out,
              kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the token count does not)."""
    insert = kv_cache is not None and slot_mapping is not None
    return num_heads, num_kv_heads, cache_kind(kv_cache) if insert else None, kv_cache.shape[2] if insert else 1


def _qkv_inputs(mode, n, qkv, positions, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads,
                q_out, kv_cache=None, slot_mapping=None, k_scale=None, v_scale=None):
    """``n`` tokens of zeros; slot -1 writes nothing (the caller's cache)."""
    dev = qkv.device
    x = torch.zeros((n, qkv.shape[1]), dtype=qkv.dtype, device=dev)
    pos = torch.zeros(n, dtype=torch.int64, device=dev)
    q = torch.empty((n, num_heads * D), dtype=torch.bfloat16, device=dev)
    insert = mode[2] is not None
    slots = torch.full((n,), -1, dtype=torch.int64, device=dev) if insert else None
    return (x, pos, q_norm_weight, k_norm_weight, cos_sin_cache, eps, num_heads, num_kv_heads, q,
            kv_cache if insert else None, slots, k_scale, v_scale), {}


qkv_norm_rope_insert = entry("attention/norm_rope_insert_d64_rot64:qkv", mode=_qkv_mode,
                             sizes=lambda mode: (1,), inputs=_qkv_inputs)(_qkv_norm_rope_insert)
