# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""K norm + RoPE and the K / V inserts of the context rows' fused K / V projection into every layer's cache
(``kernel.py``, the eager form: RoPE on the bf16 normed K)."""

import functools
import struct

import torch

from amd_kernels.warm_up import entry

from . import kernel as _kernel
from .qkv_norm_rope_insert import _run_compiled, cache_kind

D = _kernel.D


@functools.cache
def _get_kernel(nl: int, nkv: int, block_size: int, cache: str):
    return _kernel.compile_norm_rope_insert(0, nkv, nl, block_size, cache, round_norm=True)


@functools.cache
def _pointer_table(ptrs: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Persistent cache and scale pointers only (never batch slot views): built on the first call, before capture."""
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def supports_multilayer(kv_caches: list[torch.Tensor], k_scales: list[torch.Tensor] | None,
                        v_scales: list[torch.Tensor] | None) -> bool:
    """Whether ``kv_norm_rope_insert_multilayer`` takes these caches: bf16, or e4m3 (or its uint8 view) with fp32
    scales, ``[pages, kv_heads, page, 128]``, unit last stride, every layer the same dtype, shape and strides."""
    if not kv_caches or any(c.numel() == 0 for c in kv_caches):
        return False
    kind = cache_kind(kv_caches[0])
    ref = (kv_caches[0].dtype, tuple(kv_caches[0].shape[1:]), kv_caches[0].stride())
    if kind is None or any((c.dtype, tuple(c.shape[1:]), c.stride()) != ref for c in kv_caches):
        return False
    if kind == "bf16":
        return True
    return (
        k_scales is not None and v_scales is not None and len(k_scales) == len(v_scales) == len(kv_caches)
        and all(s.dtype == torch.float32 for s in (*k_scales, *v_scales))
    )


def _kv_norm_rope_insert_multilayer(
    kv: torch.Tensor,  # [n, nl * 2 * nkv * 64] bf16, per layer K heads then V heads; unit column stride
    positions: torch.Tensor,  # [n] int64
    k_norm_weight: torch.Tensor,  # [nl, 64] bf16
    cos_sin_cache: torch.Tensor,  # [rows, 64] bf16 (cos | sin)
    kv_caches: list[torch.Tensor],  # nl x [pages, nkv, page, 128] bf16 or e4m3
    slot_mappings: list[torch.Tensor],  # nl x [>= n_insert] int64 (< 0 skips)
    k_scales: list[torch.Tensor] | None,  # nl x fp32 scalar (e4m3 caches)
    v_scales: list[torch.Tensor] | None,
    eps: float,
    n_insert: int | None = None,
) -> None:
    """K norm + RoPE and the K / V inserts of the context rows into every layer's cache; only the first
    ``n_insert`` rows (None: all) have slots."""
    n = kv.shape[0]
    nl = len(kv_caches)
    if n == 0 or nl == 0:
        return
    nkv = kv_caches[0].shape[1]
    block_size = kv_caches[0].shape[2]
    cache = cache_kind(kv_caches[0])
    assert cache is not None
    assert kv.dtype == torch.bfloat16 and kv.stride(1) == 1 and kv.stride(0) % 8 == 0
    assert kv.shape[1] == nl * 2 * nkv * D
    assert positions.dtype == torch.int64 and positions.is_contiguous()
    assert k_norm_weight.shape == (nl, D) and k_norm_weight.is_contiguous()
    assert k_norm_weight.dtype == torch.bfloat16
    assert cos_sin_cache.dtype == torch.bfloat16 and cos_sin_cache.is_contiguous()
    assert cos_sin_cache.shape[1] == _kernel.ROT
    for c in kv_caches:
        assert c.dtype == kv_caches[0].dtype and c.stride() == kv_caches[0].stride()
        assert tuple(c.shape[1:]) == tuple(kv_caches[0].shape[1:])
    n_insert = n if n_insert is None else min(n_insert, n)
    if cache == "fp8":
        scale_ptrs = [s.data_ptr() for s in k_scales] + [s.data_ptr() for s in v_scales]
    else:
        scale_ptrs = [0] * (2 * nl)
    table = _pointer_table(tuple([c.data_ptr() for c in kv_caches] + scale_ptrs), kv.device)
    slots = slot_mappings[0]
    slot_stride = 0
    if any(s.data_ptr() != slots.data_ptr() for s in slot_mappings):
        slots = torch.stack([s[:n_insert] for s in slot_mappings])
        slot_stride = slots.stride(0)
    assert slots.dtype == torch.int64 and slots.stride(-1) == 1
    es = kv_caches[0].element_size()
    _run_compiled(
        _get_kernel(nl, nkv, block_size, cache),
        kv.data_ptr(),
        positions.data_ptr(),
        cos_sin_cache.data_ptr(),
        0,
        k_norm_weight.data_ptr(),
        0,
        table.data_ptr(),
        0,
        0,
        slots.data_ptr(),
        n,
        kv.stride(0),
        slot_stride,
        n_insert,
        struct.unpack("<I", struct.pack("<f", eps))[0],
        cos_sin_cache.shape[0],
        kv_caches[0].stride(0) * es,
        kv_caches[0].stride(1) * es,
        kv_caches[0].stride(2) * es,
        torch.cuda.current_stream(kv.device),
    )


def _ml_mode(kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales, v_scales, eps,
             n_insert=None):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the row count does not)."""
    return len(kv_caches), kv_caches[0].shape[1], kv_caches[0].shape[2], cache_kind(kv_caches[0])


def _ml_inputs(mode, n, kv, positions, k_norm_weight, cos_sin_cache, kv_caches, slot_mappings, k_scales, v_scales,
               eps, n_insert=None):
    """``n`` rows of zeros with slot -1 (writes nothing; the caller's caches)."""
    nl, nkv, _, _ = mode
    dev = positions.device
    x = torch.zeros((n, nl * 2 * nkv * D), dtype=torch.bfloat16, device=dev)
    pos = torch.zeros(n, dtype=torch.int64, device=dev)
    slots = torch.full((n,), -1, dtype=torch.int64, device=dev)
    return (x, pos, k_norm_weight, cos_sin_cache, kv_caches, [slots] * nl, k_scales, v_scales, eps), {}


kv_norm_rope_insert_multilayer = entry("attention/norm_rope_insert_d64_rot64:multilayer", mode=_ml_mode,
                                       sizes=lambda mode: (8,), inputs=_ml_inputs)(_kv_norm_rope_insert_multilayer)
