# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Compile cache and launch wrapper of the DSA prologue's q_a / kv_a norms and fp8_ds_mla KV insert."""

import functools
import struct

import torch

from amd_kernels.warm_up import entry

from ..common import on_gfx950
from . import kernel as _kernel


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_kernel(insert: bool):
    return _kernel.compile_qkv_a_norm_kv_insert(insert)


def supports_qkv_a_norm_kv_insert(qkv_a: torch.Tensor, kv_cache: torch.Tensor | None) -> bool:
    if not on_gfx950() or qkv_a.dtype != torch.bfloat16 or qkv_a.dim() != 2 or qkv_a.stride(1) != 1:
        return False
    if qkv_a.shape[1] < _kernel.ROW or qkv_a.stride(0) % 8:
        return False
    if kv_cache is None:
        return True
    return (
        kv_cache.dim() == 3
        and kv_cache.dtype == torch.uint8
        and kv_cache.shape[1] == 64
        and kv_cache.shape[2] == _kernel.RECORD
        and kv_cache.stride(2) == 1
        and kv_cache.stride(1) == _kernel.RECORD
        and kv_cache.stride(0) % 16 == 0
    )


def _qkv_a_norm_kv_insert(
    qkv_a: torch.Tensor,  # [n, 2112 (+ pad)] bf16 rows [q 1536 | kv 512 | k_pe 64], unit column stride
    q_norm_weight: torch.Tensor,  # [1536] bf16
    kv_norm_weight: torch.Tensor,  # [512] bf16
    eps: float,
    q_out: torch.Tensor,  # [n, 1536] bf16 contiguous
    kv_out: torch.Tensor,  # [n, 512] bf16 contiguous
    kv_cache: torch.Tensor | None = None,  # uint8 [pages, 64, 656] fp8_ds_mla
    slot_mapping: torch.Tensor | None = None,  # int64 [<= n], < 0 skips the token
) -> None:
    """q_out = RMSNorm(q), kv_out = RMSNorm(kv); with ``kv_cache``, the fp8_ds_mla record of
    (kv_out, k_pe) of the first len(slot_mapping) tokens goes to ``slot_mapping``."""
    n = qkv_a.shape[0]
    if n == 0:
        return
    assert qkv_a.dtype == torch.bfloat16 and qkv_a.stride(1) == 1 and qkv_a.stride(0) % 8 == 0
    assert qkv_a.shape[1] >= _kernel.ROW
    for w, d in ((q_norm_weight, _kernel.Q), (kv_norm_weight, _kernel.KV)):
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.numel() == d
    assert q_out.dtype == kv_out.dtype == torch.bfloat16 and q_out.is_contiguous() and kv_out.is_contiguous()
    assert q_out.shape == (n, _kernel.Q) and kv_out.shape == (n, _kernel.KV)
    insert = kv_cache is not None and slot_mapping is not None
    if insert:
        assert supports_qkv_a_norm_kv_insert(qkv_a, kv_cache)
        assert slot_mapping.dtype == torch.int64 and slot_mapping.is_contiguous()
        n_slots = min(n, slot_mapping.shape[0])
        page_b = kv_cache.stride(0)
        cache_ptr, slot_ptr = kv_cache.data_ptr(), slot_mapping.data_ptr()
    else:
        n_slots = page_b = cache_ptr = slot_ptr = 0
    _run_compiled(
        _get_kernel(insert),
        qkv_a.data_ptr(),
        q_norm_weight.data_ptr(),
        kv_norm_weight.data_ptr(),
        q_out.data_ptr(),
        kv_out.data_ptr(),
        cache_ptr,
        slot_ptr,
        n,
        n_slots,
        qkv_a.stride(0),
        _kernel.RECORD,
        struct.unpack("<i", struct.pack("<f", eps))[0],
        page_b,
        torch.cuda.current_stream(),
    )


def _mode(qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache=None, slot_mapping=None):
    """What a call fixes about its kernel (``amd_kernels.warm_up``; the token count does not)."""
    return (kv_cache is not None and slot_mapping is not None,)


def _inputs(mode, n, qkv_a, q_norm_weight, kv_norm_weight, eps, q_out, kv_out, kv_cache=None, slot_mapping=None):
    """One token (zeros; slot -1 writes nothing)."""
    dev = qkv_a.device
    x = torch.zeros((n, qkv_a.shape[1]), dtype=torch.bfloat16, device=dev)
    q = torch.empty((n, _kernel.Q), dtype=torch.bfloat16, device=dev)
    kv = torch.empty((n, _kernel.KV), dtype=torch.bfloat16, device=dev)
    slots = torch.full((n,), -1, dtype=torch.int64, device=dev) if mode[0] else None
    return (x, q_norm_weight, kv_norm_weight, eps, q, kv, kv_cache if mode[0] else None, slots), {}


qkv_a_norm_kv_insert = entry(
    "attention/dsa_prologue_qkv_a_norm_kv_insert_h7168:qkv_a", mode=_mode, sizes=lambda mode: (1,), inputs=_inputs
)(_qkv_a_norm_kv_insert)
