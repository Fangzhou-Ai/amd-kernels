# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""RMS norm + NeoX RoPE (all 64 dims) + bf16 / e4m3 K/V cache insert of 64-dim heads, FlyDSL (gfx950):
``qkv_norm_rope_insert`` for one attention layer's fused qkv projection output, ``kv_norm_rope_insert_multilayer``
for one fused K/V projection row that feeds every layer's cache. The contract is ``reference.py``."""

import torch

ROTARY_DIM = 64


def rope_blocker(head_size: int, rotary_dim: int, neox: bool, cos_sin_cache: torch.Tensor) -> str | None:
    """Why a layer's RoPE does not fit these kernels (None: it fits): NeoX over all 64 dims of a 64-dim head, a
    contiguous bf16 cos | sin cache [rows, 64]."""
    if head_size != ROTARY_DIM or rotary_dim != ROTARY_DIM or not neox:
        return f"RoPE on {rotary_dim} of {head_size} dims{'' if neox else ', not NeoX'} (NeoX on 64 of 64)"
    c = cos_sin_cache
    if c.dtype != torch.bfloat16 or c.dim() != 2 or c.shape[1] != ROTARY_DIM or not c.is_contiguous():
        return f"cos / sin cache {tuple(c.shape)} {c.dtype} (contiguous bf16 [rows, {ROTARY_DIM}])"
    return None
