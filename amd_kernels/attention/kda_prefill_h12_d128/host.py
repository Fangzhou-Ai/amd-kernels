# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry points of the KDA prefill kernels: stage launches (kernels in scan.py / norm.py, contract in reference.py)."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled

from . import norm as _norm
from . import scan as _scan
from .reference import D, NH, RECORD_BYTES

SCAN_WAVES = 4
SCAN_NVT = 1


@functools.cache
def _scan_launcher():
    return _scan.compile_scan(SCAN_WAVES, SCAN_NVT)


@functools.cache
def _norm_launcher(w_f32):
    return _norm.compile_norm(4, 4, w_f32)


def scan(ws, cu_seqlens, chunk_offsets, state_cache, state_indices, has_initial_state, out):
    """Chunk scan over the workspace records ``ws`` [H, Nc, RECORD_BYTES]; updates ``state_cache`` rows in place and writes the
    bf16 outputs of the tokens into ``out`` [T, 12, 128] (row stride ``out.stride(0)``)."""
    n = cu_seqlens.numel() - 1
    assert ws.dtype == torch.uint8 and ws.shape[0] == NH and ws.shape[2] == RECORD_BYTES and ws.is_contiguous()
    assert state_cache.dtype == torch.float32 and state_cache.shape[1:] == (NH, D, D) and state_cache.stride(1) == D * D
    assert out.dtype == torch.bfloat16 and out.shape[1:] == (NH, D) and out.stride(1) == D and out.stride(2) == 1
    assert chunk_offsets.dtype == torch.int64 and cu_seqlens.dtype == torch.int32 and state_indices.dtype == torch.int32
    has = has_initial_state.to(torch.uint8) if has_initial_state.dtype != torch.uint8 else has_initial_state
    _run_compiled(
        _scan_launcher(), ws.data_ptr(), ws.shape[1], state_cache.data_ptr(), state_cache.stride(0) * 4,
        state_indices.data_ptr(), has.data_ptr(), cu_seqlens.data_ptr(), chunk_offsets.data_ptr(), out.data_ptr(),
        out.stride(0), out.numel() * 2, n * NH, torch.cuda.current_stream())
    return out


def gated_norm(o, gate, norm_weight, eps, out=None):
    """``out = bf16(o * rsqrt(mean(o^2) + eps) * norm_weight * sigmoid(gate))`` per 128-wide row; ``out`` may be ``o``."""
    t = o.shape[0]
    out = o if out is None else out
    assert o.is_contiguous() and out.is_contiguous() and gate.shape == o.shape and gate.stride(2) == 1 and gate.stride(1) == D
    wf32 = norm_weight.dtype == torch.float32
    assert wf32 or norm_weight.dtype == torch.bfloat16
    span = (t - 1) * gate.stride(0) + NH * D
    _run_compiled(_norm_launcher(wf32), o.data_ptr(), gate.data_ptr(), norm_weight.data_ptr(), out.data_ptr(), gate.stride(0),
                  t * NH, span * 2, float(eps), torch.cuda.current_stream())
    return out
