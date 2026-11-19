# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry points of the KDA prefill kernels: stage launches (kernels in scan.py / norm.py, contract in reference.py)."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import norm as _norm
from . import prologue as _prologue
from . import scan as _scan
from .reference import D, NH, RECORD_BYTES

WALK_WAVES = 4
WALK_NVT = 1
CHAIN_WAVES = 4
CHAIN_NVT = 1
CHAIN_STAGES = 2
WALK_STAGES = 2
MIN_CHUNKS = 4
MIN_GROUP_CHUNKS = 5
MAX_GROUPS = 32
NUM_CUS = 256


@functools.cache
def _walk_launcher(ckpt=False, foldwalk=False):
    return _scan.compile_scan(WALK_WAVES, WALK_NVT, False, WALK_STAGES, ckpt, False, foldwalk)


@functools.cache
def _chain_launcher():
    return _scan.compile_scan(CHAIN_WAVES, CHAIN_NVT, True, CHAIN_STAGES)


@functools.cache
def _norm_launcher(w_f32):
    return _norm.compile_norm(4, 2, w_f32)


SHORT_CHAIN = 20  # chunks: a sequence this short (every sequence) is walked whole
MAX_UNITS = 128  # (pair, group) units of one launch: 128 fill the 32 CUs of each of the 8 XCDs (two slice groups per unit)


def plan_groups(chunk_counts):
    """Chunk groups per sequence for sequences of ``chunk_counts`` chunks (host ints): the most groups whose (pair, group) units
    still fit one round of the CUs, none for short chains (the chain launch and the fold cost more than they save)."""
    if max(chunk_counts) <= SHORT_CHAIN:
        return 1
    best = 1
    for g in range(2, MAX_GROUPS + 1):
        units = NH * sum(min(g, max(1, nt // MIN_CHUNKS)) for nt in chunk_counts)
        if units > MAX_UNITS or -(-max(chunk_counts) // g) < MIN_GROUP_CHUNKS:
            break
        best = g
    return best


def chunk_scan(ws, cu_seqlens, chunk_offsets, state_cache, state_indices, has_initial_state, out, chunk_counts=None, groups=None,
         checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """Chunk scan over the workspace records ``ws`` [H, Nc, RECORD_BYTES]; updates ``state_cache`` rows in place and writes the
    bf16 outputs of the tokens into ``out`` [T, 12, 128] (row stride ``out.stride(0)``). ``chunk_counts``: the chunks of each
    sequence as host ints (the grouping of long chains follows it; evenly split when ``None``); ``groups`` forces the grouping.
    ``checkpoint_offsets`` [N] int32 (tokens into the sequence, a multiple of 64, 0 = off) snapshot the state after that many
    tokens into ``checkpoint_state`` [rows, 12, 128, 128] fp32, row ``checkpoint_rows[n]`` (``n`` when ``None``; negative: off)."""
    n = cu_seqlens.numel() - 1
    nc = ws.shape[1]
    assert ws.dtype == torch.uint8 and ws.shape[0] == NH and ws.shape[2] == RECORD_BYTES and ws.is_contiguous()
    assert state_cache.dtype == torch.float32 and state_cache.shape[1:] == (NH, D, D) and state_cache.stride(1) == D * D
    assert out.dtype == torch.bfloat16 and out.shape[1:] == (NH, D) and out.stride(1) == D and out.stride(2) == 1
    assert chunk_offsets.dtype == torch.int64 and cu_seqlens.dtype == torch.int32 and state_indices.dtype == torch.int32
    has = has_initial_state.to(torch.uint8) if has_initial_state.dtype != torch.uint8 else has_initial_state
    if groups is None:
        groups = plan_groups(chunk_counts if chunk_counts is not None else [-(-nc // n)] * n)
    g = groups
    dev = ws.device
    if g > 1:
        gb = torch.empty(n * NH, g, D, D, dtype=torch.float32, device=dev)
        gc = torch.empty(n * NH, g, _scan.FRAGS_C, dtype=torch.uint8, device=dev)
        ga = torch.empty(n * NH, g, D, dtype=torch.float32, device=dev)
        sin = torch.empty(n * NH, g, D, D, dtype=torch.float32, device=dev)
    else:
        gb = gc = ga = sin = ws
    stream = torch.cuda.current_stream()
    common = (ws.data_ptr(), nc, state_cache.data_ptr(), state_cache.stride(0) * 4, state_indices.data_ptr(), has.data_ptr(),
              cu_seqlens.data_ptr(), chunk_offsets.data_ptr(), out.data_ptr(), out.stride(0))
    ckpt = checkpoint_state is not None
    if ckpt:
        assert checkpoint_state.dtype == torch.float32 and checkpoint_state.shape[1:] == (NH, D, D) and checkpoint_offsets.dtype == torch.int32
        rows = checkpoint_rows if checkpoint_rows is not None else torch.arange(n, dtype=torch.int32, device=dev)
        ck = (checkpoint_state.data_ptr(), checkpoint_state.stride(0) * 4, checkpoint_offsets.data_ptr(), rows.data_ptr())
    else:
        ck = (ws.data_ptr(), 0, ws.data_ptr(), ws.data_ptr())
    nbfold = ((n * NH + 7) // 8) * 8 * 8 if g > 1 else 0
    tail = (g, MIN_CHUNKS, n * NH, *ck)
    full = (*common, gb.data_ptr(), gc.data_ptr(), ga.data_ptr(), sin.data_ptr(), *tail)
    if g > 1:
        _run_compiled(_chain_launcher(), *full, 0, stream)
    _run_compiled(_walk_launcher(ckpt, nbfold > 0), *full, nbfold, stream)
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


def _prefill(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
             cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5, table=None,
             chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """The KDA prefill of one layer: conv + prologue, chunk scan, gated RMSNorm. Writes ``out`` [T, 12, 128] bf16, rolls
    ``conv_state`` and updates ``state_cache`` rows in place. ``table``: ``prologue.chunk_table`` of the step (built once per step
    and shared by the layers; built here when ``None``); ``chunk_counts``: the chunks of each sequence as host ints (sizes the
    groups of long chains); ``checkpoint_*``: see ``chunk_scan``."""
    ws = _prologue.prologue(mixed, raw_g, beta_logit, A_log, dt_bias, lower_bound, scale, conv_weight, conv_state, state_indices,
                            has_initial_state, cu_seqlens, chunk_indices, table=table)
    chunk_scan(ws, cu_seqlens, chunk_offsets, state_cache, state_indices, has_initial_state, out, chunk_counts,
         checkpoint_state=checkpoint_state, checkpoint_offsets=checkpoint_offsets, checkpoint_rows=checkpoint_rows)
    return gated_norm(out, gate, norm_weight, norm_eps, out)


def _mode(mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices, has_initial_state,
          cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5, table=None,
          chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """What a call fixes about its kernels (``amd_kernels.warm_up``): the norm weight dtype and the checkpoint export."""
    return (norm_weight.dtype == torch.float32, checkpoint_state is not None)


def _inputs(mode, n, mixed, raw_g, beta_logit, gate, A_log, dt_bias, conv_weight, conv_state, state_cache, state_indices,
            has_initial_state, cu_seqlens, chunk_indices, chunk_offsets, norm_weight, norm_eps, out, lower_bound, scale=D ** -0.5,
            table=None, chunk_counts=None, checkpoint_state=None, checkpoint_offsets=None, checkpoint_rows=None):
    """One sequence of ``n`` tokens (zeros; the launches are compiled, not run)."""
    dev = mixed.device
    chunks = -(-n // 64)
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    ci = torch.tensor([[0, k] for k in range(chunks)], dtype=torch.int32, device=dev)
    co = torch.tensor([0, chunks], dtype=torch.int64, device=dev)
    zero = torch.zeros(1, dtype=torch.int32, device=dev)
    rows = mixed.shape[1] if mixed.dim() == 2 else 0
    x = torch.zeros((n, max(rows, 4608)), dtype=mixed.dtype, device=dev)[:, :4608]
    xg = torch.zeros((n, 6288), dtype=gate.dtype, device=dev)
    g = xg[:, 4608:6144].view(n, NH, D)
    args = (x, torch.zeros((n, NH, D), dtype=raw_g.dtype, device=dev), xg[:, 6272:6284], g, A_log, dt_bias, conv_weight, conv_state,
            state_cache, zero, torch.zeros(1, dtype=torch.bool, device=dev), cu, ci, co, norm_weight, norm_eps,
            torch.empty((n, NH, D), dtype=out.dtype, device=dev), lower_bound)
    kwargs = {"chunk_counts": [chunks]}
    if mode[1]:
        kwargs.update(checkpoint_state=checkpoint_state, checkpoint_offsets=zero, checkpoint_rows=None)
    return args, kwargs


prefill = entry("attention/kda_prefill_h12_d128", mode=_mode, sizes=lambda mode: (64, 2048), inputs=_inputs)(_prefill)
