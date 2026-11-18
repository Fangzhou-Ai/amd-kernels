# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the MoE router: the load-time weight layout, K split by rows, the launches, the warm-up."""
import functools
import struct

import torch

from amd_kernels.warm_up import entry

from .kernel import MT_MAX, compile_router_partials_e896_h7168, partial_bytes
from .reference import E, H, TOPK
from .topk import compile_router_sigmoid_bias_top16_e896

MAX_M = 16 * MT_MAX
# row tile (16 rows) -> K splits (graph replay sweep over 4-28 at M 12-192: 14 is within noise of the best everywhere)
KS_BY_MT = {mt: 14 for mt in range(1, MT_MAX + 1)}


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [896, 7168] bf16 contiguous router weight (either layout) and ``bias`` the fp32 [896]
    correction bias (or None)."""
    ok = tuple(w.shape) == (E, H) and w.dtype == torch.bfloat16 and w.is_contiguous()
    return ok and (bias is None or (tuple(bias.shape) == (E,) and bias.dtype == torch.float32))


def shuffle_weight(w: torch.Tensor) -> torch.Tensor:
    """The load-time layout [56 groups, 112 K blocks, 16 rows, 64 K] of the router weight, as [896, 7168] (bit-exact)."""
    assert w.shape == (E, H) and w.dtype == torch.bfloat16
    return w.view(E // 16, 16, H // 64, 64).permute(0, 2, 1, 3).contiguous().view(E, H)


def shuffle_weight_(w: torch.Tensor) -> torch.Tensor:
    """shuffle_weight in place: ``w`` keeps its storage and holds the load-time layout."""
    ws = shuffle_weight(w)
    w.copy_(ws)
    del ws
    return w


def unshuffle_weight(ws: torch.Tensor) -> torch.Tensor:
    assert ws.shape == (E, H) and ws.dtype == torch.bfloat16
    return ws.view(E // 16, H // 64, 16, 64).permute(0, 2, 1, 3).contiguous().view(E, H)


def m_tile_fragments(m: int) -> int:
    return (m + 15) // 16


@functools.cache
def _partials_kernel(mt: int, ks: int, shuf: bool):
    return compile_router_partials_e896_h7168(mt, ks, int(shuf))


@functools.cache
def _topk_kernel(ks: int):
    return compile_router_sigmoid_bias_top16_e896(ks)


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


_WORK: dict = {}


def _workspace(device: torch.device, stream: int):
    """The partial slabs of one (device, stream)."""
    key = (device.index, stream)
    ws = _WORK.get(key)
    if ws is None:
        ws = torch.empty(max(partial_bytes(MAX_M, ks) for ks in KS_BY_MT.values()), dtype=torch.uint8, device=device)
        _WORK[key] = ws
    return ws


def _bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


def _partials(x: torch.Tensor, w: torch.Tensor, shuffled: bool):
    m = x.shape[0]
    assert x.ndim == 2 and x.shape[1] == H and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert supports(w) and x.device == w.device and 1 <= m <= MAX_M
    stream = torch.cuda.current_stream()
    mt = m_tile_fragments(m)
    ks = KS_BY_MT[mt]
    part = _workspace(x.device, stream.cuda_stream)
    _run(_partials_kernel(mt, ks, bool(shuffled)), x.data_ptr(), w.data_ptr(), part.data_ptr(), m, stream)
    return part, ks


def _top16(part: torch.Tensor, ks: int, bias: torch.Tensor, routed_scaling_factor: float, out: tuple, m: int):
    weights, ids = out
    assert weights.stride(0) == ids.stride(0) and weights.stride(1) == ids.stride(1) == 1
    _run(_topk_kernel(ks), part.data_ptr(), bias.data_ptr(), weights.data_ptr(), ids.data_ptr(), m,
         weights.stride(0), _bits(routed_scaling_factor), torch.cuda.current_stream())


def router_logits(x: torch.Tensor, w: torch.Tensor, *, shuffled: bool = False) -> torch.Tensor:
    """fp32 logits [M, 896] = the partial slabs of the GEMM launch summed in slice order (what the top-k launch
    computes); for tests and benchmarks, serving never materialises them."""
    m = x.shape[0]
    if m > MAX_M:
        return torch.cat([router_logits(x[i:i + MAX_M], w, shuffled=shuffled) for i in range(0, m, MAX_M)])
    part, ks = _partials(x, w, shuffled)
    slabs = part[:partial_bytes(m, ks)].view(torch.float32).view(ks, m, E)
    logits = slabs[0].clone()
    for k in range(1, ks):
        logits += slabs[k]
    return logits


def top16(logits: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float = 1.0,
          out: tuple | None = None):
    """(weights fp32 [M, 16], ids int32 [M, 16]) from fp32 logits [M, 896]: sigmoid + bias + top-16 + renormalise."""
    m = logits.shape[0]
    assert logits.dtype == torch.float32 and logits.shape[1] == E and logits.is_contiguous()
    assert bias.dtype == torch.float32 and bias.shape == (E,) and bias.is_contiguous()
    if out is None:
        out = (torch.empty(m, TOPK, dtype=torch.float32, device=logits.device),
               torch.empty(m, TOPK, dtype=torch.int32, device=logits.device))
    if m:
        _top16(logits, 1, bias, routed_scaling_factor, out, m)
    return out


def _router(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, routed_scaling_factor: float = 1.0,
            out: tuple | None = None, *, shuffled: bool = False):
    """(topk_weights fp32 [M, 16], topk_ids int32 [M, 16]) of the router: x [M, 7168] @ W^T, sigmoid, + bias, top-16,
    renormalised, times ``routed_scaling_factor``. Slots in ascending expert id; equal biased scores: lowest id first.
    W = w (row-major [896, 7168]) or shuffle_weight's layout. Any M (above 192 rows: launches of 192 rows).
    Deterministic, no atomics, CUDA-graph safe. One call at a time per stream (the workspace)."""
    m = x.shape[0]
    assert bias.dtype == torch.float32 and bias.shape == (E,) and bias.is_contiguous()
    if out is None:
        out = (torch.empty(m, TOPK, dtype=torch.float32, device=x.device),
               torch.empty(m, TOPK, dtype=torch.int32, device=x.device))
    if m > MAX_M:
        for i in range(0, m, MAX_M):
            _router(x[i:i + MAX_M], w, bias, routed_scaling_factor, (out[0][i:i + MAX_M], out[1][i:i + MAX_M]),
                    shuffled=shuffled)
        return out
    if m:
        part, ks = _partials(x, w, shuffled)
        _top16(part, ks, bias, routed_scaling_factor, out, m)
    return out


def _mode(x, w, bias, routed_scaling_factor=1.0, out=None, *, shuffled=False):
    """What a call fixes about its kernels besides M (``amd_kernels.warm_up``): the weight layout."""
    return (bool(shuffled),)


def _inputs(mode, n, x, w, bias, routed_scaling_factor=1.0, out=None, *, shuffled=False):
    return (torch.zeros(n, H, dtype=torch.bfloat16, device=x.device), w, bias, routed_scaling_factor), {
        "shuffled": shuffled}


router_sigmoid_bias_top16_e896_h7168 = entry(
    "moe/router_sigmoid_bias_top16_e896_h7168", mode=_mode, sizes=lambda mode: [16 * mt for mt in range(1, MT_MAX + 1)],
    inputs=_inputs)(_router)
