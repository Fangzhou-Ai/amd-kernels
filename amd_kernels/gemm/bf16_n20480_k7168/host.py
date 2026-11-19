# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the lm_head shard GEMM: the band by row count, the launch, the warm-up."""
import functools

import torch

from amd_kernels.warm_up import entry

from . import band
from .reference import K, N

MAX_M = band.MAX_ROWS


def supports(w: torch.Tensor, bias: torch.Tensor | None = None) -> bool:
    """Whether ``w`` is the [20480, 7168] bf16 contiguous shard this GEMM takes; the GEMM has no bias."""
    return bias is None and tuple(w.shape) == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous()


@functools.cache
def _kernel(rows: int, push: str | None = None):
    return band.compile_band_gemm(band.band_of(rows), push)


@functools.cache
def _dummies(device: torch.device):
    """Arguments a launch of another mode ignores (one launcher signature serves every mode)."""
    return (torch.zeros(1, dtype=torch.int64, device=device), torch.zeros(1, dtype=torch.int32, device=device),
            torch.zeros(24, dtype=torch.int64, device=device), torch.zeros(8, dtype=torch.int32, device=device))


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def _bf16_gemm(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor | None = None, *,
               row_indices: torch.Tensor | None = None, push: tuple | None = None) -> torch.Tensor | None:
    """out [M, 20480] bf16 = x [M, 7168] bf16 @ W^T, fp32 accumulate, W the row-major shard.

    1 <= M <= 256. Deterministic, no atomic output adds. CUDA-graph safe (no host sync, no allocation when out is
    given). The first call of each mode (contiguous / push) compiles its variants (``amd_kernels.warm_up``).

    push = (RowAllToAll.send_args(buf) of all_to_all_bf16_n20480_ranks8, counts), with row_indices (contiguous device
    int64 [M], each index in [0, len(x)), duplicates valid, re-read on every graph replay; no out): the rows
    x[row_indices] are written straight into the receivers' buffers and the exchange is published; the caller then
    runs RowAllToAll.wait. counts = CUDA int32 [8] (the same on every rank, summing to M): rows sorted by owner, the
    rows of owner r land in rank r's buffer; counts = None: every rank receives all M rows (all-gather). Returns None.
    The same bits as the contiguous call on x[row_indices].
    """
    assert (row_indices is None) == (push is None), "row_indices and push go together"
    assert x.ndim == 2 and x.shape[1] == K and x.dtype == torch.bfloat16 and x.is_contiguous()
    assert w.shape == (N, K) and w.dtype == torch.bfloat16 and w.is_contiguous() and w.device == x.device
    stream = torch.cuda.current_stream()
    idx, counter, peers, counts = _dummies(x.device)
    if push is not None:
        (peers, counter, flag, rank, buf), counts_arg = push
        m = row_indices.numel()
        assert out is None and 1 <= m <= MAX_M
        assert counter.dtype == torch.int32 and counter.device == x.device and peers.dtype == torch.int64
        assert row_indices.ndim == 1 and row_indices.dtype == torch.int64 and row_indices.is_contiguous()
        assert row_indices.device == x.device and x.numel() * x.element_size() <= 0xFFFFFFFF
        if counts_arg is not None:
            assert counts_arg.shape == (8,) and counts_arg.dtype == torch.int32 and counts_arg.is_contiguous()
            counts = counts_arg
        _run(_kernel(band.band_of(m)[0], "owner" if counts_arg is not None else "all"), x.data_ptr(), w.data_ptr(), 0,
             row_indices, x.shape[0], m, counter, flag, peers, counts, rank, buf, stream)
        return None
    m = x.shape[0]
    assert 1 <= m <= MAX_M
    if out is None:
        out = torch.empty(m, N, dtype=torch.bfloat16, device=x.device)
    assert out.shape == (m, N) and out.dtype == torch.bfloat16 and out.is_contiguous() and out.device == x.device
    _run(_kernel(band.band_of(m)[0]), x.data_ptr(), w.data_ptr(), out.data_ptr(), idx, m, m, counter, 0, peers, counts,
         0, 0, stream)
    return out


def _mode(x, w, out=None, *, row_indices=None, push=None):
    return ("contiguous" if push is None else ("all" if push[1] is None else "owner"),)


_KEEP: list = []  # the private exchange state of the warm-up call being made


def _inputs(mode, n, x, w, out=None, *, row_indices=None, push=None):
    """One warm-up call at ``n`` rows (zeros). A variant compiled before the warm-up launches even in compile-only
    mode, so a push call gets a private receive buffer, flag block and counter, never the caller's exchange."""
    if mode[0] == "contiguous":
        return (torch.zeros(n, K, dtype=torch.bfloat16, device=x.device), w), {}
    dev = x.device
    recv = torch.zeros(band.RECV_ROWS * band.V, dtype=torch.bfloat16, device=dev)
    flag = torch.zeros(128, dtype=torch.int32, device=dev)
    peers = torch.tensor([recv.data_ptr()] * 16 + [flag.data_ptr()] * 8, dtype=torch.int64, device=dev)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    counts = torch.zeros(8, dtype=torch.int32, device=dev)
    counts[0] = n
    _KEEP[:] = [recv, flag, peers, counter, counts]
    return (x, w), {"row_indices": torch.zeros(n, dtype=torch.int64, device=dev),
                    "push": ((peers, counter, flag.data_ptr(), 0, 0), counts if mode[0] == "owner" else None)}


# one row count per band
WARMUP_ROWS = [row[0] for row in band.BANDS]

bf16_gemm_n20480_k7168 = entry("gemm/bf16_n20480_k7168", mode=_mode, sizes=lambda mode: WARMUP_ROWS,
                               inputs=_inputs)(_bf16_gemm)
