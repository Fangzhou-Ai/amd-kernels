# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry point of the prefill indexer's top-2048 select: workspace and the three launches (kernel.py, contract in reference.py)."""

import functools

import torch

from amd_kernels.common.tensor_shim import _run_compiled
from amd_kernels.warm_up import entry

from . import kernel as _kernel

TOPK = 2048
CAP = _kernel.CAP
_workspaces: dict = {}


@functools.cache
def _launchers():
    return _kernel.compile_threshold(), _kernel.compile_stream(), _kernel.compile_select()


def workspace_bytes(rows: int, width: int) -> int:
    """Scratch of one call: per-row threshold / count words and the candidate lists (8 bytes per entry)."""
    return 4 * (5 * rows + 2 * rows * min(CAP, width))


def _workspace(device: torch.device, rows: int, width: int) -> torch.Tensor:
    need = workspace_bytes(rows, width) // 4
    buf = _workspaces.get(device)
    if buf is None or buf.numel() < need:
        buf = torch.empty(need, dtype=torch.int32, device=device)
        _workspaces[device] = buf
    return buf


def _mode(*args, **kwargs):
    return ()


def _sizes(mode):
    return (1,)


def _warm_inputs(mode, n, logits, ks, ke, out):
    dev = logits.device
    return (
        (
            torch.zeros(n, 4096, dtype=torch.float32, device=dev),
            torch.zeros(n, dtype=torch.int32, device=dev),
            torch.full((n,), 4096, dtype=torch.int32, device=dev),
            torch.empty(n, TOPK, dtype=torch.int32, device=dev),
        ),
        {},
    )


@entry("indexer.mxfp4_prefill_topk_d128_h32_top2048", mode=_mode, sizes=_sizes, inputs=_warm_inputs)
def topk_rows(logits: torch.Tensor, ks: torch.Tensor, ke: torch.Tensor, out: torch.Tensor, order: torch.Tensor | None = None) -> torch.Tensor:
    """Write the window-relative positions of every row's 2048 largest logits of ``[ks, ke)`` into ``out`` (int32 [rows, 2048]),
    ascending, ties to the smaller position, -1 filling rows of fewer than 2048 keys. ``order`` (int32 [rows], optional) is the
    order in which the streaming launch takes the rows, longest window first (``argsort(ke - ks, descending=True)``): it only
    balances a chunk of requests of different lengths and never changes the result."""
    rows, width = logits.shape
    assert logits.dtype == torch.float32 and logits.stride(1) == 1 and logits.stride(0) % 4 == 0
    assert logits.data_ptr() % 16 == 0
    assert ks.dtype == ke.dtype == torch.int32 and ks.is_contiguous() and ke.is_contiguous()
    assert ks.shape == ke.shape == (rows,)
    assert out.dtype == torch.int32 and out.shape == (rows, TOPK) and out.stride(1) == 1
    assert order is None or (order.dtype == torch.int32 and order.shape == (rows,) and order.is_contiguous())
    if rows == 0:
        return out
    cstride = min(CAP, width)
    ws = _workspace(logits.device, rows, width)
    meta, cnt, cand = ws[: 4 * rows], ws[4 * rows : 5 * rows], ws[5 * rows :]
    threshold, stream, select = _launchers()
    s = torch.cuda.current_stream()
    stride = logits.stride(0)
    _run_compiled(threshold, logits.data_ptr(), ks.data_ptr(), ke.data_ptr(), meta.data_ptr(), stride, rows, s)
    _run_compiled(stream, logits.data_ptr(), ks.data_ptr(), ke.data_ptr(), meta.data_ptr(), cnt.data_ptr(), cand.data_ptr(), stride, cstride, 0 if order is None else order.data_ptr(), rows, s)
    _run_compiled(select, logits.data_ptr(), ks.data_ptr(), ke.data_ptr(), cnt.data_ptr(), cand.data_ptr(), out.data_ptr(), stride, cstride, out.stride(0), rows, s)
    return out
