# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``fp8_sparse_mla_decode`` against ``reference.py``.

Served shapes: rows 12 / 24 / 40 / 65 / 80 / 128 / 192 of 4-row requests with ~256 keys per row, the CUDA-graph padding
rows (empty slices), key counts around the 128-key tile and up to 2048, a strided q and out. Unreferenced cache slots are
NaN. Every case: against the fp32 reference, two launches bit for bit equal, a CUDA graph replayed into NaN-filled outputs.
"""

import pytest
import torch
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import (
    fp8_sparse_mla_decode,
    supports_fp8_sparse_mla_decode,
)
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048.reference import H, L, check_close, reference

from .cases import SERVED_ROWS, SM_SCALE, make_batch, require_gfx950, served_counts


def run_case(counts, seed, q_pad=0, out_pad=0):
    gen = torch.Generator().manual_seed(seed)
    q, cache, indptr, slots = make_batch(counts, gen)
    rows = len(counts)
    ref_out, ref_lse = reference(q, cache, indptr, slots, SM_SCALE)
    if q_pad:
        big = torch.zeros(rows, H, q.shape[2] + q_pad, dtype=q.dtype, device=q.device)
        big[:, :, : q.shape[2]] = q
        q = big[:, :, : q.shape[2]]
    out = None
    if out_pad:
        out = torch.full((rows, H, L + out_pad), float("nan"), dtype=torch.bfloat16, device=q.device)[:, :, :L]
    assert supports_fp8_sparse_mla_decode(q, cache, indptr, slots, out)
    o1, l1 = fp8_sparse_mla_decode(q, cache, indptr, slots, SM_SCALE, out)
    o2, l2 = fp8_sparse_mla_decode(q, cache, indptr, slots, SM_SCALE)
    torch.cuda.synchronize()
    check_close(o1.cpu(), l1.cpu(), ref_out.cpu(), ref_lse.cpu())
    assert torch.equal(o1, o2) and torch.equal(l1, l2)
    og = torch.full_like(o2, float("nan"))
    lg = torch.full_like(l2, float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout, glse = fp8_sparse_mla_decode(q, cache, indptr, slots, SM_SCALE)
    og.copy_(gout), lg.copy_(glse)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(gout, o2) and torch.equal(glse, l2)


@pytest.mark.parametrize("rows", SERVED_ROWS)
@torch.inference_mode()
def test_served_rows(rows):
    require_gfx950()
    gen = torch.Generator().manual_seed(rows)
    run_case(served_counts(rows, gen), rows)


@pytest.mark.parametrize("rows,pad", [(40, 15), (80, 15), (192, 63)])
@torch.inference_mode()
def test_padding_rows(rows, pad):
    require_gfx950()
    gen = torch.Generator().manual_seed(rows + 1)
    run_case(served_counts(rows, gen, pad=pad), rows + 1)


@pytest.mark.parametrize(
    "counts",
    [
        [1, 2, 3, 0],
        [127, 128, 129, 255],
        [256, 257, 384, 385],
        [512, 1000, 2047, 2048],
        [2048, 2048, 2048, 2048],
        [0, 0, 0, 0],
    ],
    ids=lambda c: "_".join(map(str, c)),
)
@torch.inference_mode()
def test_key_counts(counts):
    require_gfx950()
    run_case(counts, sum(counts) + 5)


@torch.inference_mode()
def test_strided_q_and_out():
    require_gfx950()
    gen = torch.Generator().manual_seed(7)
    run_case(served_counts(24, gen), 7, q_pad=64, out_pad=64)
