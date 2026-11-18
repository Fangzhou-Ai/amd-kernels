# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entry, launch configuration and scratch of the MXFP4 paged index scorer + top-2048 (two launches)."""

import functools

import torch

from amd_kernels.warm_up import entry

from . import lists as _lists
from . import merge as _merge
from .geometry import CHUNK_MAX, MAX_ROWS, NCH_MAX, QMAX, TOPK, chunk_size, num_chunks, scratch_words

HEADS = 32
PAGE = 64


def cache_views(cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Index sections of a merged cache ``[pages, 64, 68]`` uint8 (page stride >= 4352 bytes) as the scorer's payload
    ``[pages, 1, 4, 64, 16]`` and scales ``[pages, 1, 4, 64]`` views, without a copy."""
    if (
        cache.dtype != torch.uint8
        or cache.ndim != 3
        or cache.shape[1:] != (PAGE, 68)
        or cache.stride(2) != 1
        or cache.stride(1) != 68
        or cache.stride(0) < 4352
    ):
        raise ValueError("the index cache must be uint8 [pages, 64, 68] sections of pages >= 4352 bytes apart")
    pages, stride, start = cache.shape[0], cache.stride(0), cache.storage_offset()
    payload = cache.as_strided((pages, 1, 4, PAGE, 16), (stride, 4096, 1024, 16, 1), start)
    scales = cache.as_strided((pages, 1, 4, PAGE), (stride, 256, 64, 1), start + 4096)
    return payload, scales


@functools.cache
def _scratch(device: torch.device):
    """Key and position planes of the chunk lists and their meta, one set a device (launches on one stream)."""
    words = scratch_words(MAX_ROWS)
    keys = torch.zeros(words[0], dtype=torch.int32, device=device)
    pos = torch.zeros(words[1], dtype=torch.int32, device=device)
    meta = torch.zeros(scratch_words(MAX_ROWS)[2], dtype=torch.int32, device=device)
    return keys, pos, meta


def tile_rows(max_query_len: int, batch: int) -> int:
    """Rows of a sequence a chunk workgroup scores: all of them, or a share of them when few sequences leave
    workgroups idle."""
    if not 1 <= max_query_len <= QMAX:
        raise ValueError(f"1 - {QMAX} query rows a sequence, got {max_query_len}")
    import os
    t = int(os.environ.get("KK_TILE", "0"))
    return min(max_query_len, t) if t else max_query_len


@functools.cache
def grid_chunks(width: int) -> int:
    """Chunks per sequence the chunk kernel's grid covers: the most any bound up to ``width`` keys gets."""
    return max(num_chunks(n, chunk_size(n)) for n in range(64, width + 1, 64))


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _chunk_exe(rows: int, uniform: bool = False, timing: bool = False):
    return _lists.compile_chunk_lists(rows, uniform, timing)


@functools.cache
def _merge_exe(uniform: bool = False, timing: bool = False):
    return _merge.compile_merge(uniform, timing)


def _score_topk(
    q_payload: torch.Tensor,
    q_scales: torch.Tensor,
    weights: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    row_ends: torch.Tensor,
    max_query_len: int,
    out_indices: torch.Tensor,
    out_scores: torch.Tensor,
    uniform: bool = False,
    dbg: torch.Tensor | None = None,
) -> None:
    rows = q_payload.shape[0]
    batch = query_start_loc.numel() - 1
    assert rows <= MAX_ROWS and batch >= 1
    assert q_payload.shape[1:] == (HEADS, 64) and q_payload.is_contiguous() and q_scales.is_contiguous()
    assert weights.dtype == torch.float32 and weights.shape == (rows, HEADS) and weights.is_contiguous()
    assert block_table.dtype == torch.int32 and block_table.stride(1) == 1 and block_table.shape[0] >= batch
    assert query_start_loc.dtype == torch.int32 and row_ends.dtype == torch.int32 and row_ends.shape == (rows,)
    assert out_indices.dtype == torch.int32 and out_indices.shape == (rows, TOPK) and out_indices.stride(1) == 1
    assert out_scores.dtype == torch.float32 and out_scores.shape == (rows, TOPK) and out_scores.stride(1) == 1
    payload, scales = cache_views(cache)
    width = block_table.shape[1] * PAGE
    if width > NCH_MAX * CHUNK_MAX:
        raise ValueError(f"at most {NCH_MAX * CHUNK_MAX} keys a sequence, got a block table of {width}")
    nch = grid_chunks(width)
    keys, pos, meta = _scratch(cache.device)
    stream = torch.cuda.current_stream()
    common = (
        q_payload.data_ptr(), q_scales.data_ptr(), payload.data_ptr(), scales.data_ptr(), block_table.data_ptr(),
        weights.data_ptr(), row_ends.data_ptr(), query_start_loc.data_ptr(),
    )
    geom = (block_table.stride(0), cache.stride(0), cache.shape[0] - 1, batch)
    tile = tile_rows(max_query_len, batch)
    tiles = -(-max_query_len // tile)
    _run_compiled(_chunk_exe(tile, uniform, dbg is not None), *common, keys.data_ptr(), pos.data_ptr(), meta.data_ptr(), *geom,
                  nch, max_query_len, dbg.data_ptr() if dbg is not None else 0, batch * nch * tiles, stream)
    _run_compiled(_merge_exe(uniform, dbg is not None), *common, keys.data_ptr(), pos.data_ptr(), meta.data_ptr(),
                  out_indices.data_ptr(), out_scores.data_ptr(), *geom, out_indices.stride(0), out_scores.stride(0),
                  max_query_len, dbg.data_ptr() + 4 * 4096 * 16 if dbg is not None else 0, rows, stream)


def _mode(*args, **kwargs):
    return ()


def _sizes(mode):
    """The rows a sequence may have: one chunk-kernel variant each (the merge kernel is one variant)."""
    return tuple(range(1, QMAX + 1))


def _inputs(mode, n, q_payload, q_scales, weights, cache, block_table, query_start_loc, row_ends, max_query_len,
            out_indices, out_scores):
    """One sequence of ``n`` rows over zeros (compile-only: nothing launches)."""
    device = q_payload.device
    z = lambda *shape, dtype: torch.zeros(*shape, dtype=dtype, device=device)
    args = (
        z(n, HEADS, 64, dtype=torch.uint8), z(n, 1, 4, 16, 4, dtype=torch.uint8), z(n, HEADS, dtype=torch.float32), cache,
        z(1, block_table.shape[1], dtype=torch.int32), torch.tensor([0, n], dtype=torch.int32, device=device),
        z(n, dtype=torch.int32), n, z(n, TOPK, dtype=torch.int32), z(n, TOPK, dtype=torch.float32),
    )
    return args, {}


mxfp4_paged_score_topk = entry("indexer/mxfp4_paged_score_topk_d128_h32_top2048", mode=_mode, sizes=_sizes, inputs=_inputs)(
    _score_topk
)
mxfp4_paged_score_topk.__doc__ = """The DSA decode indexer's scoring + per-row top-2048 in two launches. Replaces vLLM's
``score_paged`` + ``_select(logits, row_ends, selected, score_rows)`` of a decode batch (the logits never reach memory).

Args (positional; the operands of ``score_paged`` plus the outputs): ``q_payload`` uint8 [rows, 32, 64] and ``q_scales``
uint8 [rows, 1, 4, 16, 4] (the quantized query rows), ``weights`` fp32 [rows, 32] (head weights times
``weight_scale``), ``cache`` uint8 [pages, 64, 68] index pages (page stride >= 4352 B), ``block_table`` int32 [B,
width / 64], ``query_start_loc`` int32 [B + 1], ``row_ends`` int32 [rows], ``max_query_len`` (1 - 6 rows a
sequence), ``out_indices`` int32 [rows, 2048] and ``out_scores`` fp32 [rows, 2048] written in full for every row
(``-1`` / ``-inf`` past a row's live keys). Rows past ``query_start_loc[-1]`` are padding rows. CUDA-graph safe; the
scratch is one set a device, so two launches must not run concurrently."""
