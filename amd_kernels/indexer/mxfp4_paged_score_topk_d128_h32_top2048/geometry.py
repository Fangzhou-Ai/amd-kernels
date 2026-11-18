# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Shapes shared by the chunk kernel, the merge kernel and the host: chunk size, list sizes and the scratch layout.

A sequence's keys are cut into ``nch = ceil(longest row bound / cb)`` chunks of ``cb`` keys, ``cb`` a multiple of 512 in
``[512, CHUNK_MAX]`` chosen from the bound so that a sequence has about ``CHUNKS_TARGET`` chunks (a long context is
scored by as many workgroups as a short one, up to the chunk size's cap). A row keeps from chunk ``c`` its ``tb`` largest
keys (all of them if it has fewer) and from the sequence's last chunk, where the newest keys are, all of its keys, so a row's pool is
``(nch - 1) * tb + cb`` entries, at most ``POOL``. A row's pool is ``keys`` and ``positions`` int32 planes of ``POOL`` entries;
``meta`` holds per (row, chunk) the list's smallest key and the chunk's live key count.
"""

import flydsl.expr as fx
from flydsl.expr.typing import Int32 as I

TOPK = 2048
CHUNK_MAX = 2048
CHUNK_STEP = 512
import os
CHUNKS_TARGET = int(os.environ.get('KK_TARGET', '40'))
NCH_MAX = 64  # CHUNK_MAX chunks cover 131072 keys, 1 Mi tokens at DCP8
POOL = 20480
QMAX = 6  # query rows a sequence may have
MAX_ROWS = 256  # rows one call may hold
INFO_OFFSET = MAX_ROWS * NCH_MAX * 16  # bytes: the per-row records follow the meta plane
OVF_KEYS = 7 * 2048  # keys of chunks the merge scores again
POOL_TARGET = 12288  # entries a pool is cut to where the exactness bound below allows


def chunk_size(max_end: int) -> int:
    per = -(-max_end // CHUNKS_TARGET)
    cb = -(-per // CHUNK_STEP) * CHUNK_STEP
    return min(max(cb, CHUNK_STEP), CHUNK_MAX)


def num_chunks(max_end: int, cb: int) -> int:
    return max(1, -(-max_end // cb))


def list_size(nch: int, cb: int) -> int:
    """Keys a chunk list keeps: a quarter of the chunk, cut to the pool target, but at least ``floor`` so that the
    chunks which can hold selected keys beyond their list (at most 2047 // list size) fit the overflow."""
    slots = OVF_KEYS // cb
    floor = -(-(2047 // (slots + 1) + 1) // 32) * 32
    pool = (POOL_TARGET - cb) // max(nch - 1, 1) * 8 // 9 // 32 * 32 if nch > 1 else cb
    return min(max(floor, min(max(cb // 4 // 32 * 32, 32), pool)), cb)


def chunk_size_rt(max_end):
    """``chunk_size`` on a kernel value."""
    cb = (((max_end + (CHUNKS_TARGET - 1)) // CHUNKS_TARGET + (CHUNK_STEP - 1)) // CHUNK_STEP) * CHUNK_STEP
    return fx.min(fx.max(cb, I(CHUNK_STEP)), I(CHUNK_MAX))


def list_size_rt(nch, cb):
    """``list_size`` on kernel values."""
    slots = I(OVF_KEYS) // cb
    floor = ((I(2047) // (slots + 1) + 1 + 31) // 32) * 32
    pool = ((I(POOL_TARGET) - cb) // fx.max(nch - 1, I(1)) * 8 // 9 // 32) * 32
    pool = (nch > 1).select(pool, cb)
    return fx.min(fx.max(floor, fx.min(fx.max(cb // 4 // 32 * 32, I(32)), pool)), cb)


def slot_size(tb):
    """Entries a chunk's list may take: its ``tb`` plus the keys that share the threshold's histogram bin."""
    return tb + tb // 8


def scratch_words(max_rows: int) -> tuple[int, int, int]:
    """Words of the key plane, the position plane and the meta plane for ``max_rows`` rows."""
    return max_rows * POOL, max_rows * POOL, max_rows * NCH_MAX * 4 + MAX_ROWS * 4
