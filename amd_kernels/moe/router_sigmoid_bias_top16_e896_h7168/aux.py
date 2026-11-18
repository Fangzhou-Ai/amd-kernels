# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The BM=16 MoE sort of the routed experts, emitted by the router launch (AITER's auxiliary sort layout, the one
``register_presorted_moe`` takes: sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, m_indices,
reverse_sorted) and the zeroed [M, 3584] bf16 output buffer the atomic stage 2 accumulates into.

How it is made without a global sort pass. Every wave that finished a token ORs the token's bit into one 256-bit mask
per routed expert (an expert's tokens, scratch ``mask``), writes its token's flag, and waits for every token's flag.
Then each wave, from the masks alone, derives the same layout: lane l counts the tokens of experts 14 l .. 14 l + 13,
rounds each count up to 16 and a DPP scan over the lanes gives every expert's first row; a pair's row is its expert's
first row plus the number of lower tokens in the expert's mask (ascending token order inside an expert, the same bits
on every run). Each wave writes its token's 16 rows; the padding rows (token id M, weight 0, m_indices M), the zeroed
masks and the moe buffer are written by every workgroup before it arrives, so a valid row is never overwritten
by a pad. The wave of token 0 also writes the expert id of every 16-row block and ``num_valid_ids`` (padded rows, M).
"""
from types import SimpleNamespace

import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .topk import _dpp, _scan

BM = 16
MASK_WORDS = 8  # 32-bit words of one expert's token mask (M <= 256)
NE = 896
EPL = 14  # experts per lane
FLAG_BASE = NE * MASK_WORDS  # the token flags follow the masks (int32 words in the workspace)
WORKSPACE_WORDS = FLAG_BASE + 256
SC1 = 0x10
MAX_ZERO_VECS = 192 * 3584 * 2 // 16  # 16-byte vectors of the largest moe buffer, [192, 3584] bf16


def aux_capacity(m: int, topk: int = 16, num_experts: int = NE) -> int:
    """Rows of the sorted arrays: every route plus BM - 1 padding rows for each expert that can be active."""
    routes = m * topk
    active = min(num_experts, routes)
    return (routes + active * (BM - 1) + BM - 1) // BM * BM


def atomic_or(addr, value):
    """Device-scope ``atomicrmw or`` on a global int32 (no result used: ``global_atomic_or`` without return)."""
    ptr = llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), fx.as_ir_value(fx.Int64(addr)))
    llvm.AtomicRMWOp(llvm.AtomicBinOp._or, ptr, fx.as_ir_value(fx.Int32(value)), llvm.AtomicOrdering.monotonic,
                     syncscope="agent", alignment=4)


def aux_prefill(gt, nthreads, i32_m, i32_cap, i32_zero_vecs, mask_res, sti_res, sw_res, mi_res, mb_res):
    """Thread ``gt`` of ``nthreads`` (the whole grid, at least ``16512`` threads): its share of the zeroed expert masks,
    of the padding rows of the three row arrays (token id M, weight 0, m_indices M) and of the zeroed moe buffer
    (``i32_zero_vecs`` 16-byte vectors; 0: none). Stores of the masks and rows are write-through (sc1): a consumer
    wave's valid row, written later from another XCD, can never be overtaken by a stale pad line."""
    zero = fx.Int32(0)
    buffer_ops.buffer_store(zero, mask_res, gt, mask=gt < NE * MASK_WORDS, cache_modifier=SC1)
    ok = gt < i32_cap
    buffer_ops.buffer_store(i32_m, sti_res, gt, mask=ok, cache_modifier=SC1)
    buffer_ops.buffer_store(i32_m, mi_res, gt, mask=ok, cache_modifier=SC1)
    buffer_ops.buffer_store(zero, sw_res, gt, mask=ok, cache_modifier=SC1)
    zeros = Vec.from_elements([zero, zero, zero, zero], fx.Int32)
    for k in range_constexpr((MAX_ZERO_VECS + nthreads - 1) // nthreads):
        v = gt + k * nthreads
        buffer_ops.buffer_store(zeros, mb_res, v * 4, mask=v < i32_zero_vecs)


def aux_counts(mask_res, lane):
    """Lane ``lane``'s experts 14 lane .. 14 lane + 13: token counts, padded counts and first rows (scan over the lanes),
    from the finished masks. Also ``total``: the padded rows of all experts."""
    words = [Vec(buffer_ops.buffer_load(mask_res, EPL * lane * MASK_WORDS + 4 * i, vec_width=4, dtype=fx.Int32,
                                        cache_modifier=SC1 | 1)) for i in range_constexpr(EPL * MASK_WORDS // 4)]
    count, pad = [], []
    for e in range_constexpr(EPL):
        c = fx.Int32(fx.math.ctpop(fx.Int32(words[2 * e][0])))
        for j in range_constexpr(1, 8):
            c = c + fx.Int32(fx.math.ctpop(fx.Int32(words[2 * e + j // 4][j % 4])))
        count.append(c)
        pad.append((c + (BM - 1)) & fx.Int32(-BM))
    lane_rows = pad[0]
    for e in range_constexpr(1, EPL):
        lane_rows = lane_rows + pad[e]
    incl = _scan(lane_rows)
    run = incl - lane_rows
    cum = []
    for e in range_constexpr(EPL):
        cum.append(run)
        run = run + pad[e]
    return SimpleNamespace(count=count, pad=pad, cum=cum,
                           total=fx.Int32(fx.rocdl.readlane(T.i32, incl, fx.Int32(63))))


def _lds_ptr(byte_addr):
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<3>"), fx.as_ir_value(fx.Int64(byte_addr)))


def aux_publish_cum(st, lds_base, wv, lane):
    """Every expert's first row to LDS (the dead GEMM ring at ``lds_base``: 3584 bytes a wave), for the lanes that look
    an expert up by id."""
    for e in range_constexpr(EPL):
        llvm.store(fx.as_ir_value(st.cum[e]), _lds_ptr(lds_base + (wv * NE + EPL * lane + e) * 4), alignment=4)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.rocdl.sched_barrier(0)


def aux_pair_row(lds_base, wv, mask_res, expert, tok):
    """Lane r's pair (token ``tok``, slot r, routed to ``expert``): its row, the expert's first row (an LDS read) plus the
    number of lower tokens in the expert's mask."""
    first = fx.Int32(llvm.load(T.i32, _lds_ptr(lds_base + (wv * NE + expert) * 4), alignment=4))
    w0 = Vec(buffer_ops.buffer_load(mask_res, expert * MASK_WORDS, vec_width=4, dtype=fx.Int32, cache_modifier=SC1 | 1))
    w1 = Vec(buffer_ops.buffer_load(mask_res, expert * MASK_WORDS + 4, vec_width=4, dtype=fx.Int32,
                                    cache_modifier=SC1 | 1))
    tw = tok >> 5
    low = (fx.Int32(1) << (tok & 31)) - fx.Int32(1)
    rank = fx.Int32(0)
    for w in range_constexpr(MASK_WORDS):
        word = fx.Int32((w0 if w < 4 else w1)[w % 4])
        below = (tw > w).select(word, (tw == w).select(word & low, fx.Int32(0)))
        rank = rank + fx.Int32(fx.math.ctpop(below))
    return first + rank


def aux_max_blocks(st):
    """The most 16-row blocks of one expert over the wave's experts (uniform)."""
    top = st.pad[0] >> 4
    for e in range_constexpr(1, EPL):
        top = (st.pad[e] >> 4 > top).select(st.pad[e] >> 4, top)
    for ctrl in (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143):
        r = fx.Int32(_dpp(top, ctrl))
        top = (r > top).select(r, top)
    return fx.Int32(fx.rocdl.readlane(T.i32, top, fx.Int32(63)))


def aux_expert_blocks(st, b, lane, sei_res):
    """Block ``b`` of each of the lane's experts: ``sorted_expert_ids[first block + b] = expert``."""
    for e in range_constexpr(EPL):
        buffer_ops.buffer_store(EPL * lane + e, sei_res, (st.cum[e] >> 4) + b, mask=b < (st.pad[e] >> 4),
                                cache_modifier=SC1)
