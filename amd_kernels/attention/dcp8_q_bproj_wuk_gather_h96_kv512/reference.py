# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Gathered latent query of the DCP8 DSA decode path.

Inputs: ``q_a`` bf16 [M, 1536]; per rank r in 0..7 the q_b weight ``w_qb[r]`` bf16 [2304, 1536] (12 heads x (128 nope
+ 64 rope-width pass-through) rows, head-major) and ``w_uk[r]`` bf16 [12, 128, 512] (W_UK_T of its 12 heads).
Output: bf16 [M, 96, 576], head 12 r + h from rank r: ``[bf16(q_nope @ w_uk) | q_pe]`` with
``q = bf16(q_a @ w_qb.T)`` and ``q_nope``, ``q_pe`` its first 128 and last 64 columns per head. Both matmuls
accumulate in fp32 and round once to bf16, as the unfused vLLM path (hipBLASLt GEMM, then ``torch.bmm``, then
concat, then the all-gather) does. A kernel sums K in a different order, so a q element may differ by one bf16 ulp
and a latent element then follows: the tests require per-row cosine >= 0.9999 and relative L2 <= 4e-3 against this
function, bitwise repeatability of two runs, and exact equality of the q_pe columns.
"""

import torch

HEADS_PER_RANK = 12
RANKS = 8


def q_b_reference(q_a, w_qb):
    return (q_a.float() @ w_qb.float().T).bfloat16().view(q_a.shape[0], HEADS_PER_RANK, 192)


def latent_reference(q, w_uk):
    nope = q[..., :128].transpose(0, 1).float()
    lat = torch.bmm(nope, w_uk.float()).bfloat16().transpose(0, 1)
    return torch.cat((lat, q[..., 128:]), dim=-1)


def gathered_query_reference(q_a, w_qb, w_uk):
    """``w_qb``, ``w_uk``: the 8 ranks' weights (sequences); returns bf16 [M, 96, 576]."""
    return torch.cat([latent_reference(q_b_reference(q_a, w_qb[r]), w_uk[r]) for r in range(RANKS)], dim=1)
