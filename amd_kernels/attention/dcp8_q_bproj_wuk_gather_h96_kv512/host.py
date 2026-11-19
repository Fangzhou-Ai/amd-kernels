# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Entries and the tile tiers of the DCP8 query path."""

import torch

from amd_kernels import warm_up
from .comm import MAX_ROWS

NAME_LATENT = "attention/dcp8_q_bproj_wuk_gather_h96_kv512/latent"
NAME_Q = "attention/dcp8_q_bproj_wuk_gather_h96_kv512/q"

# (max rows, q_b tile (mt, nt, nw), W_UK tile (mt, ns)) of the latent entry; (max rows, q_b tile (mt, nt), W_UK
# workgroups per head, consumer delay) of the q entry: sweeps of benchmarks/attention/dcp8_q_bproj_wuk_gather_h96_kv512
TIERS_LATENT = (
    (16, (1, 1, 4), (1, 4)),
    (48, (1, 2, 4), (1, 4)),
    (96, (2, 2, 4), (2, 4)),
    (MAX_ROWS, (2, 4, 4), (3, 4)),
)
TIERS_Q = (
    (16, 1, 1, 2, 1, 1),
    (32, 1, 2, 2, 1, 1),
    (48, 1, 2, 2, 0, 1),
    (96, 1, 4, 2, 0, 2),
    (128, 1, 2, 2, 0, 1),
    (160, 1, 2, 2, 0, 2),
    (MAX_ROWS, 1, 4, 2, 0, 2),
)


def config_latent(rows):
    for top, qb, wuk in TIERS_LATENT:
        if rows <= top:
            return qb, wuk
    raise ValueError(rows)


def config_q(rows):
    for top, mt1, nt1, ns2, c_delay, nph in TIERS_Q:
        if rows <= top:
            return mt1, nt1, ns2, top, c_delay, nph
    raise ValueError(rows)


def _check(q_a, w_qb, w_uk, heads):
    m = q_a.shape[0]
    assert 0 < m <= MAX_ROWS and q_a.shape[1] == 1536 and q_a.dtype == torch.bfloat16
    assert q_a.stride(1) == 1 and q_a.stride(0) % 8 == 0
    assert w_qb.dtype == w_uk.dtype == torch.bfloat16 and w_qb.is_contiguous() and w_uk.is_contiguous()
    assert w_uk.numel() == heads * 128 * 512
    return m


def _mode(comm, q_a, w_qb, w_uk):
    return ()


def _sizes(mode):
    return tuple(top for top, *_ in TIERS_Q)


def _inputs(mode, n, comm, q_a, w_qb, w_uk):
    return (comm, torch.zeros((n, 1536), dtype=q_a.dtype, device=q_a.device), w_qb, w_uk), {}


@warm_up.entry(NAME_LATENT, mode=_mode, sizes=_sizes, inputs=_inputs)
def q_b_wuk_allgather_latent(comm, q_a, w_qb, w_uk):
    """Latent query [M, 96, 576] of all 96 heads: this rank's q_b + W_UK heads, pushed to every rank."""
    assert comm.mode == "latent"
    m = _check(q_a, w_qb, w_uk, 12)
    return comm._latent(q_a, w_qb, w_uk, config_latent(m))


@warm_up.entry(NAME_Q, mode=_mode, sizes=_sizes, inputs=_inputs)
def q_b_allgather_wuk_all_heads(comm, q_a, w_qb, w_uk96, out=None):
    """Same result as the latent entry: q [M, 96, 192] gathered, W_UK of all 96 heads applied locally into ``out``."""
    assert comm.mode == "q"
    m = _check(q_a, w_qb, w_uk96, 96)
    if out is None:
        out = torch.empty((m, 96, 576), dtype=torch.bfloat16, device=q_a.device)
    assert out.shape == (m, 96, 576) and out.dtype == torch.bfloat16 and out.is_contiguous()
    return comm._q(q_a, w_qb, w_uk96, out, config_q(m))
