# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The whole prefill (conv + prologue, scan, gated norm) against reference.kda_prefill_reference."""

import pytest
import torch
from amd_kernels.attention.kda_prefill_h12_d128 import host
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import make_case, require_gfx950

OUT_TOL = 8e-3
STATE_TOL = 5e-3


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def chunk_indices(cu):
    rows = [(n, c) for n, c, _, _ in R.chunk_table(cu)[0]]
    return torch.tensor(rows, dtype=torch.int32, device=cu.device)


def offsets(cu):
    return torch.tensor(R.chunk_table(cu)[1], dtype=torch.int64, device=cu.device)


def run(c, check=None):
    cu = c["cu_seqlens"]
    t = int(cu[-1])
    out = torch.zeros(t, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
    conv_state, state = c["conv_state"].clone(), c["state_cache"].clone()
    host.prefill(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], conv_state, state,
                 c["state_indices"], c["has_initial_state"], cu, chunk_indices(cu), offsets(cu), c["norm_weight"], c["norm_eps"], out,
                 c["lower_bound"])
    return out, conv_state, state


@pytest.mark.parametrize("seq,init,dt_shift", [
    ([64], [True], 0.0), ([130], [False], 0.0), ([2048], [True], 0.0), ([300, 64, 77, 1], [True, False, True, True], 0.0),
    ([5, 3], [False, True], 0.0), ([1500, 700, 64, 3000], [True, False, True, True], -6.0),
], ids=["64", "130_noinit", "2048", "ragged4", "tiny", "ragged_slow"])
@torch.inference_mode()
def test_prefill(seq, init, dt_shift):
    require_gfx950()
    c = make_case(seq, init=init, dt_shift=dt_shift)
    conv_state, state = c["conv_state"].clone(), c["state_cache"].clone()
    ref = R.kda_prefill_reference(c["mixed"], c["raw_g"], c["beta_logit"], c["gate"], c["A_log"], c["dt_bias"], c["lower_bound"],
                                  c["conv_weight"], conv_state, state, c["state_indices"], c["has_initial_state"], c["cu_seqlens"],
                                  c["norm_weight"], c["norm_eps"])
    out, got_conv, got_state = run(c)
    assert rel(out, ref) < OUT_TOL
    assert rel(got_state, state) < STATE_TOL
    assert torch.equal(got_conv, conv_state)
    again, _, _ = run(c)
    assert torch.equal(out, again)


@torch.inference_mode()
def test_state_continuation():
    """Two prefill calls over the halves of a sequence (the second with the state and conv history the first left) equal one call."""
    require_gfx950()
    first, second = 640, 384
    c = make_case([first + second], init=[False])
    whole_out, whole_conv, whole_state = run(c)
    outs, conv_state, state = [], c["conv_state"].clone(), c["state_cache"].clone()
    for lo, hi, init in ((0, first, False), (first, first + second, True)):
        sub = dict(c)
        sub["mixed"], sub["raw_g"] = c["mixed"][lo:hi], c["raw_g"][lo:hi]
        sub["beta_logit"], sub["gate"] = c["beta_logit"][lo:hi], c["gate"][lo:hi]
        sub["cu_seqlens"] = torch.tensor([0, hi - lo], dtype=torch.int32, device="cuda")
        sub["has_initial_state"] = torch.tensor([init], device="cuda")
        sub["conv_state"], sub["state_cache"] = conv_state, state
        out = torch.zeros(hi - lo, R.NH, R.D, dtype=torch.bfloat16, device="cuda")
        cu = sub["cu_seqlens"]
        host.prefill(sub["mixed"], sub["raw_g"], sub["beta_logit"], sub["gate"], c["A_log"], c["dt_bias"], c["conv_weight"], conv_state,
                     state, c["state_indices"], sub["has_initial_state"], cu, chunk_indices(cu), offsets(cu), c["norm_weight"],
                     c["norm_eps"], out, c["lower_bound"])
        outs.append(out)
    assert torch.equal(torch.cat(outs), whole_out)
    assert torch.equal(state, whole_state) and torch.equal(conv_state, whole_conv)
