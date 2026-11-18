# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The prologue's workspace records and conv_state against reference.py (ragged batches, no-initial-state sequences)."""

import pytest
import torch
from amd_kernels.attention.kda_prefill_h12_d128 import prologue as P
from amd_kernels.attention.kda_prefill_h12_d128 import reference as R

from .inputs import make_case, require_gfx950

CASES = [
    ("one_chunk", [64], None),
    ("tail", [200], None),
    ("ragged", [300, 64, 130, 5, 2048], None),
    ("tiny", [1, 2, 3, 4, 65], [True, False, True, False, True]),
    ("noinit", [130, 70], [False, False]),
]
TOL = dict(qg=6e-3, nw=6e-3, kgt=6e-3, aqk=6e-3, u=6e-3, decay=1e-5)


def rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-30)).item()


@pytest.mark.parametrize("name,seq,init", CASES, ids=[c[0] for c in CASES])
@torch.inference_mode()
def test_prologue(name, seq, init):
    require_gfx950()
    c = make_case(seq, init=init)
    cu = c["cu_seqlens"]
    table, _ = R.chunk_table(cu)
    ci = torch.tensor([[n, k] for (n, k, _, _) in table], dtype=torch.int32, device="cuda")
    conv, new_cs = R.conv_reference(c["mixed"], c["conv_weight"], c["conv_state"], c["state_indices"], c["has_initial_state"], cu)
    pro = R.prologue_reference(conv[:, :R.LP], conv[:, R.LP:2 * R.LP], conv[:, 2 * R.LP:], c["raw_g"], c["beta_logit"], c["A_log"],
                               c["dt_bias"], c["lower_bound"], R.SCALE, cu)
    want = R.unpack_workspace(R.pack_workspace(pro))
    cs = c["conv_state"].clone()
    ws = P.prologue(c["mixed"], c["raw_g"], c["beta_logit"], c["A_log"], c["dt_bias"], c["lower_bound"], R.SCALE, c["conv_weight"], cs,
                    c["state_indices"], c["has_initial_state"], cu, ci)
    got = R.unpack_record(ws)
    for key, tol in TOL.items():
        assert not torch.isnan(got[key]).any(), key
        assert rel(got[key], want[key]) <= tol, (key, rel(got[key], want[key]))
    for n in range(len(seq)):
        slot = int(c["state_indices"][n])
        assert torch.equal(cs[slot], new_cs[n]), f"conv_state of sequence {n}"
    untouched = torch.ones(cs.shape[0], dtype=torch.bool)
    untouched[c["state_indices"].long().cpu()] = False
    assert torch.equal(cs[untouched.cuda()], c["conv_state"][untouched.cuda()])
