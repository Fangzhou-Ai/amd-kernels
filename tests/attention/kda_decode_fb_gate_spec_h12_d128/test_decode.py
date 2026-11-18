# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""``kda_decode_fb_gate_spec`` against ``reference.py`` at the served token counts (12 / 24 / 40 / 65 / 80 / 128 / 192
rows of 6-token sequences, the last one shorter when the count is not a multiple), padding sequences, a transposed
conv state, a fractional slot pitch and both norm-weight dtypes; two launches match bit for bit."""

import pytest
import torch
from amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128 import kda_decode_fb_gate_spec
from amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128.reference import kda_decode_fb_gate_spec_reference

from .inputs import clone_inputs, make_inputs

SERVED = [12, 24, 40, 65, 80, 128, 192]


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("needs a gfx950 GPU")


def touched(a):
    return a["state_indices"].flatten().unique()


def check(a, b):
    torch.testing.assert_close(a["out"].float(), b["out"].float(), atol=2e-2, rtol=2e-2)
    cos = torch.nn.functional.cosine_similarity(a["out"].float().flatten(), b["out"].float().flatten(), dim=0)
    assert cos > 0.9999, cos
    sl = touched(a).long()
    torch.testing.assert_close(a["state"][sl], b["state"][sl], atol=2e-3, rtol=1e-3)
    assert torch.equal(a["conv_state"], b["conv_state"])


@pytest.mark.parametrize("rows", SERVED)
@torch.inference_mode()
def test_served_rows(rows):
    require_gfx950()
    a = make_inputs(rows, seed=rows)
    b = clone_inputs(a)
    kda_decode_fb_gate_spec_reference(**b)
    kda_decode_fb_gate_spec(**a)
    torch.cuda.synchronize()
    check(a, b)
    c = clone_inputs(make_inputs(rows, seed=rows))
    kda_decode_fb_gate_spec(**c)
    assert torch.equal(c["out"], a["out"]) and torch.equal(c["state"], a["state"])


@pytest.mark.parametrize(
    "kw",
    [
        dict(pad_seqs=3),
        dict(conv_dim_first=False),
        dict(norm_f32=True),
        dict(slot_pitch=12 * 128 * 128 + 64),
        dict(tokens=4),
    ],
    ids=["padding", "conv_pos_first", "norm_f32", "slot_pitch", "tokens4"],
)
@torch.inference_mode()
def test_variants(kw):
    require_gfx950()
    a = make_inputs(40, seed=3, **kw)
    b = clone_inputs(a)
    kda_decode_fb_gate_spec_reference(**b)
    kda_decode_fb_gate_spec(**a)
    torch.cuda.synchronize()
    pad = a["state_indices"][:, 0] == 0
    if pad.any():
        rows = int(a["cu_seqlens"][-1])
        assert not torch.isnan(a["out"][:rows]).any()
    check(a, b)


@torch.inference_mode()
def test_snapshot_token():
    """With ``snapshot_token`` only that token's state is written; outputs and conv state are unchanged."""
    require_gfx950()
    a = make_inputs(40, seed=5)
    b = clone_inputs(a)
    kda_decode_fb_gate_spec_reference(**b)
    n = a["cu_seqlens"].numel() - 1
    pos = (torch.arange(n, device="cuda") % 3 + 1).to(torch.int32)
    pos = torch.minimum(pos, (a["cu_seqlens"][1:] - a["cu_seqlens"][:-1]) - 1).to(torch.int32)
    before = a["state"].clone()
    kda_decode_fb_gate_spec(**a, snapshot_token=pos)
    torch.cuda.synchronize()
    assert torch.equal(a["conv_state"], b["conv_state"])
    torch.testing.assert_close(a["out"].float(), b["out"].float(), atol=2e-2, rtol=2e-2)
    idx = a["state_indices"].long()
    for i in range(n):
        for t in range(idx.shape[1]):
            slot = int(idx[i, t])
            if t == int(pos[i]):
                torch.testing.assert_close(a["state"][slot], b["state"][slot], atol=2e-3, rtol=1e-3)
            elif t < int(a["cu_seqlens"][i + 1] - a["cu_seqlens"][i]):
                assert torch.equal(a["state"][slot], before[slot])
