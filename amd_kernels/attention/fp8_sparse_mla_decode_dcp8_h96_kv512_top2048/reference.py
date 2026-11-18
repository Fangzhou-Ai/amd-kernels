# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sparse MLA decode over an fp8_ds_mla cache, the contract of the kernel.

Inputs: ``q [rows, 96, 576]`` bf16 (512 latent + 64 rope dims), the cache as flat 656 B records, and one compact CSR
slice per row: row r attends the physical slots ``slots[indptr[r] : indptr[r + 1]]`` (this rank's share of the
row's top-2048 keys). A record is 512 e4m3 latent bytes, 4 f32 per-128-dim power-of-two scales and 64 bf16 rope values:
``k = [e4m3 * scale (block of 128 dims), rope]``, and the value of a key is its dequantized 512 latent dims.

Per row and head: ``s_j = sm_scale * q . k_j`` in fp32, ``lse = log(sum_j exp(s_j))`` (natural log, fp32),
``out = bf16(sum_j softmax(s)_j * v_j)`` accumulated in fp32. A row with no key gives ``out = 0`` and ``lse = -inf``
(CUDA-graph padding rows). ``out`` and ``lse`` are the per-rank partials of the DCP combine
(``dcp8_lse_combine_vup_mla_h12_kv512``).

Tolerance (the kernel quantizes q per (head, 128 dims) to an fp8 hi term plus an fp8 residual, takes P in fp16 scaled by
the value block's scale, and accumulates in fp32): per (row, head) cosine of ``out`` > 0.9999, max abs error of ``out``
<= 2^-6 of its row's max, ``lse`` abs error <= 1e-2; two runs match bit for bit.
"""

import torch

H = 96
L = 512
R = 64
D = L + R
RECORD_BYTES = 656
BLOCK = 128


def dequant_records(rec: torch.Tensor):
    """``rec [n, 656]`` uint8 -> (latent f32 [n, 512], rope f32 [n, 64])."""
    n = rec.shape[0]
    scale = rec[:, L : L + 16].contiguous().view(torch.float32)  # [n, 4]
    latent = rec[:, :L].contiguous().view(torch.float8_e4m3fn).float().view(n, 4, BLOCK) * scale[:, :, None]
    rope = rec[:, L + 16 :].contiguous().view(torch.bfloat16).float()
    return latent.view(n, L), rope


def reference(q, kv_rows, indptr, slots, sm_scale):
    """``(out bf16 [rows, 96, 512], lse f32 [rows, 96])`` for ``kv_rows [N, 656]`` uint8 records."""
    rows = q.shape[0]
    out = torch.zeros(rows, H, L, dtype=torch.bfloat16, device=q.device)
    lse = torch.full((rows, H), float("-inf"), dtype=torch.float32, device=q.device)
    ip = indptr.tolist()
    for r in range(rows):
        beg, end = ip[r], ip[r + 1]
        if end <= beg:
            continue
        latent, rope = dequant_records(kv_rows[slots[beg:end].long()])
        s = (q[r].float() @ torch.cat([latent, rope], 1).T) * sm_scale
        lse[r] = torch.logsumexp(s, dim=-1)
        out[r] = (torch.softmax(s, dim=-1) @ latent).to(torch.bfloat16)
    return out, lse


def check_close(out, lse, ref_out, ref_lse):
    """Assert the tolerance of the module docstring (empty rows: out 0, lse -inf exactly)."""
    o, ro = out.float(), ref_out.float()
    empty = torch.isinf(ref_lse[:, 0]) if ref_lse.numel() else torch.zeros(0, dtype=torch.bool)
    assert torch.equal(torch.isinf(lse[:, 0]) & (lse[:, 0] < 0), empty), "empty rows differ"
    live = ~empty
    assert not o[empty].abs().max().item() if empty.any() else True, "empty row has output"
    if live.any():
        o, ro = o[live], ro[live]
        cos = torch.nn.functional.cosine_similarity(o, ro, dim=-1)
        assert cos.min().item() > 0.9999, f"cosine {cos.min().item()}"
        bound = ro.abs().amax(dim=(1, 2), keepdim=True) * 2**-6
        err = (o - ro).abs()
        assert (err <= bound).all(), f"max abs error {err.max().item()} vs {bound.min().item()}"
        assert (lse[live] - ref_lse[live]).abs().max().item() <= 1e-2, "lse"
