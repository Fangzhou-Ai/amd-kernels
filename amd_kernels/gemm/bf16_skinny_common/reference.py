# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The contract: y = x @ w^T for bf16 x [M, K] and bf16 w [N, K] (the unpacked, checkpoint layout).

The product of the bf16 operands in fp64, rounded once to bf16. The kernel accumulates in fp32 (K ascending
within a split, splits summed in split order), so it differs from the reference (and from hipBLASLt / wvSplitKrc)
only in the sum order: per-row cosine >= 0.99999 and relative L2 <= 1e-3, deterministic bits. NaN / inf propagate
as in an fp32 accumulation.
"""
import torch


def bf16_linear_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (x.double() @ w.double().t()).bfloat16()
