# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""KDA chunked prefill for 12 heads of dim 128 (FlyDSL, gfx950): causal conv + SiLU, gate and q / k norms, the chunked delta-rule scan
over fp32 recurrent states and the gated RMSNorm in three launches (four for long chains). The contract is ``reference.py``."""

from .host import gated_norm, plan_groups, prefill, scan
from .prologue import chunk_table, prologue

__all__ = ["chunk_table", "gated_norm", "plan_groups", "prefill", "prologue", "scan"]
