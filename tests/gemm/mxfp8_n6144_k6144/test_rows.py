# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""6144 x 6144 dispatches through its folder: every row count's config is the folder's row, the shape is listed as
tuned, and the common table has no rows of it."""

import pytest
import torch

from amd_kernels.gemm.mxfp8_common import gemm as G
from amd_kernels.gemm.mxfp8_n6144_k6144 import host

BF16 = torch.bfloat16


def _edges():
    """Each row's window edges (bf16 and AITER default rows)."""
    ms = set()
    for rows in host.ROWS.values():
        lo = 1
        for top, _ in rows:
            ms.update({lo, top})
            lo = top + 1
    return sorted(ms)


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("allow_decode", [True, False])
def test_dispatch_reads_the_folder(out_dtype, allow_decode):
    for m in _edges():
        want = G.select_from_rows(host.ROWS, m, 6144, 6144, out_dtype, allow_decode)
        assert want is not None
        assert G._select_config(m, 6144, 6144, out_dtype, allow_decode) == want, m
        assert host.select(m, out_dtype, allow_decode) == want, m


def test_listed_and_not_in_the_common_table():
    assert (6144, 6144) in G.tuned_shapes(BF16)
    assert not any((n, k) == (6144, 6144) for n, k, _ in G._load_tuned_configs())
