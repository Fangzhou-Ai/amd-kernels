# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

The kernel has one variant: rows, strides, the key counts and the scale are runtime arguments. Warm it once, then walk
every served row count (the 7 critical sizes and the CUDA-graph sizes up to 192) and assert nothing compiles.
"""

import flydsl.compiler as flyc
import torch
from amd_kernels.attention.fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 import fp8_sparse_mla_decode

from .cases import GRAPH_ROWS, SERVED_ROWS, SM_SCALE, make_batch, require_gfx950, served_counts


@torch.inference_mode()
def test_no_compile_after_warmup(monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(3)
    q, cache, indptr, slots = make_batch(served_counts(192, gen), gen)
    fp8_sparse_mla_decode(q[:1], cache, indptr[:2], slots[: int(indptr[1])], SM_SCALE)
    torch.cuda.synchronize()
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    for rows in sorted(set(SERVED_ROWS) | set(GRAPH_ROWS)):
        fp8_sparse_mla_decode(q[:rows], cache, indptr[: rows + 1], slots[: int(indptr[rows])], SM_SCALE)
        assert not compiled, f"{rows} rows compiled {compiled}"
    torch.cuda.synchronize()
