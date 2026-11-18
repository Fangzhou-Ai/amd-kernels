# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The up-projection role on one GPU (``mode="up"``: it reads the reduced latent from the ``lat`` buffer): every served
row count and every (NS, NJ) tile, fresh inputs on every run (a stale partial or latent read shows), an output that
aliases the shared view, and bit-identical reruns."""
import pytest
import torch
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import tail_kernel as T
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.ipc import f32_bits, run_compiled
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.reference import up_proj_reference

from .workload import K, N, rand

ROWS = (1, 12, 24, 40, 65, 80, 128, 192, 256)
TILES = ((4, 2), (2, 2), (1, 2), (4, 1))


def run(m, ns, nj, runs=3):
    dev = torch.device("cuda")
    lat = torch.zeros(T.MAX_ROWS, K, dtype=torch.bfloat16, device=dev)
    table = torch.tensor([0] * 8 + [lat.data_ptr()] * 8, dtype=torch.int64, device=dev)
    counts = torch.zeros(T.CNT_WORDS, dtype=torch.int32, device=dev)
    sk = T.split_k(ns)
    part = torch.zeros(sk * T.MAX_ROWS * N, dtype=torch.float32, device=dev)
    ss = torch.zeros(T.groups(nj) * sk * T.MAX_ROWS, dtype=torch.float32, device=dev)
    tickets = torch.zeros(T.groups(nj) * 32, dtype=torch.int32, device=dev)
    kernel = T.compile_tail("up", (m + 15) // 16, ns, nj, 4)
    for r in range(runs):
        lat.copy_(rand((T.MAX_ROWS, K), 10 * r + m, dev, 1.5))
        nw = (1 + rand((K,), 10 * r + 1, dev, 0.1)).bfloat16()
        w = rand((N, K), 10 * r + 2, dev, 0.03)
        full = rand((m, 7168), 10 * r + 3, dev)
        view = full[:, 896:1792]
        shared0 = view.clone()

        def call():
            run_compiled(kernel, m, (m + 3) // 4, 0, table.data_ptr(), counts.data_ptr(), 0, 0, nw.data_ptr(),
                         w.data_ptr(), view.data_ptr(), view.stride(0), view.data_ptr(), view.stride(0),
                         part.data_ptr(), ss.data_ptr(), tickets.data_ptr(), f32_bits(1e-6), 0,
                         torch.cuda.current_stream())
            torch.cuda.synchronize()

        call()
        y, up = up_proj_reference(lat[:m], nw, w, shared0)
        err = (view.double() - y).abs()
        tol = 2.0**-8 * y.abs() + 2.0**-8 * up.abs().amax(-1, keepdim=True) + 1e-6
        assert bool((err <= tol).all()), (m, ns, nj, r, float(err.max()))
        cos = torch.nn.functional.cosine_similarity(view.double(), y, dim=1)
        assert float(cos.min()) >= 0.99999, (m, ns, nj, r, float(cos.min()))
        again = view.clone()
        view.copy_(shared0)
        call()
        assert torch.equal(view.view(torch.int16), again.view(torch.int16)), ("rerun", m, ns, nj)


@pytest.mark.parametrize("ns,nj", TILES)
@pytest.mark.parametrize("m", ROWS)
def test_up_proj(m, ns, nj):
    run(m, ns, nj)
