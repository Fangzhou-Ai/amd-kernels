# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The up-projection role on one GPU (``mode="up"``: it reads ``xn`` and the squared sums from the ``lat`` buffer): every
served row count, fresh inputs on every run (a stale latent read shows), an output that aliases the shared view, and
bit-identical reruns."""
import pytest
import torch
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import tail_kernel as T
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.ipc import f32_bits, run_compiled
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.reference import up_proj_reference

from .workload import K, N, rand

ROWS = (1, 12, 16, 24, 40, 65, 80, 128, 192, 256)


def run(m, ll2=False, runs=3):
    dev = torch.device("cuda")
    lat_buf = torch.zeros(T.LAT_ALLOC // 2, dtype=torch.bfloat16, device=dev)
    xn_rows = lat_buf[:T.MAX_ROWS * K].view(T.MAX_ROWS, K)
    ssq = lat_buf[T.MAX_ROWS * K: T.MAX_ROWS * K + 2 * T.MAX_ROWS * T.WORLD].view(torch.float32).view(T.MAX_ROWS, T.WORLD)
    table = torch.tensor([0] * 8 + [lat_buf.data_ptr()] * 8, dtype=torch.int64, device=dev)
    counts = torch.zeros(T.CNT_WORDS, dtype=torch.int32, device=dev)
    mpg = T.mpg_of((m + 15) // 16)
    kernel = T.compile_tail("up", mpg, False, ll2, ll2)
    calls = 0
    for r in range(runs):
        x = rand((T.MAX_ROWS, K), 10 * r + m, dev, 1.5)
        nw = (1 + rand((K,), 10 * r + 1, dev, 0.1)).bfloat16()
        w = rand((N, K), 10 * r + 2, dev, 0.03)
        xn = (x.float() * nw.float()).bfloat16()
        xn_rows.copy_(xn)
        sqref = (x.float() ** 2).view(T.MAX_ROWS, T.WORLD, K // T.WORLD).sum(-1)
        ssq.copy_(sqref)
        full = rand((m, 7168), 10 * r + 3, dev)
        view = full[:, 896:1792]
        shared0 = view.clone()

        def call():
            nonlocal calls
            calls += 1
            if ll2:
                fill_ll(lat_buf, xn, sqref, calls)
            run_compiled(kernel, m, (m + 7) // 8, 0, table.data_ptr(), counts.data_ptr(), 0, 0, nw.data_ptr(),
                         w.data_ptr(), view.data_ptr(), view.stride(0), view.data_ptr(), view.stride(0),
                         f32_bits(1e-6), 0, 0, torch.cuda.current_stream())
            torch.cuda.synchronize()

        call()
        y, up = up_proj_reference(x[:m], nw, w, shared0)
        err = (view.double() - y).abs()
        tol = 2.0**-8 * y.abs() + 2.0**-8 * up.abs().amax(-1, keepdim=True) + 1e-6
        assert bool((err <= tol).all()), (m, r, float(err.max()))
        cos = torch.nn.functional.cosine_similarity(view.double(), y, dim=1)
        assert float(cos.min()) >= 0.99999, (m, r, float(cos.min()))
        again = view.clone()
        view.copy_(shared0)
        call()
        assert torch.equal(view.view(torch.int16), again.view(torch.int16)), ("rerun", m)


def fill_ll(lat_buf, xn, ssq, epoch):
    """The LL image of ``xn`` and ``ssq`` in ``lat_buf`` (every payload dword followed by the epoch)."""
    raw = lat_buf.view(torch.int32)
    pay = xn.contiguous().view(torch.int32).view(T.MAX_ROWS, K // 2)
    ll = torch.stack([pay, torch.full_like(pay, epoch)], dim=-1).view(T.MAX_ROWS, K)
    raw[: T.LATLL_BYTES // 4] = ll.view(-1)
    sq = ssq.contiguous().view(torch.int32)
    raw[T.LATLL_BYTES // 4: T.LATLL_BYTES // 4 + T.MAX_ROWS * T.WORLD * 2] = torch.stack(
        [sq, torch.full_like(sq, epoch)], dim=-1).view(-1)


@pytest.mark.parametrize("ll2", (False, True))
@pytest.mark.parametrize("m", ROWS)
def test_up_proj(m, ll2):
    run(m, ll2)
