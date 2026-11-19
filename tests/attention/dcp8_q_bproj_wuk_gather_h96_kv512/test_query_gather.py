# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight-rank contract of both variants at the served row counts: eager and graph calls, replays, determinism."""

import pytest
import torch
import torch.distributed as dist
from amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512 import (
    QueryGatherComm,
    pack_q_b,
    pack_w_uk,
    q_b_allgather_wuk_all_heads,
    q_b_wuk_allgather_latent,
)
from amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512.reference import gathered_query_reference

from tests.attention.dcp8_q_bproj_wuk_gather_h96_kv512.multi_gpu8 import run_on_8_gpus

ROW_GROUPS = ((1, 5, 6, 10, 12, 15, 16, 17), (24, 30, 40, 45, 48, 49, 65), (80, 96, 97, 128), (160, 192, 193, 256))
VARIANTS = {"latent": q_b_wuk_allgather_latent, "q": q_b_allgather_wuk_all_heads}


def check(out, ref, tag):
    assert out.shape == ref.shape, (tag, out.shape)
    d = out.float() - ref.float()
    rel = float(d.norm() / ref.float().norm())
    assert rel <= 4e-3, (tag, "rel L2", rel)
    cos = torch.nn.functional.cosine_similarity(out.float().flatten(0, 1), ref.float().flatten(0, 1), dim=1)
    assert float(cos.min()) >= 0.9999, (tag, "cos", float(cos.min()))


def worker(rank, errors, mode, rows):
    dev = torch.device("cuda", rank)
    g = torch.Generator(device=dev).manual_seed(7)
    rnd = lambda *s, scale=1.0: (torch.randn(*s, device=dev, dtype=torch.float32, generator=g) * scale).bfloat16()  # noqa: E731
    w_qb = [rnd(2304, 1536, scale=0.03) for _ in range(8)]
    w_uk = [rnd(12, 128, 512, scale=0.05) for _ in range(8)]
    wq = pack_q_b(w_qb[rank])
    wu = pack_w_uk(w_uk[rank] if mode == "latent" else torch.cat(w_uk))
    comm = QueryGatherComm(rank, dist.group.WORLD, dev, mode)
    call = VARIANTS[mode]
    for m in rows:
        q_a = rnd(m, 1536)
        ref = gathered_query_reference(q_a, w_qb, w_uk)
        out = call(comm, q_a, wq, wu).clone()
        torch.cuda.synchronize()
        check(out, ref, ("eager", mode, m))
        again = call(comm, q_a, wq, wu).clone()
        assert torch.equal(out.view(torch.uint8), again.view(torch.uint8)), ("repeat", mode, m)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = call(comm, q_a, wq, wu)
        for it in range(4):
            q_a.copy_(rnd(m, 1536))
            ref = gathered_query_reference(q_a, w_qb, w_uk)
            graph.replay()
            torch.cuda.synchronize()
            check(gout, ref, ("graph", mode, m, it))
        if rank == 0:
            print("contract PASS", mode, m, flush=True)
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(3)).tolist()
    for k, i in enumerate(order):
        m = rows[i]
        q_a = rnd(m, 1536)
        check(call(comm, q_a, wq, wu), gathered_query_reference(q_a, w_qb, w_uk), ("mixed", mode, m, k))
    comm.close()


@pytest.mark.parametrize("rows", ROW_GROUPS, ids=lambda r: f"rows{r[0]}-{r[-1]}")
@pytest.mark.parametrize("mode", ["latent", "q"])
def test_query_gather_contract(mode, rows):
    run_on_8_gpus(worker, mode, rows, timeout=900)
