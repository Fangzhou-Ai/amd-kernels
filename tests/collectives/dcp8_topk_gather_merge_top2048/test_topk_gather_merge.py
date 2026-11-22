# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
from unittest.mock import patch

import torch
import torch.distributed as dist

from amd_kernels.collectives.dcp8_topk_gather_merge_top2048.reference import reference
from tests.multi_gpu8 import WORLD, run_on_8_gpus

from .workload import K, PAGE, STRIDE, make_block_table, make_plane


def check(rank, dev, comm, rows, seed, ties, graph):
    planes = [make_plane(seed, r, rows, ties) for r in range(WORLD)]
    bt = make_block_table(seed, rows)
    scores = planes[rank][0].to(dev)
    idx = planes[rank][1].to(dev)
    btd = bt.to(dev)
    slots = torch.full((rows * K,), -7, dtype=torch.int32, device=dev)
    indptr = torch.full((rows + 1,), -7, dtype=torch.int32, device=dev)
    want_slots, want_indptr = reference(torch.stack([p[0] for p in planes]), planes[rank][1], bt, rank, PAGE, STRIDE)
    runs = []
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            comm(scores, idx, btd, slots, indptr, PAGE, STRIDE)
        torch.cuda.synchronize()
        dist.barrier()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(3):
                comm(scores, idx, btd, slots, indptr, PAGE, STRIDE)
        for _ in range(2):
            slots.fill_(-7)
            indptr.fill_(-7)
            dist.barrier()
            g.replay()
            torch.cuda.synchronize()
            runs.append((slots.clone(), indptr.clone()))
    else:
        for _ in range(2):
            slots.fill_(-7)
            indptr.fill_(-7)
            comm(scores, idx, btd, slots, indptr, PAGE, STRIDE)
            torch.cuda.synchronize()
            runs.append((slots.clone(), indptr.clone()))
    for s, p in runs:
        assert torch.equal(p.cpu(), want_indptr), (rank, rows, seed)
        assert torch.equal(s[: want_slots.numel()].cpu(), want_slots), (rank, rows, seed)
    assert torch.equal(runs[0][0][: want_slots.numel()], runs[1][0][: want_slots.numel()])


def run(rank, errors):
    dev = torch.device("cuda", rank)
    from amd_kernels.collectives.dcp8_topk_gather_merge_top2048 import TopkGatherMerge

    comm = TopkGatherMerge(rank, dist.group.WORLD, dev)
    comm.warmup()
    import flydsl.compiler.jit_function as jf

    with patch.object(jf.MlirCompiler, "compile", side_effect=AssertionError("compile after warmup")):
        for n, rows in enumerate((1, 12, 40, 65, 192, 256)):
            check(rank, dev, comm, rows, n, ties=n % 2 == 0, graph=n % 2 == 1)
            check(rank, dev, comm, rows, 100 + n, ties=n % 2 == 1, graph=n % 2 == 0)
        check(rank, dev, comm, 24, 7, ties=False, graph=True)
        check(rank, dev, comm, 80, 8, ties=True, graph=True)
        check(rank, dev, comm, 12, 9, ties=2, graph=False)
        check(rank, dev, comm, 40, 11, ties=3, graph=False)
        check(rank, dev, comm, 12, 13, ties=5, graph=False)
        check(rank, dev, comm, 65, 14, ties=5, graph=True)
        check(rank, dev, comm, 12, 15, ties=6, graph=False)
        check(rank, dev, comm, 40, 16, ties=6, graph=True)
        check(rank, dev, comm, 24, 12, ties=4, graph=True)
        check(rank, dev, comm, 65, 10, ties=2, graph=True)
    comm.close()


def test_eight_rank_served_rows_ties_and_graph_replay():
    run_on_8_gpus(run, timeout=100)
