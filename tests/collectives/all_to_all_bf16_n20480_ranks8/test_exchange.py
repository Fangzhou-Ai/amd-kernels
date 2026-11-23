# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight ranks: raw bits, dynamic GPU counts, the all-gather mode, repeated exchanges into one buffer, graphs.

No arithmetic occurs here; the exact byte exchange covers every bf16 pattern (NaN payloads, signed zeros). The received
rows are read in place by a copy kernel after each wait, so a stale cache line of the receive buffer fails a case.
"""
import torch
import torch.distributed as dist

from amd_kernels.collectives.all_to_all_bf16_n20480_ranks8.reference import N, WORLD, merge_received_reference
from tests.multi_gpu8 import run_on_8_gpus


def _rccl_reference(x, counts, rank, rccl):
    n = counts[rank]
    recv = torch.empty(WORLD * n * N * 2, dtype=torch.uint8, device=x.device)
    dist.all_to_all_single(recv, x.view(torch.uint8).flatten(),
                           output_split_sizes=[n * N * 2] * WORLD,
                           input_split_sizes=[c * N * 2 for c in counts], group=rccl)
    return merge_received_reference(recv.view(torch.bfloat16).view(WORLD, n, N))


def _gather_reference(x, rows, rccl):
    parts = [torch.empty_like(x) for _ in range(WORLD)]
    dist.all_gather(parts, x, group=rccl)
    return torch.cat(parts, dim=1)[:rows]


def patterns():
    out = []
    for reqs in (3, 12, 40):
        dense = torch.bincount(torch.arange(reqs) % WORLD, minlength=WORLD).tolist()
        slots = torch.randperm(256, generator=torch.Generator().manual_seed(reqs))[:reqs]
        random = torch.bincount(slots % WORLD, minlength=WORLD).tolist()
        for requests in (dense, random):
            for steps in (1, 4):
                if reqs * steps <= 256:
                    out.append([v * steps for v in requests])
    out += [[0] * WORLD, [0, 0, 192, 0, 0, 0, 0, 0], [8, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 8]]
    return out


def _worker(rank, errors):
    dev = torch.device("cuda", rank)
    rccl = dist.new_group(backend="nccl")
    from amd_kernels.collectives.all_to_all_bf16_n20480_ranks8 import RowAllToAll
    comm = RowAllToAll(rank, dist.group.WORLD, dev)
    generator = torch.Generator(device=dev).manual_seed(17001 + rank)
    comm.warmup()
    from flydsl.compiler.jit_function import MlirCompiler
    compile_calls = []
    original = MlirCompiler.compile

    def count_compile(*args, **kwargs):
        compile_calls.append(1)
        return original(*args, **kwargs)

    MlirCompiler.compile = count_compile
    try:
        for k, counts in enumerate(patterns()):
            cs = torch.tensor(counts, device=dev, dtype=torch.int32)
            n, rows = counts[rank], sum(counts)
            sets = [torch.randint(-32768, 32768, (rows, N), dtype=torch.int16, device=dev,
                                  generator=generator).view(torch.bfloat16) for _ in range(4)]
            outs = [torch.empty(max(n, 1), WORLD * N, device=dev, dtype=torch.bfloat16) for _ in sets]
            refs = [_rccl_reference(x, counts, rank, rccl) for x in sets]
            for j, (x, out, ref) in enumerate(zip(sets, outs, refs)):
                comm(x, cs, j % 2)
                out[:n].copy_(comm.recv(n, j % 2))
                torch.cuda.synchronize()
                assert torch.equal(out[:n].view(torch.int16), ref.view(torch.int16)), (rank, k, "eager")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for j, (x, out) in enumerate(zip(sets, outs)):
                    comm(x, cs, j % 2)
                    out[:n].copy_(comm.recv(n, j % 2))
            for repeat in range(3):
                for out in outs:
                    out.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                for out, ref in zip(outs, refs):
                    assert torch.equal(out[:n].view(torch.int16), ref.view(torch.int16)), (rank, k, repeat)
            del graph, sets, outs, refs
            if rows:  # all-gather: every rank receives every source's rows (two exchanges per graph, buffers 0 / 1)
                x = torch.randint(-32768, 32768, (rows, N), dtype=torch.int16, device=dev,
                                  generator=generator).view(torch.bfloat16)
                out = torch.empty(rows, WORLD * N, device=dev, dtype=torch.bfloat16)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    comm(x, None, 0)
                    comm(x, None, 1)
                    out.copy_(comm.recv(rows, 1))
                ref = _gather_reference(x, rows, rccl)
                for _ in range(2):
                    out.fill_(float("nan"))
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(out.view(torch.int16), ref.view(torch.int16)), (rank, k, "gather")
                del graph
        x = torch.empty(80, N, dtype=torch.bfloat16, device=dev)
        out = torch.empty(80, WORLD * N, dtype=torch.bfloat16, device=dev)
        cs = torch.tensor([10] * WORLD, dtype=torch.int32, device=dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            comm(x, cs, 0)
            out.copy_(comm.recv(80, 0))
            comm(x, cs, 1)
        for k in range(16):
            counts = [80 if j == k % WORLD else 0 for j in range(WORLD)] if k % 2 else list(range(0, 16, 2))
            counts[-1] += 80 - sum(counts)
            cs.copy_(torch.tensor(counts, dtype=torch.int32))
            x.view(torch.int16).random_(-32768, 32768, generator=generator)
            ref = _rccl_reference(x, counts, rank, rccl)
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:counts[rank]].view(torch.int16), ref.view(torch.int16)), (rank, "dynamic", k)
        assert compile_calls == [], compile_calls
    finally:
        MlirCompiler.compile = original
    del graph
    comm.close()
    dist.destroy_process_group(rccl)


def test_eight_rank_exchange():
    run_on_8_gpus(_worker)
