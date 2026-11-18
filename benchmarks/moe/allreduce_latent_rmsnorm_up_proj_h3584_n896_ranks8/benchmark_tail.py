# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Eight ranks, CUDA graph replay over 48 distinct weight / activation sets (307 MB of up-projection weights, larger
than the 256 MB LLC), median of alternated rounds, slowest rank. Baseline = what vLLM runs today (stock AITER custom
all-reduce of the latent, the Triton fused RMSNorm + up-projection, AITER all-reduce of the [rows, 7168] output); the
fused entry replaces the first two and keeps the same final all-reduce. Prints a roofline table."""
import argparse
import json
import statistics

import torch
import torch.distributed as dist
from amd_kernels.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8 import LatentTailComm, allreduce_latent_rmsnorm_up_proj

K, N, H = 3584, 896, 7168
HBM_TBS = 8.0
XGMI_TBS = 0.45  # one direction of the 7 links, per GPU
NI = 48


def stock_triton():
    from vllm.models.kimi_k3.amd.ops.latent_norm_up_proj import latent_norm_up_proj_triton

    return latent_norm_up_proj_triton


def timeit(graph, rounds, reps=10):
    ts = []
    for _ in range(rounds):
        dist.barrier()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record()
        for _ in range(reps):
            graph.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / (reps * NI))
    return ts


def worker(rank, errors, rows, rounds):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    dev = torch.device("cuda", rank)
    ca = CustomAllreduce(group=dist.group.WORLD, device=dev, max_size=64 * 1024 * 1024)
    comm = LatentTailComm(rank, dist.group.WORLD, dev)
    comm_split = LatentTailComm(rank, dist.group.WORLD, dev, fused=False)
    triton_fn = stock_triton()
    ws = [(torch.randn(N, K, device=dev) * 0.03).bfloat16() for _ in range(NI)]
    nw = (1 + 0.1 * torch.randn(K, device=dev)).bfloat16()
    all_results = {}
    for m in rows:
        lat = [torch.randn(m, K, device=dev).bfloat16() for _ in range(NI)]
        full = [torch.randn(m, H, device=dev).bfloat16() for _ in range(NI)]
        full2 = [f.clone() for f in full]

        def stock(i):
            x = ca.custom_all_reduce(lat[i])
            view = full[i][:, rank * N:(rank + 1) * N]
            triton_fn(x, nw, 1e-6, ws[i], view)
            return ca.custom_all_reduce(full[i])

        def fused(i):
            view = full2[i][:, rank * N:(rank + 1) * N]
            allreduce_latent_rmsnorm_up_proj(comm, lat[i], nw, ws[i], view)
            return ca.custom_all_reduce(full2[i])

        def stock_ar_only(i):
            return ca.custom_all_reduce(lat[i])

        def fused_only(i):
            view = full2[i][:, rank * N:(rank + 1) * N]
            return allreduce_latent_rmsnorm_up_proj(comm, lat[i], nw, ws[i], view)

        def split_only(i):
            view = full2[i][:, rank * N:(rank + 1) * N]
            return allreduce_latent_rmsnorm_up_proj(comm_split, lat[i], nw, ws[i], view)

        def part_only(mode):
            def f(i):
                view = full2[i][:, rank * N:(rank + 1) * N]
                return comm._launch(lat[i], nw, ws[i], view, view, 1e-6, modes=(mode,))

            return f

        def stock_mid(i):
            x = lat[i]
            view = full[i][:, rank * N:(rank + 1) * N]
            triton_fn(x, nw, 1e-6, ws[i], view)
            return view

        def final_ar(i):
            return ca.custom_all_reduce(full[i])

        fns = dict(stock=stock, fused=fused, stock_ar=stock_ar_only, stock_triton=stock_mid, final_ar=final_ar,
                   fused_core=fused_only, split_core=split_only, ar_part=part_only("ar"), up_part=part_only("up"))
        graphs = {}
        for name, fn in fns.items():
            fn(0)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with ca.capture(), torch.cuda.graph(g):
                for i in range(NI):
                    fn(i)
            graphs[name] = g
            g.replay()
        torch.cuda.synchronize()
        res = {k: [] for k in fns}
        names = list(fns)
        for r in range(rounds):
            order = names if r % 2 == 0 else names[::-1]
            for k in order:
                res[k] += timeit(graphs[k], 1)
        every = [None] * 8
        dist.all_gather_object(every, res)
        all_results[m] = {k: statistics.median(max(v[k][j] for v in every) for j in range(len(res[k])))
                          for k in fns}
        del graphs
    if rank == 0:
        print(json.dumps(all_results), flush=True)
        report(all_results)
    comm.close()
    comm_split.close()
    ca.close()


def report(res):
    print("\nrows | stock AR | stock triton | stock final AR | stock total | fused core (AR+norm+up) | two launches"
          " | AR part | up part | fused + final AR | speedup")
    for m, r in res.items():
        print(f"{m:4d} | {r['stock_ar']:.1f} | {r['stock_triton']:.1f} | {r['final_ar']:.1f} | {r['stock']:.1f} | "
              f"{r['fused_core']:.1f} | {r['split_core']:.1f} | {r['ar_part']:.1f} | {r['up_part']:.1f} | "
              f"{r['fused']:.1f} | {r['stock'] / r['fused']:.2f}x")
    print("\nroofline of the fused core (per rank): weights 6.4 MB over HBM, latent AR over xGMI, 2 M 3584 896 FLOPs")
    print("rows | HBM floor us | xGMI floor us (2 x 7/8 M 7168 B) | MFMA floor us (1.25 PF) | fused core us | % of max floor")
    for m, r in res.items():
        hbm = (N * K * 2 + m * K * 2 * 2) / (HBM_TBS * 0.8 * 1e6)
        xg = 2 * 7 / 8 * m * K * 2 / (XGMI_TBS * 1e6)
        fl = 2 * m * K * N / (1.25e15) * 1e6
        floor = max(hbm, xg, fl)
        print(f"{m:4d} | {hbm:.2f} | {xg:.2f} | {fl:.2f} | {r['fused_core']:.1f} | {100 * floor / r['fused_core']:.0f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rows", nargs="+", type=int, default=[12, 24, 40, 65, 80, 128, 192])
    p.add_argument("--rounds", type=int, default=7)
    a = p.parse_args()
    from tests.moe.allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8.multi_gpu8 import run_on_8_gpus

    run_on_8_gpus(worker, a.rows, a.rounds, timeout=900)
