# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the fused DCP8 combine against vLLM's v_up-first sequence (mask LSE, AITER LSE all-gather, bmm_vup,
scale, AITER reduce-scatter; plus the Triton output gate with ``--gate``), 8 GPUs, inputs rotating over > 300 MB,
MAX over ranks, median us per call, roofline table.

    python3 benchmarks/attention/dcp8_lse_combine_vup_mla_h12_kv512/benchmark_combine.py [--gate] [--tokens 12,24,...]
"""
import argparse
import os
import socket
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 8
CALLS = 24
ROUNDS = 5
HBM_TBS = 8.0
XGMI_LINK_GBS = 55.0
PEAK_BF16_TFLOPS = 1250.0


def _stock_gate(a, g, out):
    from vllm.models.kimi_k3.amd.dsa import _gate_mul_kernel
    import triton

    _gate_mul_kernel[(triton.cdiv(a.numel(), 1024),)](a, g, out, a.numel(), BLOCK=1024)


def _worker(rank, port, tokens, gate, out_q):
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import DcpVupCombine, dcp8_lse_combine_vup, prepare_w_uv
    from vllm.distributed.device_communicators.aiter_custom_all_reduce import AiterCustomAllreduce
    from vllm.v1.attention.ops.dcp import dcp_vup_rs_combine

    dev = torch.device("cuda", rank)
    comm = DcpVupCombine(rank, dist.group.WORLD, dev)
    aiter = AiterCustomAllreduce(group=dist.group.WORLD, device=dev)
    g = torch.Generator(device=dev).manual_seed(rank)
    rows_out = []
    for m in tokens:
        xb = m * 96 * 1024
        nrot = max(4, min(CALLS, -(-300 * 2**20 // (xb + 12_582_912))))
        xs = [(torch.randn((m, 96, 512), generator=g, device=dev) * 0.5).bfloat16() for _ in range(nrot)]
        lses = [torch.randn((m, 96), generator=g, device=dev) * 2 + 3 for _ in range(nrot)]
        w_uv = [(torch.randn((96, 512, 128), generator=g, device=dev) * 0.05).bfloat16() for _ in range(nrot)]
        wt = [prepare_w_uv(w) for w in w_uv]
        gates = [torch.randn((m, 1536), generator=g, device=dev).bfloat16() for _ in range(nrot)]
        outs = [torch.empty((m, 12, 128), dtype=torch.bfloat16, device=dev) for _ in range(nrot)]
        gouts = [torch.empty((m, 1536), dtype=torch.bfloat16, device=dev) for _ in range(nrot)]
        nseq = max(1, m // 2)
        qsl = torch.linspace(0, m, nseq + 1).round().int().to(dev)
        sls = torch.randint(1, 5000, (nseq,), generator=g, device=dev).int()

        def ours(i):
            dcp8_lse_combine_vup(comm, xs[i], lses[i], wt[i], gouts[i].view(m, 12, 128) if gate else outs[i],
                                 sls, qsl, gate=gates[i] if gate else None)

        def stock(i):
            dcp_vup_rs_combine(xs[i], lses[i], w_uv[i], outs[i], aiter, rank, WORLD, True, sls, qsl)
            if gate:
                _stock_gate(outs[i], gates[i], gouts[i])

        def graph_of(fn, ctx):
            for i in range(nrot):
                fn(i)
            torch.cuda.synchronize()
            dist.barrier()
            gr = torch.cuda.CUDAGraph()
            with ctx():
                with torch.cuda.graph(gr):
                    for c in range(CALLS):
                        fn(c % nrot)
            return gr

        import contextlib
        g_ours = graph_of(ours, contextlib.nullcontext)
        g_stock = graph_of(stock, aiter.capture)
        times = {"ours": [], "stock": []}
        for rd in range(ROUNDS):
            for name, gr in (("stock", g_stock), ("ours", g_ours)) if rd % 2 else (("ours", g_ours), ("stock", g_stock)):
                gr.replay()
                torch.cuda.synchronize()
                dist.barrier()
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                gr.replay()
                e.record()
                torch.cuda.synchronize()
                t = torch.tensor([s.elapsed_time(e) * 1000 / CALLS])
                dist.all_reduce(t, op=dist.ReduceOp.MAX)
                times[name].append(t.item())
        rows_out.append((m, statistics.median(times["ours"]), statistics.median(times["stock"])))
        del g_ours, g_stock, xs, lses, w_uv, wt, gates, outs, gouts
        torch.cuda.empty_cache()
    if rank == 0:
        out_q.put(rows_out)
    dist.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="1,12,24,40,65,80,128,192")
    ap.add_argument("--gate", action="store_true")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(r, port, tokens, a.gate, q)) for r in range(WORLD)]
    for p in procs:
        p.start()
    rows = q.get(timeout=1800)
    for p in procs:
        p.join()
    print(f"\nDCP8 combine + W_UV{' + output gate' if a.gate else ''}: graph replay, median us per call, MAX over 8 ranks")
    print(f"{'M':>4} {'xGMI MB/rank':>12} {'HBM MB':>8} {'GFLOP':>6} {'xGMI floor us':>13} {'HBM floor us':>12} "
          f"{'stock':>8} {'fused':>8} {'speedup':>8} {'fused/floor':>11}")
    for m, ours, stock in rows:
        xgmi = 7 * m * 12 * 128 * 2 + 7 * m * 12 * 4
        hbm = m * 96 * 1024 + 96 * 128 * 512 * 2 + m * 12 * 256 + 2 * WORLD * m * 12 * 260
        flops = 2 * 96 * m * 512 * 128
        floor_x = xgmi / (7 * XGMI_LINK_GBS * 1e3)
        floor_h = hbm / (HBM_TBS * 0.8 * 1e6)
        floor = max(floor_x, floor_h, flops / (PEAK_BF16_TFLOPS * 0.8 * 1e6))
        print(f"{m:>4} {xgmi / 1e6:>12.2f} {hbm / 1e6:>8.1f} {flops / 1e9:>6.2f} {floor_x:>13.2f} {floor_h:>12.2f} "
              f"{stock:>8.1f} {ours:>8.1f} {stock / ours:>7.2f}x {ours / floor:>10.1f}x")


if __name__ == "__main__":
    main()
