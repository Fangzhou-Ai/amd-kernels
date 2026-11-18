# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph replay of the MoE router (GEMM + sigmoid + bias + top-16) against vLLM's path today: GateLinear's
fp32-out bf16 GEMM (hipBLASLt) followed by AITER's biased_grouped_topk (one group, topk_reg_kernel). One graph of 24 calls over 24 weight copies (24 x 12.8 MB > the 256 MB MALL: HBM on every call, as in
serving) and 24 activation sets, 5 replays x 15 rounds, median us. Roofline: the HBM time of the weight stream
(8 TB/s) plus the launch floor of the two dependent launches (a one-element torch kernel in the same graph, twice).
Run: python3 benchmarks/moe/router_sigmoid_bias_top16_e896_h7168/benchmark_router.py [M ...]
"""
import os
import statistics
import sys

import torch

os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")

from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168 import host, router_sigmoid_bias_top16_e896_h7168, shuffle_weight
from amd_kernels.moe.router_sigmoid_bias_top16_e896_h7168.reference import E, H, TOPK

HBM_TBPS = 8.0
PEAK_BF16_TFLOPS = 1250.0
COPIES = 24


def timing(calls):
    for f in calls[:2]:
        f()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for f in calls:
            f()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(15):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / (5 * len(calls)))
    return statistics.median(times)


if __name__ == "__main__":
    from aiter import biased_grouped_topk

    ms = [int(v) for v in sys.argv[1:]] or [1, 12, 24, 40, 65, 80, 128, 192]
    torch.manual_seed(0)
    ws = [(torch.randn(E, H, device="cuda") * 0.02).to(torch.bfloat16) for _ in range(COPIES)]
    wss = [host.shuffle_weight(w) for w in ws]
    bias = torch.randn(E, device="cuda") * 0.1
    scratch = [torch.zeros(1, device="cuda") for _ in range(COPIES)]
    floor = timing([lambda s=s: s.add_(1) for s in scratch])
    wbytes = E * H * 2
    print(f"launch floor (one trivial kernel in the graph): {floor:.2f} us")
    print(f"{'M':>4} {'stock gemm':>10} {'stock':>7} | {'gemm':>6} {'top-k':>6} {'fused':>6} | {'HBM':>5} {'floor':>6} "
          f"{'TB/s':>5} {'%floor':>6} {'TFLOPs':>6} {'%peak':>5} {'speedup':>7}")
    for m in ms:
        xs = [torch.randn(m, H, device="cuda").to(torch.bfloat16) for _ in range(COPIES)]
        sw = [torch.empty(m, TOPK, device="cuda") for _ in range(COPIES)]
        si = [torch.empty(m, TOPK, dtype=torch.int32, device="cuda") for _ in range(COPIES)]
        stock_gemm = timing([lambda x=x, w=w: torch.mm(x, w.T, out_dtype=torch.float32) for x, w in zip(xs, ws)])
        stock = timing([lambda x=x, w=w, a=a, b=b: biased_grouped_topk(torch.mm(x, w.T, out_dtype=torch.float32), bias,
                                                                         a, b, 1, 1, True, 1.0)
                        for x, w, a, b in zip(xs, ws, sw, si)])
        gemm = timing([lambda x=x, w=w: host._partials(x, w, True) for x, w in zip(xs, wss)])
        part, ks = host._partials(xs[0], wss[0], True)
        outs = [(torch.empty(m, TOPK, device="cuda"), torch.empty(m, TOPK, dtype=torch.int32, device="cuda"))
                for _ in range(COPIES)]
        topk = timing([lambda o=o: host._top16(part, ks, bias, 1.0, o, m) for o in outs])
        fused = timing([lambda x=x, w=w: router_sigmoid_bias_top16_e896_h7168(x, w, bias, shuffled=True)
                        for x, w in zip(xs, wss)])
        hbm_us = (wbytes + m * H * 2 + m * TOPK * 8) / (HBM_TBPS * 1e6)
        bound = hbm_us + 2 * floor
        tflops = 2 * m * E * H / fused / 1e6
        print(f"{m:4d} {stock_gemm:10.1f} {stock:7.1f} | {gemm:6.1f} {topk:6.1f} {fused:6.1f} | {hbm_us:5.2f} {bound:6.2f} "
              f"{wbytes / fused / 1e6:5.2f} {100 * bound / fused:6.1f} {tflops:6.0f} {100 * tflops / PEAK_BF16_TFLOPS:5.1f} "
              f"{stock / fused:6.2f}x", flush=True)
