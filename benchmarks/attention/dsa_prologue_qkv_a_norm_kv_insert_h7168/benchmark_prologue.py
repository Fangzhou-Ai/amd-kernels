# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``qkv_a_norm_kv_insert`` and ``indexer_prep`` against the unfused ops vLLM runs today
(the multi-stream DSA wrapper's chain: AITER ``rms_norm`` x2 + ``concat_and_cache_mla``; ``layer_norm``, Triton
``store_keys``, ``fill_``, the fp32 weight product, the starts clamp and ``quantize_queries``; also the two-launch
Triton ``store_keys_and_reset`` + ``quantize_queries_and_weights`` of ``VLLM_K3_FUSED_INDEXER_PREP``).

One HIP graph holds ``--copies`` launches with their own inputs and disjoint cache slots, so the inputs of all copies
exceed the 256 MB MALL together and each launch reads from HBM. Median of 5 rounds of 10 replays, us per call. The
first launch of each variant is checked against ``reference.py`` before timing. The roofline columns count the bytes a
call reads and writes (HBM 8 TB/s, 80% = 6.4 TB/s achievable) and the latency floor: one elementwise kernel that
reads and writes the same bytes in the same harness (a 1-element kernel takes about 1.6 us per launch in a graph).
"""

import argparse
import statistics

import torch
import torch.nn.functional as F
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168 import reference as R
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.indexer_prep import indexer_prep
from amd_kernels.attention.dsa_prologue_qkv_a_norm_kv_insert_h7168.qkv_a_norm_kv_insert import qkv_a_norm_kv_insert

EPS = 1e-6
TOPK = 2048
HBM_TBS = 8.0
ACHIEVABLE = 0.8
SERVED_M = [12, 24, 40, 65, 80, 128, 192]
WEIGHT_SCALE = 32**-0.5 * 128**-0.5
PAGE_STRIDE = 41984  # the merged MLA page: 64 x 656


def timeit_graph(calls, replays=10, rounds=5, warmup=3):
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
    for _ in range(warmup):
        graph.replay()
    for start, end in events:
        start.record()
        for _ in range(replays):
            graph.replay()
        end.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) / (replays * len(calls)) * 1000 for s, e in events)


def make_qkv(m, copies, gen):
    dev = "cuda"
    xs = [(torch.randn((m, 2112), generator=gen, device=dev) * 2).bfloat16() for _ in range(copies)]
    qw = (torch.randn(1536, generator=gen, device=dev) * 0.3 + 1).bfloat16()
    kw = (torch.randn(512, generator=gen, device=dev) * 0.3 + 1).bfloat16()
    pages = copies * ((m + 63) // 64) + 1
    cache = torch.zeros((pages, 64, 656), dtype=torch.uint8, device=dev)
    slots = [torch.arange(m, device=dev, dtype=torch.int64) + i * ((m + 63) // 64) * 64 for i in range(copies)]
    qo = [torch.empty((m, 1536), dtype=torch.bfloat16, device=dev) for _ in range(copies)]
    ko = [torch.empty((m, 512), dtype=torch.bfloat16, device=dev) for _ in range(copies)]
    return xs, qw, kw, cache, slots, qo, ko


def bench_qkv(m, copies, gen):
    from vllm import _custom_ops as ops
    from vllm.models.common.ops import fused_q_kv_rmsnorm
    import aiter

    xs, qw, kw, cache, slots, qo, ko = make_qkv(m, copies, gen)
    one = torch.ones(1, device="cuda")
    ours = [
        (lambda i=i: qkv_a_norm_kv_insert(xs[i], qw, kw, EPS, qo[i], ko[i], cache, slots[i])) for i in range(copies)
    ]

    def stock(i):
        aiter.rms_norm(xs[i][:, :1536], qw, EPS)
        kv_c = aiter.rms_norm(xs[i][:, 1536:2048], kw, EPS)
        ops.concat_and_cache_mla(
            kv_c, xs[i][:, 2048:2112], cache, slots[i], kv_cache_dtype="fp8_ds_mla", scale=one, skip_scale=False
        )

    def stock_fused_norm(i):
        _, kv_c = fused_q_kv_rmsnorm(xs[i][:, :1536], xs[i][:, 1536:2048], qw, kw, EPS)
        ops.concat_and_cache_mla(
            kv_c, xs[i][:, 2048:2112], cache, slots[i], kv_cache_dtype="fp8_ds_mla", scale=one, skip_scale=False
        )

    before = cache.clone()
    ours[0]()
    rq, rkv, rc, wm = R.qkv_a_norm_kv_insert_reference(xs[0], qw, kw, EPS, before, slots[0])
    R.check_norm(qo[0], rq, "q")
    R.check_norm(ko[0], rkv, "kv")
    R.check_kv_cache(cache, rc, before, wm, "cache")
    t_ours = timeit_graph(ours)
    t_stock = timeit_graph([(lambda i=i: stock(i)) for i in range(copies)])
    t_fused = timeit_graph([(lambda i=i: stock_fused_norm(i)) for i in range(copies)])
    byt = m * (2112 * 2 + 2048 * 2 + 656) + 1536 * 2 + 512 * 2
    return t_ours, t_stock, t_fused, byt


def make_index(m, copies, gen, rows_out):
    dev = "cuda"
    mk = lambda *s: (torch.randn(s, generator=gen, device=dev) * 2).bfloat16()  # noqa: E731
    qs_ = [mk(m, 32, 128) for _ in range(copies)]
    ks = [mk(m, 128) for _ in range(copies)]
    ws = [mk(m, 32) for _ in range(copies)]
    lw = (torch.randn(128, generator=gen, device=dev) * 0.3 + 1).bfloat16()
    lb = (torch.randn(128, generator=gen, device=dev) * 0.1).bfloat16()
    pages = copies * ((m + 63) // 64) + 1
    base = torch.zeros(pages * PAGE_STRIDE, dtype=torch.uint8, device=dev)
    cache = base.as_strided((pages, 64, 68), (PAGE_STRIDE, 68, 1))
    slots = [torch.arange(m, device=dev, dtype=torch.int64) + i * ((m + 63) // 64) * 64 for i in range(copies)]
    outs = [torch.zeros((rows_out, TOPK), dtype=torch.int32, device=dev) for _ in range(copies)]
    qp = [torch.empty((m, 32, 64), dtype=torch.uint8, device=dev) for _ in range(copies)]
    qsc = [torch.empty((m, 1, 4, 16, 4), dtype=torch.uint8, device=dev) for _ in range(copies)]
    wo = [torch.empty((m, 32), dtype=torch.float32, device=dev) for _ in range(copies)]
    return qs_, ks, ws, lw, lb, base, cache, slots, outs, qp, qsc, wo


def bench_index(m, copies, gen):
    from vllm.models.kimi_k3.amd.ops.fp4_indexer import (
        quantize_queries,
        quantize_queries_and_weights,
        store_keys,
        store_keys_and_reset,
    )

    qs_, ks, ws, lw, lb, base, cache, slots, outs, qp, qsc, wo = make_index(m, copies, gen, m)
    starts = torch.arange(m + 1, device="cuda", dtype=torch.int32)
    lw32, lb32 = lw.float(), lb.float()
    ours = [
        (
            lambda i=i: indexer_prep(
                qs_[i], ks[i], ws[i], lw, lb, EPS, slots[i], cache, outs[i], WEIGHT_SCALE, qp[i], qsc[i], wo[i]
            )
        )
        for i in range(copies)
    ]

    def stock(i):
        ik = F.layer_norm(ks[i], (128,), lw, lb, EPS)
        store_keys(ik, cache, slots[i])
        outs[i].fill_(-1)
        weights = ws[i].float() * WEIGHT_SCALE
        rel = starts.clamp(0, m)
        quantize_queries(qs_[i])
        return weights, rel

    def stock_fused_prep(i):
        store_keys_and_reset(ks[i], cache, slots[i], outs[i], lw32, lb32, EPS)
        quantize_queries_and_weights(qs_[i], ws[i], WEIGHT_SCALE, starts, 0, m)

    before = base.clone()
    ours[0]()
    bc = before.as_strided(cache.shape, cache.stride())
    rp, rs, rw, rt, rc, wm = R.indexer_prep_reference(ks[0], lw, lb, EPS, slots[0], qs_[0], ws[0], WEIGHT_SCALE, outs[0], bc)
    assert torch.equal(qp[0], rp) and torch.equal(qsc[0], rs) and torch.equal(wo[0], rw) and bool((outs[0] == -1).all())
    R.check_index(cache, rc, bc, wm, "cache")
    t_ours = timeit_graph(ours)
    t_stock = timeit_graph([(lambda i=i: stock(i)) for i in range(copies)])
    t_fused = timeit_graph([(lambda i=i: stock_fused_prep(i)) for i in range(copies)])
    byt = m * (8192 + 256 + 64 + 2048 + 256 + 128 + 68 + 8192) + 3 * 128 * 2
    return t_ours, t_stock, t_fused, byt


def latency_floor(byt, copies):
    """One elementwise kernel that reads and writes the call's bytes, in the same graph harness."""
    xs = [torch.zeros(byt // 4, dtype=torch.bfloat16, device="cuda") for _ in range(copies)]
    return min(timeit_graph([(lambda x=x: x.add_(1)) for x in xs]) for _ in range(3))


def row(name, m, launches_stock, launches_fused, t_ours, t_stock, t_fused, byt, copies):
    hbm = byt / (HBM_TBS * ACHIEVABLE * 1e6)
    floor = latency_floor(byt, copies)
    print(
        f"{name:18s} {m:4d} | {byt / 1e6:5.2f} MB, {hbm:4.2f} us at 6.4 TB/s | ours {t_ours:5.2f} us"
        f" {byt / t_ours / 1e6:4.2f} TB/s {100 * byt / t_ours / 1e6 / HBM_TBS:4.1f}%HBM"
        f" | same-bytes copy kernel {floor:4.2f} us ({t_ours / floor:4.2f}x)"
        f" | stock {t_stock:5.2f} us ({launches_stock} launches, {t_stock / t_ours:4.2f}x)"
        f" | stock-fused {t_fused:5.2f} us ({launches_fused} launches, {t_fused / t_ours:4.2f}x)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copies", type=int, default=100)
    ap.add_argument("--rows", type=int, nargs="*", default=SERVED_M)
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(0)
    print("kernel               M  | bytes per call | ours, achieved | vs the latency floor | vs the stock unfused chains")
    for m in args.rows:
        row("qkv_a_norm_kv_ins", m, 3, 2, *bench_qkv(m, args.copies, gen), args.copies)
    for m in args.rows:
        row("indexer_prep", m, 6, 2, *bench_index(m, args.copies, gen), args.copies)


if __name__ == "__main__":
    main()
