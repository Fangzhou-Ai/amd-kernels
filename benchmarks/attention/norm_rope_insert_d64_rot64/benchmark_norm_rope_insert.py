# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``qkv_norm_rope_insert`` and ``kv_norm_rope_insert_multilayer`` against the vLLM chains they
replace, at the served rows of a TP8 draft (12 q / 2 KV heads, head dim 64, 5 layers), with the bits compared.

- qkv (one draft layer, T = requests x k query rows): stock = the compiled draft layer's q / k norm + RoPE
  (``torch.compile`` of vLLM's native ``rms_norm`` and ``RotaryEmbedding.forward_static``, Inductor fuses them) and
  ``triton_reshape_and_cache_flash``.
- multi-layer (the context rows R = requests x (k + 1) of a step, 5 layers): stock =
  ``precompute_and_store_context_kv`` after its GEMM: the layer-major copy, the grouped ``rms_norm``, the position
  repeat, ``rotary_embedding`` and 5 ``triton_reshape_and_cache_flash`` (9 launches); for a bf16 cache also the
  one-launch Triton store of ``VLLM_K31_BEAT_CTXKV``.

One HIP graph holds ``--copies`` calls with their own inputs and disjoint slots. Minimum of 15 medians of 5 rounds of 10
replays, us per call. "floor" is one ``copy_`` launch moving the call's bytes (read + write, the same graph). The
first call of each variant is compared with the stock chain on the same inputs: bit-equal fraction of q and of the
cache elements written, and the largest difference in bf16 ulps / e4m3 codes. Needs vLLM on PYTHONPATH for the stock
columns.
"""

import argparse
import statistics

import torch

from amd_kernels.attention.norm_rope_insert_d64_rot64.kv_norm_rope_insert_multilayer import (
    kv_norm_rope_insert_multilayer,
)
from amd_kernels.attention.norm_rope_insert_d64_rot64.qkv_norm_rope_insert import (
    qkv_norm_rope_insert,
)

NQ, NKV, D, LAYERS, EPS = 12, 2, 64, 5, 1e-5
CS_ROWS = 1 << 20  # the draft's max_position_embeddings
QKV_ROWS = [("c1", 7, "bf16", 1280), ("c4", 10, "bf16", 1280), ("c16", 32, "fp8", 1152), ("c24", 52, "fp8", 1152),
            ("c32", 80, "fp8", 1152), ("c48", 192, "fp8", 1152), ("pad", 240, "fp8", 1152)]
ML_ROWS = [("c1", 8, "bf16", 1280), ("c4", 12, "bf16", 1280), ("c16", 40, "fp8", 1152), ("c24", 65, "fp8", 1152),
           ("c32", 100, "fp8", 1152), ("c48", 240, "fp8", 1152), ("prefill", 10240, "fp8", 1152)]  # fmt: skip


def timeit_graph(calls, replays=10, rounds=5, warmup=3, repeat=15):
    best = float("inf")
    for c in calls:
        c()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            c()
    torch.cuda.synchronize()
    for _ in range(repeat):
        events = [(torch.cuda.Event(True), torch.cuda.Event(True)) for _ in range(rounds)]
        for _ in range(warmup):
            graph.replay()
        for start, end in events:
            start.record()
            for _ in range(replays):
                graph.replay()
            end.record()
        torch.cuda.synchronize()
        best = min(best, statistics.median(s.elapsed_time(e) / (replays * len(calls)) * 1e3 for s, e in events))
    return best


def stock_ops():
    try:
        from vllm import _custom_ops as ops
        from vllm.ir.ops.layernorm import rms_norm
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash,
        )
    except ImportError as e:
        print(f"stock columns off: {e}")
        return None
    try:
        from vllm.models.kimi_k3.amd.ops.ctx_kv_store import store_context_kv
    except ImportError:
        store_context_kv = None
    native = rms_norm.impls["native"].impl_fn

    def norm_rope(qkv, positions, qw, kw, cs):
        t = qkv.shape[0]
        q, k, v = qkv.split([NQ * D, NKV * D, NKV * D], dim=-1)
        q = native(q.view(t, NQ, D), qw, EPS).view(t, NQ * D)
        k = native(k.view(t, NKV, D), kw, EPS).view(t, NKV * D)
        q, k = RotaryEmbedding.forward_static(positions, q, k, D, D, cs, True)
        return q, k, v

    return {"ops": ops, "norm_rope": torch.compile(norm_rope, dynamic=True, fullgraph=True),
            "insert": triton_reshape_and_cache_flash, "ctxkv": store_context_kv}


def cache_views(cache, kind):
    k, v = cache.transpose(1, 2).split(D, dim=-1)
    if kind == "fp8":
        k, v = k.view(torch.float8_e4m3fn), v.view(torch.float8_e4m3fn)
    return k, v


def compare(got, ref, written=None):
    """(bit-equal fraction, largest step) of bf16 or uint8 e4m3 tensors (on ``written`` elements)."""
    if written is not None:
        got, ref = got[written], ref[written]
    if got.dtype == torch.bfloat16:
        g, r = got.view(torch.int16).int(), ref.view(torch.int16).int()
        sm, mm = 0x8000, 0x7FFF
    else:
        g, r = got.view(torch.uint8).int(), ref.view(torch.uint8).int()
        sm, mm = 0x80, 0x7F
    if g.numel() == 0:
        return 1.0, 0
    same_sign = (g & sm) == (r & sm)
    step = torch.where(same_sign, ((g & mm) - (r & mm)).abs(), (g & mm) + (r & mm))
    return (g == r).float().mean().item(), int(step.max())


def rope_cache(gen):
    ang = torch.rand((CS_ROWS, 32), generator=gen, device="cuda") * 6.3
    return torch.cat([ang.cos(), ang.sin()], dim=1).bfloat16()


def make_cache(n_slots, page, kind):
    pages = -(-n_slots // page) + 1
    return torch.zeros((pages, NKV, page, 2 * D), dtype=torch.bfloat16 if kind == "bf16" else torch.uint8,
                       device="cuda")


def bench_qkv(label, n, kind, page, copies, stock, cs):
    gen = torch.Generator(device="cuda").manual_seed(n)
    row = (NQ + 2 * NKV) * D
    cache = make_cache(2 * n * copies, page, kind)
    perm = torch.randperm(cache.shape[0] * page, generator=gen, device="cuda")
    qw, kw = ((torch.randn(D, generator=gen, device="cuda") * 0.5 + 1).bfloat16() for _ in range(2))
    ks, vs = torch.tensor(1.0, device="cuda"), torch.tensor(1.0, device="cuda")
    inputs = [((torch.randn((n, row), generator=gen, device="cuda") * 2).bfloat16(),
               torch.randint(0, CS_ROWS, (n,), generator=gen, device="cuda"), perm[i * n:(i + 1) * n].contiguous(),
               torch.empty((n, NQ * D), dtype=torch.bfloat16, device="cuda")) for i in range(copies)]  # fmt: skip
    vkind = "auto" if kind == "bf16" else "fp8"
    kc, vc = cache_views(cache, kind)

    def ours(i):
        x, pos, slots, q = inputs[i]
        qkv_norm_rope_insert(x, pos, qw, kw, cs, EPS, NQ, NKV, q, cache, slots, ks, vs)

    def base(i):
        x, pos, slots, _ = inputs[i]
        sq, sk, sv = stock["norm_rope"](x, pos, qw, kw, cs)
        stock["insert"](sk.view(n, NKV, D), sv.view(n, NKV, D), kc, vc, slots, vkind, ks, vs)
        return sq

    out = {"ours": timeit_graph([lambda i=i: ours(i) for i in range(copies)])}
    moved = n * row * 2 + n * NQ * D * 2 + n * 2 * NKV * D * (2 if kind == "bf16" else 1)
    src = torch.empty(moved // 2, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    out["floor"] = timeit_graph([lambda: dst.copy_(src)] * copies)
    if stock is not None:
        out["stock"] = timeit_graph([lambda i=i: base(i) for i in range(copies)])
        cache.zero_()
        ours(0)
        ours_cache = cache.clone()
        cache.zero_()
        sq = base(0)
        written = torch.zeros(cache.shape, dtype=torch.bool, device="cuda")
        s = inputs[0][2]
        written[s // page, :, s % page] = True
        out["q"] = compare(inputs[0][3], sq)
        out["cache"] = compare(ours_cache, cache, written)
    report("qkv", label, n, kind, moved, out)


def bench_ml(label, n, kind, page, copies, stock, cs):
    gen = torch.Generator(device="cuda").manual_seed(n + 1)
    row = LAYERS * 2 * NKV * D
    caches = [make_cache(n * copies, page, kind) for _ in range(LAYERS)]
    perm = torch.randperm(caches[0].shape[0] * page, generator=gen, device="cuda")
    kw = (torch.randn((LAYERS, D), generator=gen, device="cuda") * 0.5 + 1).bfloat16()
    one = torch.tensor(1.0, device="cuda")
    ks = vs = [one] * LAYERS if kind == "fp8" else None
    inputs = [((torch.randn((n, row), generator=gen, device="cuda") * 2).bfloat16(),
               torch.randint(0, CS_ROWS, (n,), generator=gen, device="cuda"), perm[i * n:(i + 1) * n].contiguous())
              for i in range(copies)]  # fmt: skip
    vkind = "auto" if kind == "bf16" else "fp8"
    views = [cache_views(c, kind) for c in caches]

    def ours(i):
        kv, pos, slots = inputs[i]
        kv_norm_rope_insert_multilayer(kv, pos, kw, cs, caches, [slots] * LAYERS, ks, vs, EPS)

    def base(i):
        kv, pos, slots = inputs[i]
        ops = stock["ops"]
        all_kv = kv.view(n, LAYERS, 2, NKV, D).permute(2, 1, 0, 3, 4).contiguous()
        all_k, all_v = all_kv[0], all_kv[1]
        normed = torch.empty_like(all_k)
        ops.rms_norm(normed, all_k, kw, EPS)
        flat = normed.view(LAYERS * n, NKV * D)
        ops.rotary_embedding(pos.repeat(LAYERS), flat, None, D, cs, True)
        kf = flat.view(LAYERS, n, NKV, D)
        for lay in range(LAYERS):
            stock["insert"](kf[lay], all_v[lay], *views[lay], slots, vkind, one, one)

    def triton_ctxkv(i):
        kv, pos, slots = inputs[i]
        stock["ctxkv"](kv.view(n, LAYERS, 2, NKV, D), pos, kw, cs, EPS, True, caches, [slots] * LAYERS)

    out = {"ours": timeit_graph([lambda i=i: ours(i) for i in range(copies)])}
    moved = n * row * 2 + n * LAYERS * 2 * NKV * D * (2 if kind == "bf16" else 1)
    src = torch.empty(moved // 2, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    out["floor"] = timeit_graph([lambda: dst.copy_(src)] * copies)
    if stock is not None:
        out["stock"] = timeit_graph([lambda i=i: base(i) for i in range(copies)])
        if kind == "bf16" and stock["ctxkv"] is not None:
            out["ctxkv"] = timeit_graph([lambda i=i: triton_ctxkv(i) for i in range(copies)])
        s = inputs[0][2]
        mine = []
        for c in caches:
            c.zero_()
        ours(0)
        mine = [c.clone() for c in caches]
        for c in caches:
            c.zero_()
        base(0)
        written = torch.zeros(caches[0].shape, dtype=torch.bool, device="cuda")
        written[s // page, :, s % page] = True
        res = [compare(m, c, written) for m, c in zip(mine, caches)]
        out["cache"] = (min(r[0] for r in res), max(r[1] for r in res))
    report("multilayer", label, n, kind, moved, out)


def report(op, label, n, kind, moved, out):
    s = f"{op:10s} {label:8s} rows={n:6d} {kind:4s}: ours {out['ours']:6.2f} us ({moved / out['ours'] / 1e3:5.0f} GB/s)"
    s += f" | floor {out['floor']:5.2f}"
    if "stock" in out:
        s += f" | stock {out['stock']:6.2f} ({out['stock'] / out['ours']:4.1f}x)"
    if "ctxkv" in out:
        s += f" | triton ctxkv {out['ctxkv']:5.2f}"
    if "q" in out:
        s += f" | q bit-equal {out['q'][0]:.6f} (max {out['q'][1]} ulp)"
    if "cache" in out:
        s += f" | cache bit-equal {out['cache'][0]:.6f} (max {out['cache'][1]})"
    print(s, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=20, help="calls (inputs) per graph")
    parser.add_argument("--both-dtypes", action="store_true", help="every row count with bf16 and fp8 caches")
    args = parser.parse_args()
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        raise RuntimeError("requires gfx950")
    stock = stock_ops()
    cs = rope_cache(torch.Generator(device="cuda").manual_seed(0))
    with torch.inference_mode():
        for rows, fn in ((QKV_ROWS, bench_qkv), (ML_ROWS, bench_ml)):
            for label, n, kind, page in rows:
                for k in (("bf16", "fp8") if args.both_dtypes else (kind,)):
                    fn(label, n, k, page, args.copies, stock, cs)


if __name__ == "__main__":
    main()
