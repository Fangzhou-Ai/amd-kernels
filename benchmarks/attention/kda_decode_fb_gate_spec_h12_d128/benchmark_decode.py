# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Graph-replay timing of ``kda_decode_fb_gate_spec`` on gfx950 against the stock two-launch path.

One HIP graph holds ``--copies`` launches over distinct inputs (state slots, conv state, f_b weights, activations; the
set is larger than the 256 MB MALL), median of 5 rounds in us per launch. Rows are 6-token sequences (the last one
shorter). Arms: ``fused`` (this kernel) and ``stock``: vLLM's ``kimi_k3_kda_fb_proj`` Triton GEMM (kernel copied from
vllm/models/kimi_k3/amd/ops/kda_fb_gemm.py, 16 x 32 tiles, 2 warps) followed by AITER ``fused_recurrent_kda`` in the
configuration of ``kda_fused_recurrent_spec_decode``. Roofline: HBM bytes of the state (read once per sequence,
snapshot written per token) plus conv state, activations and output, against 8 TB/s (80% achievable).

  python3 benchmarks/attention/kda_decode_fb_gate_spec_h12_d128/benchmark_decode.py --ms 12,24,40,65,80,128,192
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from aiter.ops.triton.attention.kda import fused_recurrent_kda
from amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128 import kda_decode_fb_gate_spec
from amd_kernels.attention.kda_decode_fb_gate_spec_h12_d128.reference import (
    COL_BETA,
    COL_FA,
    COL_GATE,
    D,
    LP,
    NH,
    kda_decode_fb_gate_spec_reference,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.attention.kda_decode_fb_gate_spec_h12_d128.inputs import SPEC_TOKENS, clone_inputs, make_inputs  # noqa: E402

HBM_PEAK = 8.0e12
HBM_ACHIEVABLE = 0.8
MIN_SET_BYTES = 700_000_000


@triton.jit
def _fb_gemm(x_ptr, w_ptr, out_ptr, num_tokens, stride_x, stride_w, stride_out, N: tl.constexpr, K: tl.constexpr,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, K)
    x = tl.load(x_ptr + offs_m[:, None] * stride_x + offs_k[None, :], mask=(offs_m < num_tokens)[:, None], other=0.0)
    w = tl.load(w_ptr + offs_n[:, None] * stride_w + offs_k[None, :], mask=(offs_n < N)[:, None], other=0.0)
    acc = tl.dot(x, tl.trans(w))
    tl.store(out_ptr + offs_m[:, None] * stride_out + offs_n[None, :], acc.to(out_ptr.dtype.element_ty),
             mask=(offs_m < num_tokens)[:, None] & (offs_n < N)[None, :])


def stock_fb(x, weight):
    n_tok, k = x.shape
    out = torch.empty((n_tok, weight.shape[0]), dtype=x.dtype, device=x.device)
    _fb_gemm[(triton.cdiv(weight.shape[0], 32), triton.cdiv(n_tok, 16))](
        x, weight, out, n_tok, x.stride(0), weight.stride(0), out.stride(0), N=weight.shape[0], K=k, BLOCK_M=16,
        BLOCK_N=32, num_warps=2)
    return out


def stock_call(a):
    p = a["proj"]
    q, k, v = (p[:, i * LP:(i + 1) * LP].reshape(1, -1, NH, D) for i in range(3))
    og = p[:, COL_GATE:COL_GATE + LP].reshape(1, -1, NH, D)
    beta = p[:, COL_BETA:COL_BETA + NH].reshape(1, -1, NH)
    out = a["out"].unsqueeze(0)

    def call():
        g1 = stock_fb(p[:, COL_FA:COL_FA + D], a["f_b_weight"]).view(1, -1, NH, D)
        fused_recurrent_kda(
            q=q, k=k, v=v, g=g1, beta=beta, A_log=a["A_log"], dt_bias=a["dt_bias"], lower_bound=a["lower_bound"],
            initial_state=a["state"], cu_seqlens=a["cu_seqlens"], ssm_state_indices=a["state_indices"],
            num_accepted_tokens=a["num_accepted"], use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True, pad_slot_guard=True, conv_state=a["conv_state"],
            conv_weight=a["conv_weight"], out_gate=og, norm_weight=a["norm_weight"], norm_eps=a["norm_eps"], out=out)

    return call


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
    return statistics.median(s.elapsed_time(e) * 1e3 / (replays * len(calls)) for s, e in events)


def hbm_bytes(rows, n_seq):
    state = 12 * D * D * 4
    conv = 3 * LP * 2 * (3 + rows / n_seq + 2)
    acts = (3 * LP + LP + D + NH) * 2 + LP * 2
    return n_seq * state + rows * state + n_seq * conv + rows * acts + LP * D * 2


def check(a, rows):
    b = clone_inputs(a)
    c = clone_inputs(a)
    kda_decode_fb_gate_spec_reference(**b)
    kda_decode_fb_gate_spec(**c)
    torch.cuda.synchronize()
    cos = torch.nn.functional.cosine_similarity(c["out"].float().flatten(), b["out"].float().flatten(), dim=0)
    assert cos > 0.9999 and torch.equal(c["conv_state"], b["conv_state"]), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="12,24,40,65,80,128,192")
    ap.add_argument("--tokens", type=int, default=SPEC_TOKENS)
    ap.add_argument("--max-copies", type=int, default=16)
    args = ap.parse_args()
    print(f"{'M':>4} {'seqs':>4} {'MB':>7} {'fused us':>9} {'TB/s':>6} {'%80%HBM':>8} {'stock us':>9} {'speedup':>8}")
    for rows in (int(m) for m in args.ms.split(",")):
        first = make_inputs(rows, args.tokens, seed=0)
        n_seq = first["cu_seqlens"].numel() - 1
        per_set = first["state"].numel() * 4
        copies = min(args.max_copies, max(2, -(-MIN_SET_BYTES // per_set)))
        sets = [first] + [make_inputs(rows, args.tokens, seed=i) for i in range(1, copies)]
        check(sets[0], rows)
        fused = timeit_graph([lambda a=a: kda_decode_fb_gate_spec(**a) for a in sets])
        stock = timeit_graph([stock_call(a) for a in sets])
        mb = hbm_bytes(rows, n_seq)
        tbs = mb / (fused * 1e-6) / 1e12
        print(f"{rows:>4} {n_seq:>4} {mb / 1e6:>7.1f} {fused:>9.1f} {tbs:>6.2f} {100 * tbs * 1e12 / (HBM_PEAK * HBM_ACHIEVABLE):>7.0f}% "
              f"{stock:>9.1f} {stock / fused:>7.2f}x", flush=True)
        del sets, first


if __name__ == "__main__":
    main()
