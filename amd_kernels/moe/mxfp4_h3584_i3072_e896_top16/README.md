# MXFP4 routed experts, H3584 I384 E896 top16 (gfx950)

## What it does

The 896 routed experts of one TP8 rank of Kimi-K3.1 (moe_intermediate 3072 / 8 = 384 per rank, latent hidden 3584, top-16,
all experts on every rank), FlyDSL only. Input: the MXFP4 activations (e2m1 `[M, 1792]` + e8m0 `[M, 112]` per 32,
`quantize_activations` or the latent down-projection's epilogue) and the router's BM=16 sort. Per route: gate/up GEMM1
(SiTUv2 `beta 4`, `linear_beta 25`, no clamp; the activation re-quantized to MXFP4 with aiter's rule) and down GEMM2, the
router weight applied once; a token's output = its 16 slot rows summed in slot order in fp32, rounded once to BF16.
Weights are read in vLLM's aiter layouts (`shuffle_weight`, `e8m0_shuffle`), nothing is re-laid-out. Contract
(`reference.py`): quant bit for bit, `moe_ref` within rel. L2 2e-2 (observed 2.7e-3; aiter's own is 5.5e-3 from bf16 atomic adds);
no atomics into outputs, two runs bit for bit, graph replay = eager. M 1-192 (one launch per call).

## Entry points

- `MoeWeights.from_shuffled(w13, s13, w2, s2)`: views of the layer's tensors as vLLM holds them (`from_checkpoint` shuffles
  row-major ones); once, before graph capture.
- `mxfp4_moe_experts(x_codes, x_scales, weights, sort, out=None, fused=True)`: GEMM1 + GEMM2 in one launch (the
  GEMM2 tiles stream their weights while GEMM1 finishes and wait on their m-block's counter), then the slot sum;
  `fused=False` runs three launches (same bits). An `amd_kernels.warm_up` entry: the first call compiles everything.
- `Sort.from_aux(aux)`: the router entry's `make_aux` tuple (`moe/router_sigmoid_bias_top16_e896_h7168`); `moe_sort`
  builds it with aiter for tests. `quantize_activations(x)`, `mxfp4_moe(x, weights, ids, w)`: bf16 in, tests and benchmarks.

## Performance

One MI355X, 3 layers of weights (5.9 GB), CUDA graph replay, interleaved min of 40 per layer, us. Ceiling: the measured
nontemporal HBM read, 7.0 TB/s (`analysis/hw_ceilings.md`; 6.4-6.7 plain loads). TB/s = unique experts x 2.19 MB + the
activations, slots and output. Counters (rocprofv3 FETCH_SIZE, best of 12 replays): the one-launch kernel fetches 6.5 TB/s at
M 12, 6.9 at 24, 7.0-7.1 at M 40-192 (MFMA busy 2.6-4%); the table's min is higher than quiet-GPU time because other jobs share the card.

PERF_TABLE

## Tests

```bash
python3 -m pytest tests/moe/mxfp4_h3584_i3072_e896_top16 -q   # reference, aiter, determinism, fused = unfused, warm-up
python3 benchmarks/moe/mxfp4_h3584_i3072_e896_top16/benchmark_experts.py [M ...]
```
