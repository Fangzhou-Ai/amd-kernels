# bf16 latent down projection + MXFP4 quant, K 7168 x N 3584 (the routed experts' input)

## What it does
`codes, scales = mxfp4(bf16(x [M, 7168] @ W^T))` for the MoE layer's replicated latent `routed_expert_down_proj`
(bf16 W [3584, 7168], 51 MB, M 1-192; above 192 rows launches of 192). bf16 operands, fp32 sums (K split in 4 slices
summed in a fixed order, no output atomics, bitwise repeatable), one bf16 rounding, then the quant the aiter a4w4 MoE
takes: per row, 32 columns share an e8m0 scale `ceil(log2(amax / 6))` floored at 92, RNE e2m1 codes saturating at 6.
Output layout = `per_1x32_mx_quant_hip(y, quant_dtype=fp4x2, shuffle=False)`: codes `[M, 1792]` fp4x2, scales
`[M, 112]` e8m0, row-major (what `mxfp4_moe_sort_fwd` / `prequantized_a1` consume); optionally the bf16 `y` too. Contract
(`reference.py`): the codes and scales of the kernel's own bf16 `y` equal `quant_reference` and aiter's quant bit for
bit; `y` differs from the fp64 product only by sum order (cosine >= 0.99999). Weight layout `row` (the checkpoint's) or
`shuf` (bit-exact permutation once at load, one 1 KB DMA per fragment).

## Entry points
- `bf16_latent_down_mxfp4_quant_k7168_n3584`: the projection and quant of any row count, graph-safe, no host sync; an
  `amd_kernels.warm_up` entry (the first call of each mode, layout x bf16 output, compiles its 12 row tiles). One call
  at a time per stream (partial slabs and tile tickets live in a per-stream workspace).
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout.
- `supports`: whether a weight is the bf16 contiguous [3584, 7168] one and there is no bias.

`host.py` picks the row tile (16 rows per fragment) and the K split (`KS_BY_MT`).

## Performance
One MI355X, exclusive GPU, graph replay of 12 calls over 12 distinct weight copies (617 MB > the MALL) and activation sets,
median us. Stock = vLLM `rocm_unquantized_gemm_impl` (VLLM_ROCM_USE_AITER=1: aiter tuned hgemm) + aiter
`per_1x32_mx_quant_hip`. Floor = the 51.4 MB weight stream at the measured nontemporal HBM rate: 1.57 us graph node +
7.4 us (6.95 TB/s) = 9.0 us (a pure-read skeleton of the same grid measures 9.1-9.6).

| M | 1 | 4 | 12 | 24 | 40 | 65 | 80 | 128 | 192 |
|---|---|---|---|---|---|---|---|---|---|
| stock gemm + quant | 11.6 | 12.9 | 14.3 | 14.7 | 16.0 | 19.9 | 20.4 | 23.3 | 33.8 |
| fused | 11.9 | 11.9 | 11.7 | 12.2 | 13.3 | 15.0 | 15.6 | 19.7 | 25.8 |
| % of the 9.0 us floor | 76 | 76 | 77 | 74 | 68 | 60 | 58 | 46 | 35 |
| speedup | 0.97x | 1.09x | 1.23x | 1.20x | 1.20x | 1.32x | 1.31x | 1.19x | 1.31x |

In-kernel timestamps (`s_memrealtime`, M 12): the weight stream (first block 1.2 us, last loop end 8.0 us median) runs at
the HBM ceiling; the epilogue after the slowest main loop is 1.9 us (K halves 0.16, slab stores + drain 0.45, ticket 0.6,
slab loads + LDS tile 0.44, quant + stores 0.3). Above M ~ 40 the limiter is the per-CU load rate (~45-50 GB/s: weights
229 KB + activation 28 blocks x 2 MT KB per workgroup) and the 11 MB slab round trip at M 192.

## Tests
- `python3 -m pytest tests/gemm/bf16_latent_down_mxfp4_quant_k7168_n3584 -q`
- `python3 benchmarks/gemm/bf16_latent_down_mxfp4_quant_k7168_n3584/benchmark_down_quant.py [M ...]`
