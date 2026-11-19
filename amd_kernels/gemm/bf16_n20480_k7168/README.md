# bf16 GEMM, N 20480 x K 7168 (the TP8 lm_head vocabulary shard)

## What it does
`out [M, 20480] = x [M, 7168] @ W^T`, bf16 operands, fp32 sums in ascending K from a per-workgroup start block, one bf16
rounding: the TP8 vocabulary shard of the Kimi-K3.1 lm_head (`lm_head.weight` bf16 [163840, 7168], rank r holds rows
[20480 r, 20480 (r + 1)); W stays row-major, no load-time transform). One tensor serves the target lm_head and the
DSpark draft head (target-tied). No split-K, partials or output atomics. Contract (`reference.py`): the fp64 product
rounded once to bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3, deterministic; NaN / inf propagate as in an fp32
sum. 256 workgroups (one CU round) take 80 weight rows each and all rows of the activation (`band.BANDS`: eight M
bands up to 256 rows). Based on AITER's FlyDSL band-stream decode GEMM for this shape (`gemm_a16w16_band_gfx950`,
AMD-AGI/AKLR), with a runtime row count, bands above 128 rows, a device row map and the push epilogue.
Callers: `collectives/all_to_all_bf16_n20480_ranks8` (push modes), the Kimi-K3.1 logits and draft logits.

## Entry points
- `bf16_gemm_n20480_k7168`: the logits of 1-256 rows, graph-safe, no host sync, no allocation with `out`; with a device
  row map (`row_indices`) and `push=(exchange.send_args(), counts)` it sends the mapped rows straight to the rank that
  owns each (`counts` int32 [8], rows sorted by owner) or, with `counts=None`, to every rank, and publishes the
  exchange; the caller then runs the exchange's `wait` and reads the rows in place. An `amd_kernels.warm_up` entry: the
  first call of each mode (contiguous / owner / all) compiles its eight band variants.
- `supports`: whether a weight is the [20480, 7168] bf16 contiguous shard and there is no bias.

`host.py` picks the band by row count.

## Performance
One MI355X (shared node: minimum of 15 rounds of ten replays, graph replay of six calls over three weight copies, 880 MB),
us per call. `served` = AITER's `tgemm.mm` as vLLM calls it with VLLM_ROCM_USE_AITER=1 (the FlyDSL band-stream kernel up to 128
rows, a slow path above); `hipBLASLt` = `torch.nn.functional.linear`. Ceiling = max(293.6 MB at the measured 6.5 TB/s HBM read,
2 M N K at the library-measured 1.5 PFLOP/s bf16).

| M | 1 | 12 | 24 | 40 | 65 | 80 | 96 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| served | 42.8 | 43.1 | 43.7 | 45.5 | 51.4 | 52.7 | 54.1 | 59.4 | 98.7 | 103.4 | 143.7 |
| hipBLASLt | 51.8 | 52.3 | 47.9 | 59.4 | 65.4 | 62.4 | 58.1 | 65.2 | 74.1 | 80.3 | 91.3 |
| this entry | 43.0 | 43.1 | 43.9 | 45.7 | 49.7 | 50.9 | 53.7 | 59.4 | 73.1 | 78.2 | 92.6 |
| TB/s | 6.83 | 6.82 | 6.72 | 6.47 | 5.98 | 5.85 | 5.57 | 5.06 | 4.13 | 3.89 | 3.32 |
| ceiling | 45.2 | 45.3 | 45.4 | 45.5 | 45.7 | 45.8 | 46.0 | 46.3 | 46.5 | 46.8 | 50.1 |
| % of ceiling | 105 | 105 | 103 | 100 | 92 | 90 | 86 | 78 | 64 | 60 | 54 |

The contiguous call equals the served kernel to 3 % up to 128 rows (the same design; 65-80 rows: a BM 80 / BN 80 band, -3 %) and is
1.3-1.55x the served path above 128 rows (hipBLASLt-class). The push epilogue adds 4.3 us (owner) / 4.3-9.9 us (all ranks, 8x the
stores) when the destinations are local (a one-GPU test); the cross-GPU numbers are `collectives/all_to_all_bf16_n20480_ranks8`'s README.

Limiters (ablations, `s_memtime` stamps and sweeps in the report): up to ~40 rows the HBM weight stream; above, the L2 -> LDS stream of
the activation (every workgroup reads all of A: 1.8 MB / CU at M 128, 2.8 MB at 192, against 1.1 MB of weights) and the MFMA / LDS-read
loop (48 us alone at M 192 against 28 us at 1.98 PF), which the DMA pipeline overlaps only partly.

## Tests
- `python3 -m pytest tests/gemm/bf16_n20480_k7168 -q` (`test_push.py`: 8 GPUs)
- `python3 benchmarks/gemm/bf16_n20480_k7168/benchmark_gemm.py [M ...]`
