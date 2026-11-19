# bf16 GEMM, N 20480 x K 7168 (the TP8 lm_head vocabulary shard)

## What it does
`out [M, 20480] = x [M, 7168] @ W^T`, bf16 operands, fp32 sums, one bf16 rounding: the TP8 vocabulary shard of the Kimi-K3.1
lm_head (`lm_head.weight` bf16 [163840, 7168], rank r holds rows [20480 r, 20480 (r + 1)); W stays row-major, no load-time
transform). One tensor serves the target lm_head and the DSpark draft head (target-tied). Contract (`reference.py`): the fp64
product rounded once to bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3, deterministic bits, a row's bits independent of the
batch size and the row's position within one kernel band; NaN / inf propagate as in an fp32 sum. No atomics on the output.
Up to 96 rows the band-stream kernel (`band.py`, adapted from AITER's FlyDSL decode GEMM): 256 workgroups, 80 weight rows and all
activation rows each, fixed K order from a staggered start. From 97 rows the split-K kernel (`splitk.py`): 128 pairs of
workgroups on one XCD, each pair owning 160 columns and each member one half of K, so the activation traffic of a workgroup
halves; the two partial tiles meet through a fixed-order L2 exchange (workspace + self-clearing flags), member order 0, 1 for every row.

## Entry points
- `bf16_gemm_n20480_k7168`: the logits of 1-256 rows, graph-safe, no host sync; `row_indices` fuses the hidden-row gather (device
  int64 [M]); `push=(exchange.send_args(), counts)` sends the rows straight to the rank that owns each (counts int32 [8], rows
  sorted by owner) or, with `counts=None`, to every rank, and publishes the exchange (the caller runs `RowAllToAll.wait`).
  An `amd_kernels.warm_up` entry: the first call of each mode (contiguous / gather / owner / all) compiles its band variants.
- `supports`: whether a weight is the [20480, 7168] bf16 contiguous shard and there is no bias.
- `host.band_of`: the kernel configuration of a row count (the dispatch is `host._kernel`); the split-K bands share one workspace
  per device, so two split-K launches must not run concurrently on different streams.
- `collectives/all_to_all_bf16_n20480_ranks8`: `RowAllToAll.gemm_exchange` picks between the GEMM's own push and a local write plus
  one contiguous push by row count.

## Performance
One MI355X (`gpurun 3`, minimum of 15 rounds of ten replays, graph replay of six calls over three weight copies, 880 MB), us per
call. `served` = AITER's `tgemm.mm` as vLLM calls it (the FlyDSL band-stream kernel up to 128 rows, a slow path above);
`previous` = this entry before the split-K bands. Ceiling = max(293.6 MB at 6.5 TB/s measured HBM read, 2 M N K at the 1.5 PF
measured library bf16 GEMM).

| M | 1 | 12 | 24 | 40 | 65 | 80 | 96 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| served | 43.1 | 43.3 | 43.9 | 45.9 | 51.6 | 53.3 | 55.2 | 60.0 | 100.0 | 103.9 | 144.8 |
| previous | 43.1 | 43.4 | 44.1 | 46.2 | 50.3 | 51.6 | 54.1 | 60.4 | 74.4 | 80.6 | 95.1 |
| this entry | 43.2 | 43.3 | 44.0 | 45.9 | 50.4 | 51.6 | 53.9 | 57.6 | 65.4 | 66.2 | 85.4 |
| ceiling | 45.2 | 45.3 | 45.4 | 45.5 | 45.7 | 45.8 | 46.0 | 46.3 | 46.5 | 46.8 | 50.1 |
| % of ceiling | 104 | 104 | 103 | 99 | 91 | 89 | 85 | 80 | 71 | 71 | 59 |
| memory-side floor of the design | 42 | 42 | 42 | 43 | 43 | 44 | 46 | 47 | 49 | 50 | 61 |

The floor row is a kernel that only streams the same operands through the same rings (no math, no exchange; graph-timed, launch
included): the weight stream at HBM speed plus the activation re-read; the L2 -> LDS path carries both, so above ~64 rows the
activation traffic adds to the weight time. The split-K kernel reaches 92-97 % of that floor at 128-192 rows; the rest is the
dispatch ramp and the end of the kernel (partial exchange 2-3 us, output store drain 1.5 us) and the 6-8 us spread of the
CUs' finishing times (`analysis/kernels/bf16_n20480_k7168.md`: timestamps, ablations, every attempt that did not help).

## Tests
- `python3 -m pytest tests/gemm/bf16_n20480_k7168 -q` (`test_push.py`: 8 GPUs, `gpurun ranks8`)
- `python3 benchmarks/gemm/bf16_n20480_k7168/benchmark_gemm.py [M ...]`
