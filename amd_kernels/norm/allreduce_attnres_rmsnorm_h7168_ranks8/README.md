# Eight-rank all-reduce + AttnRes mix + RMSNorm, hidden 7168

## What it does

`AttnResComm.allreduce_attnres_rmsnorm` fuses what a Kimi-K3.1 decode layer runs after every TP8 row-parallel GEMM
(`o_proj`, the MoE final reduce, the dense MLP): the all-reduce of the bf16 partial, the AttnRes block mix of the running
prefix (`attn_res`) and the RMSNorm that follows, in one launch. Partial bf16 `[M, 7168]`, prefix bf16 `[M, 7168]`
updated in place, bank bf16 `[M, <=8, 7168]` (row `block_write_idx` written when a block ends), weights bf16 `[7168]`,
output bf16 `[M, 7168]`; M 1..256, one workgroup per row. `reference.py` is the contract: the eight partials are added
in rank order in fp32 and rounded once to bf16, the prefix add rounds to bf16, the mix and the norm are fp32 with one bf16
rounding of the output. The prefix out and the bank row are bit for bit the reference, the output is within one bf16 ulp
of the fp64 evaluation; deterministic, no atomic adds into outputs, no quantization (the consumers are bf16 GEMMs).
Optional forms: no prefix (a block's first source), no bank (a plain all-reduce + add + RMSNorm), no output norm,
`reduce=False` for an already reduced delta (no communication), `out_packed` for the tile-packed output layout of the
MFMA GEMMs.

The exchange is a two-phase reduce-scatter / all-gather by posted writes into persistent uncached mailboxes with the data
as its own signal (module docstring of `kernel.py`): nothing is registered per graph, no flags, no host work, no staging.

## Entry points

- `AttnResComm`: the communicator, a collective on the eight TP ranks (CPU group needed); 14.7 MiB of mailboxes per rank.
- `AttnResComm.allreduce_attnres_rmsnorm` / `allreduce_attnres_rmsnorm`: the fused call -> (out, prefix out).
- `AttnResComm.close`: collective, after the last call completed.

One communicator per stream; every rank makes the same calls in the same order with the same row counts. The variants
(prefix, output norm, bank, reduce, packed output) are the entry's warm-up modes (`amd_kernels.warm_up`).

## Performance

MI355X, 8 ranks, graphs of 27-100 distinct input sets (larger than the LLC), min of 9 rounds, MAX over ranks, us per
call; the call is a pre-MLP one (prefix + delta, bank of 4 rows, output norm). Baseline: AITER custom all-reduce, then
vLLM's Triton AttnRes (regtile) launch. Floor: the measured 8-GPU all-reduce floor (5 us + 2 * 7/8 * bytes / 400 GB/s)
plus 1.2 us for the prefix add, norm and store. The measured xGMI all-to-all push ceiling is 355 GB/s
per GPU (`ceilings.py --mode bw`); the one-way latency between two ranks is 0.77 us (`--mode ping`).

| M | stock | all-reduce alone | fused | speedup | % of floor | xGMI % of 355 GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 11.89 | 5.90 | 9.86 | 1.21 | 70 | 9 |
| 24 | 13.00 | 6.77 | 10.56 | 1.23 | 73 | 16 |
| 40 | 14.92 | 8.03 | 11.87 | 1.26 | 73 | 24 |
| 65 | 17.17 | 10.07 | 13.98 | 1.23 | 74 | 33 |
| 80 | 19.21 | 11.13 | 15.41 | 1.25 | 73 | 37 |
| 128 | 23.18 | 14.42 | 20.11 | 1.15 | 71 | 45 |
| 192 | 29.37 | 19.59 | 25.69 | 1.14 | 71 | 53 |

The limiter is the dependent chain, not a bandwidth: two phases of write -> flight -> poll (about 2 x 1.3 us) around a
one-wave reduce, ranks that start up to 1.3 us apart (the same skew the stock all-reduce pays), and a 1.2 us tail; HBM
use is 4-25% of its ceiling. The protocol is in the module docstring of `kernel.py`.

## Tests

```bash
python3 -m pytest tests/norm/allreduce_attnres_rmsnorm_h7168_ranks8 -q                 # 8 GPUs
python3 benchmarks/norm/allreduce_attnres_rmsnorm_h7168_ranks8/benchmark_allreduce.py   # 8 GPUs
python3 benchmarks/norm/allreduce_attnres_rmsnorm_h7168_ranks8/ceilings.py --mode ping  # 8 GPUs
```
