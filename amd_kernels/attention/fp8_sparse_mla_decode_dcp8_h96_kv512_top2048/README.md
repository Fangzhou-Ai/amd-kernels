# fp8_sparse_mla_decode_dcp8_h96_kv512_top2048

## What it does

The core attention of a DSA layer at DCP8, per rank: 96 latent heads (the DCP-gathered q, bf16 `[rows, 96, 576]` =
512 latent + 64 rope dims) attend this rank's keys of each row's global top-2048, read from the fp8_ds_mla cache
(656 B records: 512 e4m3 | 4 f32 per-128 power-of-two scales | 64 bf16 rope) through one compact CSR slice per row
(`indptr [rows + 1]`, `slots` = `page * block_stride + offset`, ~256 keys per row at DCP8, up to 2048). One launch writes
the partial latent output `[rows, 96, 512]` bf16 and the natural-log LSE `[rows, 96]` fp32 that
`dcp8_lse_combine_vup_mla_h12_kv512` consumes as is (row strides are free; padding rows with an empty slice give 0 and
-inf).

Numerics (`reference.py`, the contract): fp32 scores `scale * q.k`, fp32 softmax and accumulation, bf16 output. The kernel
quantizes q per (head, 128 dims) to an fp8 hi term plus an fp8 residual, runs QK on the fp8 latent in scaled MFMAs, takes P
in fp16 with the value scale folded in and accumulates in fp32: cosine > 0.9999 and max error <= 2^-6 of the row max (measured
0.5%, the bf16 rounding of the output), LSE within 1e-2 (measured 3e-4). Two runs are bit for bit equal; no atomics.
Preconditions: per-128 scales are powers of two <= 2^7 (vLLM's fp8_ds_mla writer), every slot is inside the cache.

One variant serves every batch size (rows is a runtime argument). A workgroup is one (row, group of 16 heads); it runs on
XCD `p % 8`, with a row's six head groups and a request's rows on one XCD.

## Entry points

- `fp8_sparse_mla_decode`: the decode call of a DSA layer; replaces vLLM's `triton_sparse_mla_fwd` call at DCP
  (`rocm_aiter_mla_sparse.py:1672`), same inputs and `(out, lse)`.
- `supports_fp8_sparse_mla_decode`: the shape / dtype / alignment check (gfx950, 96 heads, 656 B flat records, int32 CSR).
- Dispatch: none (single kernel); the entry is registered with `amd_kernels.warm_up`, so its first call compiles it.

## Performance

Graph replay, inputs cold (q, output, slots over >600 MB; random 4 GiB cache), median us per launch, ~256 keys per row,
4 rows per request, vs AITER's Triton `sparse_mla_fwd` (vLLM's path today). Roofline: bytes (q, output, LSE, unique
KV records, slots) at 6.4 TB/s, FLOPs at 1.25 PF.

| rows | Triton us | this kernel us | speedup | roofline us | % of roofline |
|---|---|---|---|---|---|
| 12 | 18.2 | 16.7 | 1.09x | 0.5 | 3% |
| 24 | 23.2 | 17.0 | 1.37x | 1.0 | 6% |
| 40 | 25.8 | 17.5 | 1.48x | 1.7 | 10% |
| 65 | 29.9 | 24.7 | 1.21x | 2.8 | 11% |
| 80 | 30.7 | 25.9 | 1.19x | 3.4 | 13% |
| 128 | 52.0 | 39.6 | 1.31x | 5.4 | 14% |
| 192 | 69.9 | 55.5 | 1.26x | 8.3 | 15% |

At 2048 keys per row (no DCP split): 12 / 40 / 80 / 192 rows take 47.6 / 55.4 / 82.5 / 183 us vs 51 / 81 / 143 / 357 us
(1.07-1.95x, 448 TFLOP/s at 192 rows).

The kernel is latency-bound, not bandwidth- or FLOP-bound: a workgroup's critical path is three dependent cold loads (indptr,
slots, the gathered KV rows, ~7 us) plus ~2.4 us per 128-key tile and the epilogue. At large row counts the 6x head-group
split (6 workgroups per row) leaves ~4.5 workgroups per CU in flight at 2 resident.

## Tests

```bash
python3 -m pytest tests/attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 -q
python3 benchmarks/attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048/benchmark_decode.py
```
