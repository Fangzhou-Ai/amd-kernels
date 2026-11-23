# Sparse MLA decode at DCP8, fp8_ds_mla cache (96 latent heads, kv_lora 512, top-2048)

## What it does

The core attention of a DSA layer at DCP8, per rank: 96 latent heads (the DCP-gathered q, bf16 `[rows, 96, 576]` =
512 latent + 64 rope dims) attend this rank's keys of each row's global top-2048, read from the fp8_ds_mla cache
(656 B records: 512 e4m3 | 4 f32 per-128 power-of-two scales | 64 bf16 rope) through one compact CSR slice per row
(`indptr [rows + 1]`, `slots` = `page * block_stride + offset`, ~256 keys per row at DCP8, up to 2048). One launch writes
the partial latent output `[rows, 96, 512]` bf16 and the natural-log LSE `[rows, 96]` fp32 that
`dcp8_lse_combine_vup_mla_h12_kv512` consumes as is (row strides are free; padding rows with an empty slice give 0 and
-inf). q may instead come pre-quantized (`q8`, `reference.quantize_q`: fp8 hi + fp8 residual + E8M0 scales + bf16 rope,
`[rows, 96, 1168]` uint8), which the q-gather kernel can emit.

Numerics (`reference.py`, the contract): fp32 scores `scale * q.k`, fp32 softmax and accumulation, bf16 output. QK runs on
the fp8 latent in scaled MFMAs with q as hi + residual, P goes to fp16 with the value scale folded in, accumulation in
fp32: cosine > 0.99999 against the Triton kernel, LSE within 1e-2. Two runs are bit for bit equal; the q8 and bf16 q paths
are bit for bit equal; no atomics. Preconditions: per-128 scales are powers of two <= 2^7 (vLLM's fp8_ds_mla writer),
every slot is inside the cache.

## Entry points

- `fp8_sparse_mla_decode`: the decode attention of a DSA layer at DCP, in place of vLLM's `triton_sparse_mla_fwd`
  call; returns the same `(out, lse)`.
- `supports_fp8_sparse_mla_decode`: the shape / dtype / alignment check (gfx950, 96 heads, 656 B flat records, int32 CSR).

`host.py` picks the kernel by row count (whole row, tile loop, two tile-loop units per workgroup); all are registered
with `amd_kernels.warm_up`.

## Performance

Graph replay, min of 20 replays, us per launch, ~256 keys per row, 4 rows per request, MI355X. Inputs cold (each launch
its own q, slots and output, > 600 MB); hot q: one q for every launch (as served: the q-gather kernel just wrote it).
Baselines in the same harness: AITER's Triton `sparse_mla_fwd` and vLLM's FlyDSL 96-head kernel
(`VLLM_ROCM_USE_AITER_FLYDSL_SPARSE_MLA=1`). Roofline: q + output + LSE + unique records + slots at 6.4 TB/s
(the MFMA floor is below it: 5.2 us at 240 rows).

| rows | Triton | FlyDSL | this | vs Triton | vs FlyDSL | FlyDSL, hot q | this, hot q | roofline | % roofline |
|---|---|---|---|---|---|---|---|---|---|
| 12 | 18.3 | 16.9 | 12.7 | 1.44x | 1.34x | 16.5 | 11.4 | 0.5 | 4% |
| 24 | 23.2 | 17.2 | 13.6 | 1.71x | 1.26x | 16.9 | 12.6 | 1.0 | 8% |
| 40 | 25.7 | 17.7 | 14.8 | 1.73x | 1.19x | 17.3 | 13.8 | 1.7 | 12% |
| 65 | 29.8 | 18.5 | 18.9 | 1.57x | 0.98x | 18.0 | 18.2 | 2.8 | 15% |
| 80 | 30.6 | 19.2 | 19.4 | 1.58x | 0.99x | 18.6 | 18.6 | 3.4 | 17% |
| 128 | 51.4 | 26.6 | 22.4 | 2.30x | 1.19x | 25.5 | 20.3 | 5.5 | 24% |
| 129 | 51.8 | 26.8 | 32.5 | 1.59x | 0.82x | 25.4 | 31.0 | 5.5 | 17% |
| 192 | 69.0 | 34.8 | 35.6 | 1.94x | 0.98x | 33.0 | 34.0 | 8.2 | 23% |
| 240 | 79.3 | 38.1 | 37.4 | 2.12x | 1.02x | 35.2 | 35.9 | 10.2 | 27% |
| 256 | 81.8 | 39.0 | 38.6 | 2.12x | 1.01x | 36.2 | 36.3 | 10.9 | 28% |

At 129-256 rows the two-unit kernel is 1.4-4 us faster than the tile loop on the same inputs; against FlyDSL at 192-256
rows the difference is within the run-to-run spread (about 1 us). The pre-quantized `q8` saves 0.1-1.6 us here but costs
the q-gather kernel more than that to emit (+0.7 us at 12 rows, +8 us at 192), so bf16 q is the default.

## Tests

```bash
python3 -m pytest tests/attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 -q
python3 benchmarks/attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048/benchmark_decode.py [--hot-q]   # FlyDSL column: vLLM on PYTHONPATH
```
