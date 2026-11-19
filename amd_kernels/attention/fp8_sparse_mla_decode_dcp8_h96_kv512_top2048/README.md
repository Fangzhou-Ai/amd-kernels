# fp8_sparse_mla_decode_dcp8_h96_kv512_top2048

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

Two kernels, picked by row count. Rows <= 42 (`kernel_row.py`): one workgroup of 8 waves per (row, 16-head group) holds
the row's 288 keys in LDS, splits QK over tokens and PV over dims, and loops a second pass for longer rows. Rows >= 43
(`kernel_tile.py`): one workgroup per (row, 3 head groups) streams 64-key tiles through a 3-deep LDS ring, one DMA wave
feeding QK and PV waves a tile apart, with a lazy online-softmax rescale.

## Entry points

- `fp8_sparse_mla_decode(q, kv_cache, indptr, slots, sm_scale, out=None, q8=None)`: the decode call of a DSA layer;
  replaces vLLM's `triton_sparse_mla_fwd` call at DCP (`rocm_aiter_mla_sparse.py:1664-1684`), same `(out, lse)`.
- `supports_fp8_sparse_mla_decode`: the shape / dtype / alignment check (gfx950, 96 heads, 656 B flat records, int32 CSR).
- Dispatch: rows <= 42 whole-row kernel, otherwise tile kernel; both registered with `amd_kernels.warm_up`.

## Performance

Graph replay, inputs cold, min of 15 replays, us per launch, ~256 keys per row, 4 rows per request, MI355X. "skeleton" is
the same launch grid doing only the index + KV gather DMA with its barriers (no math); roofline is bytes at 6.4 TB/s.

| rows | v1 | v2 bf16 q | v2 q8 | skeleton | v2 q8 / skeleton | roofline |
|---|---|---|---|---|---|---|
| 12 | 16.2 | 11.7 | 11.6 | 7.1 | 61% | 0.7 |
| 24 | 16.7 | 12.8 | 12.6 | 7.1 | 56% | 1.4 |
| 40 | 17.2 | 13.6 | 13.6 | 7.2 | 53% | 2.4 |
| 65 | 23.8 | 20.2 | 19.1 | 12.1 | 63% | 3.8 |
| 80 | 25.2 | 20.6 | 19.6 | 11.7 | 60% | 4.7 |
| 128 | 37.9 | 22.6 | 21.6 | 16.7 | 77% | 7.6 |
| 192 | 54.0 | 38.4 | 36.8 | 23.5 | 64% | 11.4 |

The 90%-of-skeleton bar is not met at any row count (53-77%); `analysis/kernels/sparse_mla_v2.md` has the stamps, PMC
counters and attempts. Persistent CTAs and a cross-workgroup token split were not built.

## Tests

```bash
python3 -m pytest tests/attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048 -q
python3 benchmarks/attention/fp8_sparse_mla_decode_dcp8_h96_kv512_top2048/benchmark_decode.py
```
