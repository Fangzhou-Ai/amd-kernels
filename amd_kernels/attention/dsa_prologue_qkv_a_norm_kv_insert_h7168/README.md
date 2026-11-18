# DSA decode prologue around the qkv_a GEMM: norms + fp8_ds_mla insert, indexer prep

## What it does

Two launches replace the ~9 small ops that run between the dense bf16 GEMMs of one DSA layer in decode (the qkv_a
GEMM, `q_b`, `wq_index` stay separate). Kimi K3.1 DSA is NoPE: k_pe is stored as is, there is no RoPE. Slot mappings are
already DCP-localized (a token another rank owns has slot -1): a slot < 0 writes no cache byte.

- `qkv_a_norm_kv_insert`: the qkv_a row `[q 1536 | kv 512 | k_pe 64]` bf16 -> `q_a` / `kv_a` RMSNorm outputs (bf16,
  fp32 accumulation, one rounding) and the fp8_ds_mla record of each owned token at `cache[slot // 64, slot % 64]`
  (512 e4m3 codes of the normed kv, 4 fp32 power-of-two tile scales, 64 bf16 k_pe; 656 bytes).
- `indexer_prep`: the wk_index key row -> LayerNorm (bf16 weight / bias) -> MXFP4 -> the indexer page slot (4096
  payload bytes + 256 e8m0 scale bytes per 64-slot page, any physical page stride >= 4352); the wq_index output
  `[n, 32, 128]` -> MXFP4 payload `[n, 32, 64]` + AITER row-group scales `[n, 1, 4, 16, 4]`; `weights = wv_index out
  .float() * weight_scale`; the top-k buffer rows set to -1.

Numerics contract (`reference.py`): the query payload / scales / weights and the top-k reset are bit-equal; norm outputs
are bit-equal except one-ulp moves from the summation order (>= 99.9% equal); fp8 codes within one step where the tile
scale matches; key payload follows the LayerNorm rounding. Tested bit-equal to the stock AITER / Triton / C++ ops on
random data. Slots are int64, rows of `qkv_a` need a row stride divisible by 8 elements.

## Entry points

- `qkv_a_norm_kv_insert.qkv_a_norm_kv_insert`: norms + KV insert of one layer (without a cache it norms only).
- `qkv_a_norm_kv_insert.supports_qkv_a_norm_kv_insert`: whether the inputs fit (gfx950, bf16, 656-byte rows).
- `indexer_prep.indexer_prep`: the indexer's key store, query quant, weights and top-k reset of one layer, in the form
  `_fp4_indexer` takes as `q_payload`, `q_scales`, `q_weights` (the starts are the decode `query_start_loc`).
- `indexer_prep.supports_indexer_prep`: whether the inputs fit (32 heads x 128, `[pages, 64, 68]` cache, top-k % 1024).

Both are entries (`amd_kernels.warm_up`): one variant per mode (insert or not; top-k width), the row count is a
runtime argument.

## Performance

One MI355X; one graph of 100 launches with their own inputs (together larger than the MALL), median us per call.
Stock = the multi-stream wrapper's ops: AITER `rms_norm` x2 + `concat_and_cache_mla` (3 launches); `layer_norm`,
`store_keys`, `fill_`, weights product, starts clamp, `quantize_queries` (6 launches). Both are latency-bound (HBM time
at 6.4 TB/s: 0.02-0.27 us and 0.04-0.58 us); a bare launch in a graph is 1.6 us.

| M | qkv_a: ours / stock | indexer_prep: ours / stock |
|---|---|---|
| 12 | 2.07 / 5.81 | 1.92 / 12.94 |
| 40 | 2.34 / 6.16 | 2.07 / 15.06 |
| 80 | 2.63 / 6.51 | 2.21 / 16.53 |
| 128 | 2.71 / 6.84 | 2.37 / 18.18 |
| 192 | 2.83 / 7.02 | 2.71 / 21.12 |

## Tests

```bash
python3 -m pytest tests/attention/dsa_prologue_qkv_a_norm_kv_insert_h7168 -q
python3 benchmarks/attention/dsa_prologue_qkv_a_norm_kv_insert_h7168/benchmark_prologue.py
```
