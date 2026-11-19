# Prefill indexer top-2048 select (DSA, gfx950)

## What it does

The select half of the DSA prefill indexer, after the FP4 scorer (`indexer/mxfp4_prefill_score_d128_h32`): for each row of an fp32
logits tile `[rows, width]` it returns the window-relative positions of the 2048 largest logits of the row's key window `[ks, ke)`,
ascending, ties to the smaller position, -1 padding rows of fewer than 2048 keys. The result is exact and bit-identical to
`reference.py` (= AITER `top_k_per_row_prefill(stable=True)` with `ks` subtracted, which is what the vLLM call site computes with its
`selected - starts` epilogue); the tile's positions outside the windows are never used. Logits must be finite or +-inf, not NaN; the
row stride is a multiple of 4 floats (the scorer's `pages * 64`), windows up to about 1M keys.

Four launches, one workgroup per row, no host sync:

- `threshold`: a weighted sample of the window (its last 4096 and first 512 keys exactly, about 1% stratified 64-key blocks of the
  middle) gives the value above which about 4.6K keys lie; windows of at most 8192 keys skip it.
- `stream`: one nontemporal read of the window; every value >= the threshold goes into an LDS list that is ascending in position
  by construction, copied out with its count.
- `retry`: rows whose count is outside [2048, 8192] (none in 9600 random rows; clustered data fails more often) stream again with a
  re-ranked threshold; every other row exits at once.
- `select`: exact 2048th largest of each list (12-bit radix round, then a brute-force rank of the few keys sharing the last bin) and
  an ordered compaction; a row all passes failed on (all-equal keys, pathological structure) takes three radix passes over the window.

## Entry points

- `host.topk_rows`: the select for one tile of rows (`logits`, `ks`, `ke` -> `out` int32 [rows, 2048], optional row order).
- `host.workspace_bytes`: the scratch of a call (cached per device and stream, grown on demand).

## Performance

(table from benchmarks/indexer/mxfp4_prefill_topk_d128_h32_top2048/benchmark_topk.py)

## Tests

```bash
python3 -m pytest tests/indexer/mxfp4_prefill_topk_d128_h32_top2048 -q
python3 benchmarks/indexer/mxfp4_prefill_topk_d128_h32_top2048/benchmark_topk.py
```
