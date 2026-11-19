# Prefill indexer top-2048 select (DSA, gfx950)

## What it does

The select half of the DSA prefill indexer, after the FP4 scorer (`indexer/mxfp4_prefill_score_d128_h32`): for each row of an fp32 logits tile
`[rows, width]` it returns the window-relative positions of the 2048 largest logits of the row's key window `[ks, ke)`
(ascending, ties to the smaller position, -1 padding rows of fewer than 2048 keys). The result is exact and bit-identical to
`reference.py` (= AITER `top_k_per_row_prefill(stable=True)` with `ks` subtracted, which is what the vLLM call site computes with its
`selected - starts` epilogue). Three launches, one workgroup per row, no host sync:

- `threshold`: a stratified 1.25% sample of 64-key blocks gives three value thresholds with ~3.6K / 11K / 2.6K expected hits;
- `stream`: one nontemporal read of the window; every value >= the first threshold goes into an LDS list that is ascending in position
  by construction; the hit counts of the other two thresholds make a failed row (count outside [2048, 6144]) pick a threshold of
  exactly known count for one more pass (at most three passes);
- `select`: exact 2048th largest of the list (12-bit radix round + brute-force rank of the few entries sharing the final bin), ordered
  compaction. Rows whose passes all failed (all-equal keys, pathological structure) take an exact radix over the window: slow, rare, correct.

Windows of at most 6144 keys skip the threshold (the list is the window). Positions outside the window are never used as data.

## Entry points

- `host.topk_rows(logits, ks, ke, out)`: `logits` fp32 [rows, width] (row stride free), `ks` / `ke` int32 [rows], `out` int32 [rows, 2048].
- `host.workspace_bytes(rows, width)`: scratch of a call (cached on the device, grown on demand).

## Performance

(filled from benchmarks/indexer/mxfp4_prefill_topk_d128_h32_top2048/benchmark_topk.py)

## Tests

```bash
python3 -m pytest tests/indexer/mxfp4_prefill_topk_d128_h32_top2048 -q
python3 benchmarks/indexer/mxfp4_prefill_topk_d128_h32_top2048/benchmark_topk.py
```
