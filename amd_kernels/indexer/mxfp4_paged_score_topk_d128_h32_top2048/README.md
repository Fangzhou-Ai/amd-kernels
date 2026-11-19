# MXFP4 paged index scorer + per-row top-2048, head dim 128, 32 heads (gfx950)

## What it does
The DSA decode indexer's `score_paged` + `_select(..., score_rows)` (vLLM `dsa.py:462`, `:475`) in two launches, the fp32
logits `[rows, width]` never written. Queries fp4 `[rows, 32, 64]` + e8m0 scales, index pages `[pages, 64, 68]` uint8
(page stride >= 4352 B), head weights fp32, a block table, causal `row_ends` (1 - 6 rows a sequence, <= 256 rows,
<= 131072 keys a rank). Out: per row the 2048 best keys, score desc (ties in any order), `-1` / `-inf` past the live
keys; rows past `query_start_loc[-1]` are padding. Scores are AITER `flydsl_pa_mqa_logits_fp4_rowgroup`'s bit for bit
(the MFMA scale epilogue is its own, see `scoring.py`); the selection is exact (a key above the 2048th is never lost).
Contract: `reference.py` (torch dequant, logits and `check_selection`, scores within 1e-4 relative).
Design: `lists.py` workgroup a (sequence, chunk, row tile) scores a chunk into LDS as order-preserving int32 keys and
keeps its top list (wave a row, relaxed threshold, ballot compaction); `merge.py` workgroup a row bins the lists'
scores into a float histogram, takes the threshold bin by exact rank, and re-reads the full scores of a chunk whose list
held more selected keys than it kept.

## Entry points
- `mxfp4_paged_score_topk(q_payload, q_scales, weights, cache, block_table, query_start_loc, row_ends, max_query_len,
  out_indices, out_scores)`: an `amd_kernels.warm_up` entry; CUDA-graph safe; scratch is one set a device.
- `prefer_fused(batch, width)`: the measured gate (<= 8 sequences, >= 16384 keys); outside it the unfused pair is as fast.
- `cache_views(cache)`: payload and scale views of a merged cache without a copy. `host.plan`: rows a workgroup scores
  and chunks a sequence takes, by batch (<= 4: 2 rows, 40 chunks; <= 16: 2 rows, 20; else whole rows, 12).

## Performance
Graph replay, cold pages (24 inputs over a 3 GiB pool), us per call, median. Stock = AITER row-group scorer into fp32
logits + `top_k_per_row_decode`. Ceiling = max(bytes / 6.4 TB/s, flops / 8.24 PF) + 2 graph launches (1.57 us) + 2 cold
dependent loads (0.245 us) = 4-11 us; one decode step's critical path of this design is ~15 us (K1 span 9: launch + 1.4
entry, q/w + page loads, 0.9 MFMA, barrier, 2.3 select; K2 7-8), so the table is 15-25% of the measured ceiling.

| Sequences x rows | Keys a rank | Stock | Fused | Speedup | % of ceiling |
|---|---|---|---|---|---|
| 2 x 6 | 15360 / 20480 / 94208 | 21.7 / 21.4 / 43.5 | 17.7 / 17.4 / 31.5 | 1.23x / 1.23x / 1.38x | 23 / 23 / 17 |
| 3 x 4 | 8192 / 32768 / 94208 | 15.5 / 25.6 / 44.3 | 18.6 / 21.4 / 33.1 | 0.83x / 1.19x / 1.34x | 17 / 22 / 19 |
| 8 x 4 / 8 x 5 | 94208 / 16384 | 52.9 / 23.9 | 48.2 / 23.0 | 1.10x / 1.04x | 17 / 21 |
| 10 x 4 / 13 x 5 / 20 x 4 | 32768 | 31.3 / 37.3 / 38.4 | 31.3 / 44.6 / 46.6 | 1.00x / 0.84x / 0.82x | 21 / 15 / 21 |
| 48 x 4 | 8192 / 32768 | 23.0 / 57.1 | 35.8 / 65.3 | 0.64x / 0.87x | 19 / 23 |

Below the bar (within 10% of the ceiling) by design: two dependent launches, each ~1.4 us cold start, and per-chunk
select passes of ~1 us each (VALU and barrier bound); beyond 8 sequences the chunk kernel's 1 - 2 workgroups a CU
(118 - 220 VGPRs, 35 - 106 KB LDS) leaves the HBM idle during selects.

## Tests
```bash
HIP_VISIBLE_DEVICES=1 python3 -m pytest tests/indexer/mxfp4_paged_score_topk_d128_h32_top2048 -q  # shapes, chunk ladder, concentrated, ties, padding, warm-up
python3 benchmarks/indexer/mxfp4_paged_score_topk_d128_h32_top2048/benchmark_score_topk.py [--quick]
```
