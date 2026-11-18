# Fused Markov draft chain of a Kimi-K3.1 DSpark draft, vocab 163840, Markov rank 512

## What it does

The loop of vLLM's `DSparkSpeculator._sample_sequential` after the base logits, for `draft_sample_method=probabilistic`
(draft logits cached for rejection): per step s and request r, `bias = bf16(w1[prev] . w2^T)` (fp32 accumulation, one
bf16 rounding: the GEMV), `logits = bf16(base[r, s] + bias)` stored into the draft logits cache column s (honouring
`DraftLogitsCache.row_mapping`), then vLLM's Gumbel draw (murmur3 uniform of (seed ^ (s << 32), sample_pos - 1 + 2^30,
column), `-log(-log1p_neg_stable(u))`, `logits / T` with IEEE division, T = 0 plain argmax, invalid rows plain argmax and
no cache write) and the argmax (first column on ties) into `draft_tokens[r, s]`, which is the next step's `prev`. One
launch per step in place of the stock ~9 (embedding gather, GEMV, add, Gumbel kernel, argmax, gather, copy): the
`markov_w2` rows (163840 x 512 bf16, 168 MB, read once a step for all rows) are the only large traffic besides the
base-logit read and cache write (2 x n x 327 KB). Shapes: 1-48 requests, any step count, vocab 163840, rank 512,
`markov_w1` / `markov_w2` bf16 contiguous. Caller: the K3.1 DSpark speculator at c16-c48 (8 / 13 / 20 / 48 requests,
4 / 4 / 3 / 3 steps). Contract (`reference.py`): bit for bit vLLM's `gumbel_sample` on the cached logits (checked
outside this repository, 434 draws, 0 mismatches); the bias sum order differs from the GEMV's (a few logits one bf16
ulp, 1.5e-6 of elements); inputs finite.

## Entry points

- `markov_gumbel_fused`: the k steps over `base` [n, k, 163840] into `draft_tokens` and the cache, graph safe, warms its
  three row-tile variants up on its first call; the speculator's call site replaces its `for i in range(n_spec)` loop.
- `scratch_dwords`: size of the int32 scratch (ticket counter + per-workgroup pairs), zeroed once by the caller.
- `available`: gfx950 with FlyDSL importable.

`host.py` picks the variant by row count (16 / 32 / 48 rows); one compiled step kernel serves every step and temperature.

## Performance

One MI355X, graph replay, 3 rotated `markov_w2` copies, 8 input sets (HBM-cold), T 1.0,
median of 5 alternated rounds of 10 replays, us per chain. Baseline = vLLM's loop in the same harness.

| requests / steps (conf) | 8 / 4 (c16) | 13 / 4 (c24) | 20 / 3 (c32) | 48 / 3 (c48) |
|---|---|---|---|---|
| vLLM loop | 238.0 | 276.9 | 202.7 | 307.6 |
| this entry | 138.8 | 141.8 | 117.3 | 140.1 |
| measured ceiling (streaming skeleton + extra bytes) | 115 | 118 | 90 | 100 |
| % of the measured ceiling | 83% | 83% | 77% | 71% |

Ceiling: a FlyDSL skeleton with this kernel's weight-load pattern and no math reads `markov_w2` in 28.0 us (6.0 TB/s, grid / depth /
cache-hint / interleave variants 27.3-29.5 us; includes launch), plus 2 x n x 327 KB a step at 6.1 TB/s, per step. Not in the
ceiling and the remaining gap: the prologue (previous-token gather, `w1` rows, request state: 3.3 us at 16 rows, 6.8 us at 48; the
weight stream cannot start before the loads of that chain without delaying them), the ticket tail (2.7 us at 16 rows, 5.7 us at 48)
and ~5 us of spread between the workgroups' finish times. At T 0 (no noise): 129.5 / 133.4 / 117.5 / 137.7.

## Tests

```bash
python3 -m pytest tests/sampling/markov_gumbel_fused_v163840_r512 -q
python3 benchmarks/sampling/markov_gumbel_fused_v163840_r512/benchmark_markov_gumbel_fused.py [--temp 0.7]
```
