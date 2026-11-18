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

One MI355X, graph replay, 3 rotated `markov_w2` copies, 8 input sets (HBM-cold), T 1.0 (T 0.7 within 2 us; T 0: 138.7 / 140.9 / 119.6 / 140.0 us against vLLM's 237.7 / 272.4 / 185.5 / 275.1),
median of 5 alternated rounds of 10 replays, us per chain. Baseline = vLLM's loop in the same harness.

| requests / steps (conf) | 8 / 4 (c16) | 13 / 4 (c24) | 20 / 3 (c32) | 48 / 3 (c48) |
|---|---|---|---|---|
| vLLM loop | 247.6 | 285.8 | 199.3 | 306.5 |
| this entry | 149.0 | 151.5 | 119.9 | 142.6 |
| HBM MB / % of 6.4 TB/s | 692 / 73% | 705 / 73% | 543 / 71% | 598 / 65% |

Limiter: `markov_w2` streams at ~5.5 TB/s (a `torch.sum` over 1.5 GB reads at 5.4 TB/s on this node), the base-logit read and
cache write add 2 x n x 327 KB a step, and each step ends with a ~3 us ticket tail.

## Tests

```bash
python3 -m pytest tests/sampling/markov_gumbel_fused_v163840_r512 -q
python3 benchmarks/sampling/markov_gumbel_fused_v163840_r512/benchmark_markov_gumbel_fused.py [--temp 0.7]
```
