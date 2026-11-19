# KDA decode layer K1: AttnRes, in_proj, conv, delta rule, gated norm in one launch (hidden 7168, 12 heads, head dim 128)

## What it does

One Kimi-K3 KDA layer of a spec-verify step (`R` requests x `L` tokens, `S = R L <= 16` rows) from the AttnRes seam to
the gated core output (o_proj's input), on one rank of TP8: `prefix += delta`, the softmax mix of `nb + 1` residual
sources and its RMSNorm -> x; `proj = x @ w_in^T` (6288 rows: q k v g 4 x 1536, f_a 128, beta 12 + 4 pad); the width-4
conv with the conv state; `g1 = f_b f_a`, the lower-bound gate, q / k l2norm; the delta-rule recurrence on the
fp32 state `[12, 128, 128]` of each request (a state snapshot per token to the slot `st_idx` names); the gated RMSNorm.
It reads vLLM's weights and states in place. Contract: `reference.py` (pure torch; bf16 rounding of every hand-off as
the stock path; `prefix` and `blocks` exact, states <= 1e-3 and core <= 3e-3 relative L2, cosine >= 0.9999: the kernel
sums in another order).

One persistent launch, 256 CTAs x 512 threads, all resident, handing off through tagged mailboxes in scratch
(`Workspace`: step epoch and layer slot make every tag unique, nothing is zeroed). Token CTAs (one a row) run AttnRes
whole; every other CTA streams 26 rows of in_proj (16 + 10, 373 KB) into registers at kernel start, while one of its waves
polls for x and DMAs it into LDS; the MFMAs run as the weights land; the KDA tasks (request x head x 16 state rows)
follow on the same CTAs. A request whose first slot is the null slot (graph padding) reads zeros and writes nothing.

## Entry points

- `kda_mono_decode_k1`: the launch (an `amd_kernels.warm_up` entry: the first call of a mode compiles rows `qlen ..
  16`); returns `core` [S, 1536] bf16.
- `Workspace`: scratch, epoch and fail-stop words; `begin_step()` once a step before its first launch (graph-capturable),
  `failed()` after a wait gave up (10 s).

## Performance

Graph replay of 4 launches over distinct weight sets (360 MB of in_proj, past the LLC), 4 AttnRes blocks, median us a launch, same harness for
vLLM's stock K1; floor = bytes at the measured ceilings (6.5 TB/s read, 6.0 TB/s write) + the 1.57 us graph launch gap; skeleton = the same hand-off chain and weight
stream with no math (analysis report): 21.5 us at 12 rows, 21.2 at 6.

| rows x tokens a request | stock us | this kernel us | speedup | floor us | % of floor |
|---|---|---|---|---|---|
| 12 x 6 (CI c4) | 45.4 | 30.5 | 1.49x | 17.5 | 57% |
| 6 x 6 (graph pad) | 35.5 | 28.1 | 1.26x | 16.5 | 59% |
| 4 x 4 | 33.0 | 26.3 | 1.26x | 16.2 | 62% |
| 8 x 4 | 35.5 | 28.1 | 1.26x | 16.9 | 60% |
| 8 x 8 | 37.1 | 30.0 | 1.24x | 16.8 | 56% |
| 16 x 4 (4 requests: two task rounds) | 54.8 | 37.3 | 1.47x | 18.3 | 49% |
| 16 x 8 | 49.7 | 33.7 | 1.48x | 18.1 | 54% |

The weight stream lands at ~6.3 TB/s (the measured ceiling); the kernel is that stream plus a ~15 us dependent tail (reduce / hand-offs / conv / gates / recurrence / norm).

## Tests

```bash
python3 -m pytest tests/attention/kda_mono_decode_k1_h12_d128_ranks8 -q
python3 benchmarks/attention/kda_mono_decode_k1_h12_d128_ranks8/benchmark_k1.py
```
