# attention/kda_decode_fb_gate_spec_h12_d128

## What it does

One launch for a Kimi Delta Attention (KDA) layer's spec-verify decode on one rank (12 heads of dim 128, TP8): the f_b
gate GEMM, the causal conv (width 4) + SiLU of q / k / v, the delta-rule recurrence over each sequence's tokens with
per-token state snapshots, and the sigmoid-gated RMSNorm. It replaces `kimi_k3_kda_fb_proj` + `causal_conv1d_update` +
`fused_recurrent_kda` + `FusedRMSNormGated` (or the AITER `kda_fused_recurrent_spec_decode` pair f_b GEMM + decode).

Inputs: the fused in_proj output `proj` [M, 6288] bf16 (q | k | v | gate | f_a | beta | pad, row stride a multiple of
8), `f_b_weight` [1536, 128] bf16, `A_log` [12] and `dt_bias` [1536] fp32, `conv_weight` [3, 4, 1536] fp32, the conv
state [slots, 4608, len] bf16 in the served (state_len, dim) layout (dim innermost), the recurrent state [slots, 12,
128, 128] fp32 (any slot pitch), `cu_seqlens` / `state_indices` [N, T] / `num_accepted` [N] int32, the output norm
weight (bf16 or fp32). It updates the conv and recurrent states in place and writes `out` [M, 12, 128] bf16. Token t of a
sequence reads its state from slot `state_indices[n, num_accepted - 1]` and writes the snapshot after token t to slot
`state_indices[n, t]` (slot <= 0 writes nothing; a sequence whose seed slot is <= 0 writes zeros).

Numerics (`reference.py` is the contract): fp32 throughout; g1 = bf16(f_a @ f_b^T) rounded once as the stock GEMM's
output; gate, conv + SiLU, L2 norms, recurrence and norm follow the AITER spec-decode kernel (cosine >= 0.9999 and
states within 2e-3 of the reference in the tests, identical conv state, two runs bit for bit equal).

## Entry points

- `kda_decode_fb_gate_spec`: the layer call (`uniform=True` when every sequence has exactly `state_indices.shape[1]`
  rows, which skips the `cu_seqlens` loads; `snapshot_token` [N] keeps only that token's state snapshot).
- `blocker`: why a call does not fit (None: it fits); the caller falls back to the stock chain otherwise.
- The wave count and row split per batch live in `host.py` (`waves_for`, `split_for`); kernels in `kernel.py`.

## Performance

Graph replay, 1 MI355X, inputs > 700 MB (cold), median us per layer call. `floor` = skeleton kernel (same grid, table
lookups, state read, snapshot writes, no math). Ceiling = 6.2 TB/s (measured blend of 6.5 read / 6.0 write, hw_ceilings.md).

| M (rows x tok) | MB | fused us | TB/s | % ceiling | floor us | fused/floor | stock f_b + AITER us | speedup |
|---|---|---|---|---|---|---|---|---|
| 12 (2 x 6) | 11.8 | 10.0 | 1.18 | 19 | 5.5 | 1.82 | 16.2 | 1.63x |
| 24 (6 x 4) | 24.9 | 8.7 | 2.87 | 46 | 5.8 | 1.49 | 14.7 | 1.69x |
| 40 (10 x 4) | 41.2 | 9.9 | 4.16 | 67 | 7.1 | 1.40 | 17.5 | 1.77x |
| 65 (13 x 5) | 63.9 | 13.7 | 4.66 | 75 | 10.4 | 1.32 | 21.5 | 1.57x |
| 80 (20 x 4) | 81.9 | 15.6 | 5.25 | 85 | 14.6 | 1.07 | 22.1 | 1.41x |
| 128 (32 x 4) | 130.9 | 25.5 | 5.14 | 83 | 19.5 | 1.31 | 32.3 | 1.27x |
| 192 (48 x 4) | 196.1 | 34.4 | 5.69 | 92 | 30.6 | 1.13 | 45.1 | 1.31x |

Small M is latency-bound (launch 1.6 us + prologue ~3.5 us + tail ~1 us); the large tiers are at 83-92% of the HBM
ceiling. Not helpful (measured, <3%): conv write-back placement, state-load ordering, store interleaving, extra waves.
Details and attempts log: analysis/kernels/kda_decode_fb_gate_spec_h12_d128.md.

## Tests

```bash
python3 -m pytest tests/attention/kda_decode_fb_gate_spec_h12_d128 -q
python3 benchmarks/attention/kda_decode_fb_gate_spec_h12_d128/benchmark_decode.py
```
