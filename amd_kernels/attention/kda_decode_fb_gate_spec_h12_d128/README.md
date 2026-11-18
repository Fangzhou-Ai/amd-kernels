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

TBD

## Tests

```bash
python3 -m pytest tests/attention/kda_decode_fb_gate_spec_h12_d128 -q
python3 benchmarks/attention/kda_decode_fb_gate_spec_h12_d128/benchmark_decode.py
```
