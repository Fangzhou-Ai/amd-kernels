# KDA chunked prefill, 12 heads x 128 (Kimi-K3.1, TP8)

## What it does

The whole prefill of one KDA layer in FlyDSL (gfx950), replacing the stock conv launch, the HIP/Triton prologue, the chunk
kernels and the gated norm: conv1d(4) + SiLU on q / k / v with conv-state read / roll, the gate from the f_b output (`A_log`,
`dt_bias`, lower bound), q / k L2 norm and beta sigmoid, written as bf16 fragment-major chunk records (w, qg, kgt, Aqk, decay, u:
73 KB per head and chunk of 64); the chunked delta-rule scan over fp32 recurrent states (ragged batches, initial and conv state
per sequence, final state and chunk-aligned checkpoint snapshots written in place); the gated RMSNorm (sigmoid gate) epilogue.
Contract: `reference.py` (stock numerics, fp32 state).

Two paths with the same contract and the same numerics (output rel-L2 <= 4.4e-3 against the fp32 reference, state <= 2.2e-3):

- one launch (`mega.py`) for batches up to 4096 tokens (<= 3 sequences): prologue workgroups, walk workgroups (the serial state recurrence of a sequence and
  head, two compute and two DMA waves) and output workgroups (`O = qg S^T + Aqk Vn` and the norm) run side by side and hand
  records and states over through flags in memory (write-through stores, agent-scope polls, every flag reset by its consumer);
- staged (`prologue.py`, `scan.py`, `norm.py`): prologue, chain / fold / walk of long chains cut in groups (exact
  `S_out = S_in M_g + B_g`), gated norm.

## Entry points

- `prefill(...)` (`host.py`): one layer; `uses_mega(chunk_counts)` picks the path from the chunks per sequence (host ints), `impl="mega"|"v1"` forces one.
- `chunk_table`: per-step chunk table, built once per step and shared by the layers (staged and one-launch paths).
- `chunk_scan`, `gated_norm`, `plan_groups`, `walk_states`, `chunk_outputs`: the stage launches (tests, benchmarks).

## Performance

MI355X, graph replay, inputs larger than the LLC, min of 15, us per layer call (`gpurun 2`). Floors at the measured ceilings
(read 6.5 TB/s, write 6.0 TB/s): `io` the unavoidable HBM traffic, `+ws` with the workspace write and re-reads; MFMA at 1.98 PF.

| Tokens (seqs) | path | v2 | staged | stock | v2 / stock | io floor | % | +ws floor | % | MFMA |
|---|---|---|---|---|---|---|---|---|---|---|
| 256 | one launch | 28.7 | 32.9 | 42.9 | 1.50x | 1.0 | 3 | 2.7 | 9 | 0.3 |
| 512 | one launch | 31.5 | 40.3 | 52.7 | 1.67x | 1.7 | 5 | 5.1 | 16 | 0.6 |
| 1024 | one launch | 41.6 | 53.7 | 69.0 | 1.66x | 3.2 | 8 | 10.0 | 24 | 1.2 |
| 2048 | one launch | 57.5 | 66.5 | 106.6 | 1.86x | 6.1 | 11 | 19.8 | 34 | 2.4 |
| 3072 | one launch | 74.9 | 83.4 | 135.9 | 1.81x | 9.1 | 12 | 29.5 | 39 | 3.6 |
| 3840 | one launch | 86.9 | 89.4 | 149.8 | 1.72x | 11.3 | 13 | 36.8 | 42 | 4.5 |
| 4096 | staged | 91.1 | 91.1 | 163.1 | 1.79x | 12.0 | 13 | 39.3 | 43 | 4.8 |
| 6144 | staged | 123.1 | 123.1 | 218.1 | 1.77x | 17.9 | 15 | 58.8 | 48 | 7.3 |
| 10240 | staged | 189.4 | 189.4 | 335.3 | 1.77x | 29.7 | 16 | 97.8 | 52 | 12.1 |
| 16384 | staged | 292.7 | 292.7 | 507.6 | 1.73x | 47.4 | 16 | 156.3 | 53 | 19.4 |
| 1024 + 1024 | one launch | 52.5 | 63.8 | | | 6.1 | 12 | 19.8 | 38 | 2.4 |
| 1200 + 1200 + 600 | one launch | 71.4 | 81.2 | | | 8.9 | 12 | 29.3 | 41 | 3.6 |
| 4 x 700 | staged | 69.8 | 69.8 | | | 8.3 | 12 | 27.0 | 39 | 3.3 |
| 300 + 64 + 77 + 1 | one launch | 30.2 | 37.5 | | | 1.5 | 5 | 5.3 | 18 | 0.7 |
| 3000 + 1500 + 700 + 60 | staged | 114.0 | 114.0 | | | 15.4 | 13 | 50.7 | 44 | 6.3 |

Neither HBM nor the MFMAs bind: the prologue is VALU-issue / latency bound (VALU busy 35%, 8.8 M VALU instructions per
4096-token launch), the state recurrence of a head is a serial chain of about 0.65 us per chunk (LDS 17%, MFMA 12% busy on the
CUs it runs on), and the one-launch path starts the chain only when the first records exist (about 22 us). See
`analysis/kernels/kda_prefill_v2.md`.

## Tests

```bash
python3 -m pytest tests/attention/kda_prefill_h12_d128 -q
python3 benchmarks/attention/kda_prefill_h12_d128/benchmark_prefill.py --tokens 256,1024,4096,16384 --stages
python3 benchmarks/attention/kda_prefill_h12_d128/benchmark_timeline.py --lens 4096
```
