# KDA chunked prefill, 12 heads x 128 (Kimi-K3.1, TP8)

## What it does

The whole prefill of one KDA layer in FlyDSL (gfx950), replacing the stock conv launch, the HIP/Triton prologue, the chunk
kernels and the gated norm: (1) conv1d(4)+SiLU on q/k/v with conv-state read/roll, the gate from the f_b output
(`A_log`, `dt_bias`, lower bound), q/k L2 norm and beta sigmoid, written as bf16 fragment-major chunk records (tiles
w, qg, kgt, Aqk, decay, u: 73 KB per head and chunk of 64); (2) the chunked delta-rule scan over fp32 recurrent states,
ragged batches, initial state / conv state per sequence, final state (and chunk-aligned checkpoint snapshots) written
in place; (3) the gated RMSNorm (sigmoid gate) epilogue. Long chains are split in groups by an exact decomposition
`S_out = S_in M_g + B_g` (chain launch + fold hosted by the walk launch), deterministic, no atomics into outputs.
Contract: `reference.py` (stock numerics, fp32 state).

## Entry points

- `prefill(...)` (`host.py`): the three stages, writes `out [T,12,128]` bf16, updates `conv_state` and `state_cache` in place.
- `chunk_table`: per-step chunk table, built once per step and shared by all layers.
- `chunk_scan`, `gated_norm`, `plan_groups`: the stage launches; `amd_kernels.warm_up` entry compiles the variants.

## Performance

MI355X, graph replay, inputs larger than the LLC, min of 15, us per layer call (`gpurun 2`). Floors: HBM at the measured
ceilings (read 6.5 TB/s, write 6.0 TB/s); "+ws" adds this design's bf16 workspace round trip; MFMA at 1.98 PF (bf16 dense).

| Tokens (seqs) | fused | stock | speedup | HBM floor | %HBM floor | +ws floor | % of +ws | MFMA floor |
|---|---|---|---|---|---|---|---|---|
| 2048 (1) | 67.3 | 107.3 | 1.59x | 6.1 | 9% | 19.8 | 29% | 2.4 |
| 4096 (1) | 93.3 | 164.2 | 1.75x | 12.0 | 13% | 39.3 | 42% | 4.8 |
| 4096 (4) | 83.1 | 145.8 | 1.75x | 12.0 | 14% | 39.3 | 47% | 4.8 |
| 5260 (3000+1500+700+60) | 116.2 | 218.6 | 1.88x | 15.4 | 13% | 50.7 | 44% | 6.3 |
| 10240 (1) | 194.0 | 338.4 | 1.74x | 29.7 | 15% | 97.8 | 50% | 12.1 |
| 16384 (1) | 300.1 | 517.2 | 1.71x | 47.4 | 16% | 156.3 | 52% | 19.4 |

Stages at 4096 (us): prologue 43.2, scan 50.0 (no-math skeleton with the same DMAs and launches: 47.1), norm 7.5.
Binding resource: not MFMA (compute is 5% of the time) and not HBM bandwidth; the scan is bound by per-CU LDS/DMA
ingest and its serial per-chunk walk (at least 1.4 us per chunk, measured), the prologue by VALU issue (VALU busy 36%, memory
not binding). See `analysis/kernels/kda_prefill_h12_d128.md`.

## Tests

```bash
python3 -m pytest tests/attention/kda_prefill_h12_d128 -q
python3 benchmarks/attention/kda_prefill_h12_d128/benchmark_prefill.py --tokens 2048,4096,10240,16384 --stages
```
