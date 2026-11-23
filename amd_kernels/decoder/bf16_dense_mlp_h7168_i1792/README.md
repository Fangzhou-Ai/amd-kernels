# Dense MLP of the drafter: bf16 gate_up + SwiGLU, down, hidden 7168, intermediate 1792 per rank

## What it does

One rank's shard (TP8) of the DSpark drafter's dense MLP (vLLM's `KimiMLP`, hidden_act silu, `reduce_results=False`):
`out = down(silu(x gate^T) * (x up^T))` for 1-256 rows, the rank's bf16 [rows, 7168] partial that the next
all-reduce + RMSNorm sums. `gate_up` is the merged [3584, 7168] weight (gate rows, then up rows),
`down` [7168, 1792]. Two launches: gate_up with the SwiGLU epilogue (fp32 accumulators, bf16 intermediate written in the
packed layout `down` reads) and down. Contract (`reference.py`): fp64 products, the intermediate rounded once to bf16 (gate
and up are not rounded to bf16 first, unlike the stock `silu_and_mul` input), cosine >= 0.99999, relative L2 <= 1e-3,
bitwise repeatable (split sums in a fixed order, tickets only count).
Weights and activations are read in the MFMA tile layout (a 16-row x 32-K tile is 1 KB in lane order): `pack_gate_up`,
`pack_down` at load, `pack_rows` for the activations. A producer that writes its 16 B packs to the tile address
(row r, columns 8j..8j+7 -> `(((r // 16) * 224 + j // 4) * 64 + (j % 4) * 16 + r % 16) * 16` bytes) needs no pack
launch; `pack.unpack_rows` is the inverse.

## Entry points

- `pack_gate_up`, `pack_down`: the packed weights, once at load (free the row-major tensors).
- `pack_rows`, `packed_rows_buffer`, `packed_tiles`: the packed activations of ``rows`` rows (tiles padded to the tier).
- `MlpWorkspace`: partial-sum slabs, ticket counters and the packed intermediate of one stream.
- `dense_mlp`: the rank's partial output; graph safe; the first call warms every row tier up (`amd_kernels.warm_up`).
- `host.swiglu_gate_up`, `host.down_proj`: its two launches.

`host.TIERS` holds the tile / split / wave grid of both GEMMs per row tier; `kernel.py` the design (K split across
workgroups of one XCD, last arrival sums, weights bypass the L2).

## Performance

One MI355X, graph replay of 24 calls over six rotated copies of the 77 MB weights, min of 15 rounds, us per call. Stock =
vLLM `rocm_unquantized_gemm_impl` (gate_up, down) + `silu_and_mul`. Floor = the weights at the 6.5 TB/s cold-read
ceiling (11.9 us) + two launch floors (1.6 us); from 160 rows on, two row blocks read the weights twice.

| rows | 16 | 32 | 60 | 80 | 128 | 144 | 160 | 192 | 224 | 240 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stock | 22.8 | 27.1 | 29.9 | 38.1 | 38.8 | 41.6 | 41.9 | 43.9 | 44.7 | 49.6 | 47.2 |
| this entry | 17.3 | 19.1 | 23.9 | 26.2 | 31.8 | 34.3 | 34.7 | 37.0 | 39.6 | 42.3 | 42.3 |
| % of floor | 87 | 79 | 63 | 58 | 47 | 44 | 44 | 41 | 38 | 36 | 36 |

## Tests

- `python3 -m pytest tests/decoder/bf16_dense_mlp_h7168_i1792 -q` (1 GPU)
- `python3 benchmarks/decoder/bf16_dense_mlp_h7168_i1792/benchmark_mlp.py` (1 GPU, needs vLLM)
