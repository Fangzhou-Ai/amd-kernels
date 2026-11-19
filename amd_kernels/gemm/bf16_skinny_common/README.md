# bf16 skinny linear: the dense decode GEMMs on one tile table

## What it does
`y [M, N] = x [M, K] @ W^T`, bf16 operands, fp32 sums, one bf16 rounding, M = 1..192 rows (more rows: more M tiles), for
the 15 dense weight shapes of a Kimi-K3.1 TP8 decode step (`SHAPES`: KDA in_proj 6288x7168, o_proj 7168x1536, DSA
qkv_a / q_b / g_proj / wq_index / wk_index / wv_index, latent down_proj 3584x7168, shared gate_up / down, layer-0 MLP,
lm_head shard 20480x7168, drafter context_proj 7168x35840 and context_kv 2880x7168). W is read from HBM in a load-time
packed layout (`pack_weight`, a permutation: every 16-byte lane load is one piece of a 1 KB run). Contract
(`reference.py`): the fp64 product rounded once to bf16; per-row cosine >= 0.99999, rel. L2 <= 1e-3, deterministic bits
(K split across workgroups is summed in split order by the last workgroup of a tile, no output atomics). The tile
(`wn`, `wk`, `ng`, `s`, depth, ring) comes from `config.py`, measured per (shape, M tier) by graph-replay sweeps.

## Entry points
- `pack_weight`, `unpack_weight`: the load-time layout; replace the parameter once before graph capture, free the
  row-major tensor (the packed [N, K] bf16 tensor has the same shape and bytes).
- `bf16_linear`: `x [M, K] @ W^T` into `out`, graph-safe after the first call of a shape outside capture (an
  `amd_kernels.warm_up` entry: the first call compiles the 13 tile variants of the shape and allocates its split-K
  workspaces); one in-flight call per shape and device unless the caller passes its own `workspace`.
- `warmup_bf16_linear`: the same warm-up, explicitly. `supports`, `prefers_stock`: shape known / the stock op measured
  as fast as the kernel at that row count (`STOCK_FASTER`: small K at M >= 65-128, qkv_a, lm_head at M 65-128).
- Assumption: workgroup b runs on XCD b % 8 (the split-K tiles are exchanged through one XCD's L2; checked by
  `tests/.../test_xcd_placement.py`). `kernel.py` documents the tile; `host.py` maps M to a tier.

## Performance
One MI355X, graph replay, rotated weight copies (> 768 MB, cold HBM), minimum of 7 rounds, us per call: this kernel /
stock (vLLM `rocm_unquantized_gemm_impl`, AITER on: wvSplitKrc, hipBLASLt, AITER asm), (% of the floor). Floor =
1.57 us (empty graph kernel) + bytes / 6.9 TB/s (measured nontemporal HBM read, `analysis/hw_ceilings.md`).
Bytes = W + x + y.

| N x K | op | M=12 | M=24 | M=40 | M=65 | M=80 | M=128 | M=192 |
|---|---|---|---|---|---|---|---|---|
| 6288x7168 | KDA in_proj_qkvgfab | 16.5 / 17.9 (89%) | 18.1 / 19.3 (81%) | 18.9 / 21.4 (78%) | 21.4 / 22.9 (70%) | 21.8 / 24.0 (69%) | 26.3 / 28.6 (58%) | 35.1 / 41.9 (44%) |
| 7168x1536 | o_proj (KDA, DSA) | 5.5 / 5.9 (88%) | 6.0 / 6.5 (80%) | 6.6 / 7.2 (74%) | 8.4 / 8.3 (58%) | 8.7 / 8.6 (57%) | 10.3 / 10.0 (49%) | 13.5 / 10.9 (39%) |
| 3584x7168 | latent down_proj | 10.8 / 11.6 (84%) | 11.1 / 12.0 (82%) | 12.0 / 12.8 (76%) | 13.8 / 17.2 (67%) | 14.1 / 17.8 (66%) | 17.1 / 20.7 (55%) | 23.6 / 31.1 (41%) |
| 1536x7168 | shared gate_up, g_proj | 6.5 / 7.0 (74%) | 7.1 / 7.4 (68%) | 7.6 / 8.2 (64%) | 9.3 / 9.9 (53%) | 9.4 / 10.1 (53%) | 12.0 / 11.8 (43%) | 16.4 / 16.4 (32%) |
| 2112x7168 | DSA fused_qkv_a | 8.1 / 8.1 (74%) | 8.8 / 8.5 (68%) | 10.5 / 9.1 (58%) | 12.8 / 10.4 (48%) | 12.9 / 10.9 (48%) | 15.4 / 12.8 (41%) | 21.1 / 21.2 (31%) |
| 7168x768 | shared down | 3.6 / 5.9 (88%) | 3.9 / 4.6 (83%) | 4.3 / 4.9 (76%) | 5.1 / 5.4 (65%) | 5.4 / 5.5 (62%) | 6.7 / 6.2 (52%) | 8.9 / 7.2 (40%) |
| 2304x1536 | DSA q_b | 3.6 / 5.6 (73%) | 4.2 / 5.4 (63%) | 4.9 / 5.1 (54%) | 5.5 / 5.6 (49%) | 5.7 / 5.6 (47%) | 6.8 / 6.2 (40%) | 9.1 / 7.2 (31%) |
| 4096x1536 | wq_index | 4.1 / 6.1 (84%) | 4.6 / 6.0 (75%) | 5.6 / 7.2 (62%) | 6.6 / 6.3 (53%) | 6.9 / 6.5 (51%) | 8.2 / 7.5 (44%) | 10.8 / 8.7 (34%) |
| 128x7168 | wk_index | 4.3 / 5.9 (44%) | 4.2 / 5.6 (45%) | 4.9 / 5.2 (39%) | 6.1 / 7.1 (33%) | 6.1 / 7.1 (33%) | 8.0 / 7.1 (26%) | 10.5 / 8.4 (21%) |
| 32x7168 | wv_index | 4.3 / 5.7 (39%) | 4.8 / 5.3 (36%) | 5.5 / 5.2 (31%) | 6.7 / 6.9 (27%) | 6.7 / 6.9 (27%) | 10.2 / 6.9 (19%) | 14.2 / 11.5 (14%) |
| 20480x7168 | lm_head shard | 44.4 / 43.9 (100%) | 44.7 / 44.5 (99%) | 46.2 / 45.9 (96%) | 55.0 / 50.2 (81%) | 56.1 / 51.4 (80%) | 61.3 / 57.1 (74%) | 74.7 / 106.4 (61%) |
| 8448x7168 | layer-0 gate_up | 21.3 / 24.0 (90%) | 24.0 / 25.5 (80%) | 26.1 / 28.4 (74%) | 28.0 / 34.0 (69%) | 28.3 / 35.1 (69%) | 33.7 / 40.0 (58%) | 42.2 / 43.4 (47%) |
| 7168x4224 | layer-0 down | 10.7 / 12.4 (97%) | 12.3 / 13.4 (85%) | 13.4 / 15.1 (78%) | 15.7 / 17.0 (67%) | 16.1 / 17.7 (66%) | 20.7 / 22.9 (52%) | 26.3 / 24.5 (42%) |
| 2880x7168 | drafter context_kv | 9.9 / 12.5 (76%) | 10.0 / 15.6 (77%) | 11.1 / 20.5 (69%) | 13.2 / 23.7 (59%) | 13.7 / 23.7 (57%) | 16.3 / 21.8 (49%) | 22.2 / 25.8 (37%) |
| 7168x35840 | drafter context_proj | 78.7 / 89.8 (97%) | 79.3 / 89.9 (96%) | 82.5 / 93.1 (93%) | 87.2 / 118.8 (88%) | 88.5 / 123.0 (87%) | 100.8 / 143.8 (77%) | 158.2 / 274.4 (50%) |

## Tests
- `python3 -m pytest tests/gemm/bf16_skinny_common -q` (every shape x 21 row counts against the fp64 reference,
  bitwise repeats, NaN / inf, graph replays after the warm-up with no compile, XCD placement)
- `python3 benchmarks/gemm/bf16_skinny_common/benchmark_linear.py [--shapes N,K ...] [--rows M ...]`
