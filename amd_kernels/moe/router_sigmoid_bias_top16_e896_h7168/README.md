# MoE router, E 896 x H 7168, sigmoid + correction bias + top-16

## What it does
One launch: `topk_weights fp32 [M, 16]`, `topk_ids int32 [M, 16]` from `x bf16 [M, 7168]`, the gate `W bf16 [896, 7168]`
(replicated) and the fp32 `e_score_correction_bias [896]`, M <= 192 (above that, launches of 192 rows): the bf16 GEMM
(fp32 sums), AITER's `biased_grouped_topk` with one group (sigmoid, + bias, top-16, renormalise, times
`routed_scaling_factor`) and, optionally (`aux`), the BM=16 MoE sort of those 16 experts in AITER's auxiliary layout
(`register_presorted_moe`: sorted_token_ids, sorted_weights, sorted_expert_ids, num_valid_ids, m_indices,
reverse_sorted) and the zeroed [M, 3584] bf16 buffer the atomic stage 2 accumulates into.

Contract (`reference.py`): logits = the bf16 products summed in fp32 (fp64 in the reference: the kernel's split-K sum order
differs, logits agree to ~1e-6); the top-16 is bit for bit `aiter.biased_grouped_topk` on the same logits (the
`topk_reg_kernel` serving runs): ids, slot order (choice descending, AITER's tie rank for equal choices), and weights
`sigmoid * (scale / sum)` with the 16-term sum in slot order. The sort is deterministic (rows of an expert in ascending
(token, slot) order, padding rows: token id M, weight 0, m_indices M; AITER's own sort orders an expert's rows by LDS
atomics) and bit for bit `moe_sort_reference`. Two runs give the same bits; no atomic adds into outputs (the sort's
token masks are bitwise ORs). Layouts of W: `row` (the checkpoint's) or `shuf` (a bit-exact permutation made once at
load: one DMA reads 1 KB in one piece).

Design: 196 workgroups each stream a (column tile x K slice) of W and the activations through LDS (DMA) into MFMA
(v_mfma_f32_16x16x32_bf16), store fp32 partials, and arrive on their tile's counter. Workgroup t < M then takes token t:
it polls the counters of the tiles holding each 256-expert group, loads the slab sums, and selects the top-16 in one wave
(pivot = the smallest row-of-16-lanes 4th-largest lane maximum; the keys at or above it, 16-70 of 896, are ranked
against each other in LDS; a plateau of exact ties takes a sort + 16-round slow path). With `aux` every token's wave then
ORs its bit into one 256-bit mask per routed expert, waits for all token flags and derives its rows from the masks.

## Entry points
- `router_sigmoid_bias_top16_e896_h7168`: the router (x, W, bias, routed_scaling_factor, optional `out`, `shuffled`,
  optional `aux`); an `amd_kernels.warm_up` entry, the first call of each mode compiles every row-tile variant.
- `make_aux`: the sort's output tensors of one M (AITER's capacity).
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout; replace the parameter once
  before graph capture.
- `router_logits`, `top16`: the GEMM half (summed logits) and the top-16 launch alone, for tests and benchmarks.
- `supports`: whether a weight is the [896, 7168] bf16 contiguous gate and the bias the fp32 [896].

`host.py` holds the row-tile -> (K splits, tile shape) table; `kernel.py` the launch, `topk.py` the top-16 stages,
`aux.py` the sort.

## Performance
PERF_TABLE

## Tests
- `python3 -m pytest tests/moe/router_sigmoid_bias_top16_e896_h7168 -q`
- `python3 benchmarks/moe/router_sigmoid_bias_top16_e896_h7168/benchmark_router.py [M ...]`
