# MoE router, E 896 x H 7168, sigmoid + correction bias + top-16

## What it does
`topk_weights fp32 [M, 16]`, `topk_ids int32 [M, 16]` from `x bf16 [M, 7168]`, the gate `W bf16 [896, 7168]`
(replicated) and the fp32 `e_score_correction_bias [896]`: the Kimi-K3 router (`GateLinear` fp32 logits, then
AITER-style `biased_grouped_topk` with one group) in two launches, M <= 192 (above that, launches of 192 rows).
The GEMM launch writes fp32 split-K partials; the top-k launch sums them in slice order, applies sigmoid, adds the bias,
selects the 16 largest biased scores, renormalises the unbiased sigmoids and multiplies by `routed_scaling_factor`.
Slots are in ascending expert id; equal biased scores take the lowest ids (torch.topk's pick among exact ties changes
from run to run, so the unfused path has no rule to match). Contract (`reference.py`): logits = fp32 sum of the bf16
products (fp64 here: the kernel's logits differ by sum order, ~1e-6), ids equal the reference's on the kernel's own
logits, weights within 4 ulp (exp / divide rounding and the slot-order 16-term sum), two runs bit for bit; finite
inputs. No split-K atomics: the slabs are summed in a fixed order. Layouts of W: `row` (the checkpoint's) or `shuf`
(a bit-exact permutation made once at load: one DMA reads 1 KB in one piece); both give the same bits.

## Entry points
- `router_sigmoid_bias_top16_e896_h7168`: the router (x, W, bias, routed_scaling_factor, optional out, `shuffled`);
  an `amd_kernels.warm_up` entry, the first call of the process compiles every row-tile variant.
- `shuffle_weight`, `shuffle_weight_` (in place), `unshuffle_weight`: the `shuf` layout; replace the parameter once
  before graph capture.
- `router_logits`, `top16`: the two halves for tests and benchmarks (summed logits; top-16 of given logits).
- `supports`: whether a weight is the [896, 7168] bf16 contiguous gate and the bias the fp32 [896].

`host.py` holds the row-tile -> K split table; `kernel.py` the GEMM launch, `topk.py` the top-k launch.

## Performance
PERF_TABLE

## Tests
- `python3 -m pytest tests/moe/router_sigmoid_bias_top16_e896_h7168 -q`
- `python3 benchmarks/moe/router_sigmoid_bias_top16_e896_h7168/benchmark_router.py [M ...]`
