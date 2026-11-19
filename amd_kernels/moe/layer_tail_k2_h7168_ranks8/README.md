# moe/layer_tail_k2_h7168_ranks8

One decode layer's tail on each of 8 tensor-parallel ranks, in a single persistent kernel (256 CTAs, one launch):
`o_proj` (+ all-reduce) -> AttnRes (softmax mix over the block sources, RMSNorm) -> router (sigmoid top-k, bias) ->
shared SiTU MLP and routed MXFP4 experts (gate/up, down) -> latent all-gather + RMSNorm -> up projection -> all-reduce.
Rows 1 .. 16 (served: 6 and 12); H = 7168, latent 3584, 896 routed experts (each rank holds its TP shard).

```python
from amd_kernels.moe.layer_tail_k2_h7168_ranks8 import LayerTailRunner, LayerTailWeights, layer_tail
comm = LayerTailRunner(device, group, rank)          # collective, once
comm.step_begin()                                    # once a forward, before its first layer
out = layer_tail(comm, weights, core, prefix, blocks, nblocks=n, reset=False, layer=i)
```

`core` [S, 1536] is o_proj's input, `prefix` [S, 7168] is updated in place (written when `reset`, a block-write
layer), `blocks` [S, B, 7168] the AttnRes sources. Weights are used as vLLM holds them (no copies, no requantization).
`layer_tail` is a `warm_up` entry: its first call of a mode (nblocks, reset) compiles the served widths 4 .. 16 with
nothing launched.

## Design
- Every CTA does o_proj tasks (two rounds of 256) with its weights in flight at once; the router, latent and shared
  gate/up weights are loaded under them, before x exists (x arrives through flag-tagged words, no extra barrier).
- Stages hand data over by flag-tagged words in a per-rank scratch and the peers' mailboxes (store + local poll,
  about 0.7 us a hop); no grid barrier. A wait that gives up sets a health word instead of hanging.
- Routed experts: one task per (picked expert, 768-row group); weights issued a task ahead, w2 two experts a wave.
- Numerics are the stock K2's, bit for bit (same accumulation order, same bf16 roundings).

## Tests and benchmark
`tests/moe/layer_tail_k2_h7168_ranks8`: `test_layer_tail.py` (8 GPUs: pure-torch `reference.py` at 12/6/16/8 rows, ranks
agree, graph replay), `test_warm_up.py` (no late compile, nothing launched). Benchmark with roofline:
`benchmarks/moe/layer_tail_k2_h7168_ranks8/benchmark_layer_tail.py` (torchrun, 8 ranks).
