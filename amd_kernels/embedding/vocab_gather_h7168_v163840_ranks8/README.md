# Vocabulary embedding by owner reads on eight ranks, hidden 7168, vocabulary 163840

## What it does

Reads each token's embedding row straight from the rank that owns its vocabulary row: eight TP ranks keep their bf16
vocabulary shards (eight equal contiguous `[20480, 7168]` shards in rank order, no added-vocabulary gaps) and peers map
them once at load time over HIP IPC. One stateless kernel in place of the mask / gather / fill + all-reduce of
`VocabParallelEmbedding`: no weight copy, no payload sum, no staging buffer and no per-call handshake. Token `t` reads
shard `t // 20480`, row `t % 20480`; ids are a contiguous 1-D int32 or int64 tensor; the output is a contiguous bf16
`[rows, 7168]` on each rank. Caller: the Kimi-K3.1 target embedding (TP8). Contract (`reference.py`): finite rows bit
for bit; bf16 subnormals and signed zeros become +0 (AITER's fp32 flush when it adds the one owner to zeros); invalid
ids (< 0 or >= 163840) write +0; replays and eager calls are bitwise equal on every rank.

## Entry points

- `VocabGather`: the collective load-time mapping of the eight shards, built on every rank after selecting its GPU and
  after this rank's shard reaches its final allocation; it checks equal shard sizes, publishes the GPU-written weight
  to peers and exchanges the IPC handles. The weights stay alive and unchanged until `close()`.
- `VocabGather.__call__`: the `[rows, 7168]` bf16 rows of the ids, eager or inside a CUDA graph; no inter-rank
  synchronization per call.
- `VocabGather.warm_up`: compiles the int32 and int64 variants (one per id dtype, any row count).
- `VocabGather.close`: collective on every rank, after all graph use.

No dispatch: one kernel per id dtype (`host.py`), rows a runtime argument.

## Performance

MI355X, eight ranks, graph replay of 20 distinct random token batches over a warm shard, minimum of nine rounds, MAX over
ranks, us per call. Baseline = vLLM's VocabParallelEmbedding in the same graph: compiled mask, `F.embedding`,
`masked_fill_` and AITER's custom all-reduce. Ceiling = launch gap (1.6 us) + one remote read round trip (1.0 us) + 7/8 of
M x 14336 B at the measured 365 GB/s per-GPU ingress of an 8-GPU all-read (balanced links; random rows load the busiest
link ~30 % above the mean at M 192).

| M | 1 | 12 | 24 | 40 | 65 | 80 | 128 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| vLLM | 13.2 | 18.8 | 18.0 | 18.8 | 22.3 | 26.3 | 28.4 | 35.9 | 43.3 |
| this entry | 2.3 | 2.9 | 3.4 | 4.2 | 5.5 | 6.3 | 9.4 | 12.9 | 16.3 |
| ceiling | 2.6 | 3.0 | 3.4 | 4.0 | 4.8 | 5.3 | 7.0 | 9.2 | 11.4 |
| % of ceiling | 114 | 106 | 99 | 95 | 88 | 85 | 75 | 71 | 70 |

Threads per row (128 / 224 / 448 / 896, 7 / 4 / 2 / 1 workgroups per row) were swept at M 12 / 80 / 192: within 6 %.

## Tests

```bash
python3 -m pytest tests/embedding/vocab_gather_h7168_v163840_ranks8 -q          # 8 GPUs
python3 benchmarks/embedding/vocab_gather_h7168_v163840_ranks8/benchmark_gather.py   # 8 GPUs
```
