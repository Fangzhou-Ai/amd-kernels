# bf16 owner-row all-to-all and all-gather of logits, 20480 columns per source, eight ranks

## What it does
`RowAllToAll` moves the vocabulary-sharded logits rows of Kimi-K3.1 TP8 to the rank that owns each request, straight
into the final `[owned rows, 163840]` layout (the batch-sharded sampler's input), in place of RCCL's `all_to_all_single`
+ permute; with `counts=None` every rank receives every row: the all-gather of `tensor_model_parallel_all_gather(logits)`
-> `[M, 163840]`. The senders write into the receivers' buffers (push) and the receiver only waits on eight flag words in
its own memory. Input: bf16 `[rows, 20480]`, rows <= 256, sorted stably by owner. Contract (`reference.py`): rank r
receives rows `[sum(counts[:r]), sum(counts[:r + 1]))` of every source, concatenated along the vocabulary, every bf16
bit kept (NaN payloads too). Protocol (module docstring of `kernel.py`): a receive buffer and a 512-byte uncached flag
block per rank mapped into every peer over HIP IPC, system-scope stores, a two-level ticket, a device sequence counter
(no host state in a graph). The rows are valid until the next exchange. No atomic output adds.

## Entry points
- `RowAllToAll`: a collective; a receive buffer of `max_rows x 163840 x 2` B per rank (84 MB at 256 rows).
- `RowAllToAll.warmup`: a collective before capture; compiles both pushes and the wait.
- `RowAllToAll.__call__`: push of rows `[rows, 20480]` (owner-sorted with `counts`, a CUDA int32 `[8]` the same on every
  rank; or all-gather without) and wait; read the rows with `recv(n)`.
- `RowAllToAll.send_args`, `RowAllToAll.wait`, `RowAllToAll.recv`: for a producer that pushes the rows itself
  (`gemm/bf16_n20480_k7168`), then the wait and the in-place view.
- `RowAllToAll.close`: a collective, after every graph that uses the communicator is retired.

All ranks call every exchange in the same order, empty receivers included; one communicator per stream.

## Performance

MI355X, eight ranks, `benchmark_exchange.py`: hidden rows replicated, owner-sorted rows (request r on rank r % 8, 4 rows per
request), the lm_head GEMM included in every column, graph replay of 6 distinct inputs over three weight copies (880 MB),
minimum of nine rounds, MAX over ranks, us per call. `vLLM AG` = `rocm_unquantized_gemm_impl` +
`tensor_model_parallel_all_gather` (served today); `vLLM A2A` = GEMM + `all_to_all_single` + permute (vLLM's batch-sharded
sampling); `this AG` / `this owner` = `bf16_gemm_n20480_k7168` pushing to every rank / to the owners + `wait` (rows read in
place). Ceiling = the GEMM at its HBM or MFMA ceiling (6.5 TB/s / 1.5 PF) + 2.4 us (wait launch gap 1.6 + one flag hop 0.8)
+ the busiest link's bytes at 49 GB/s (the measured 8-GPU all-to-all, 343 GB/s out over seven links).

| M | 12 | 24 | 40 | 65 | 80 | 128 | 192 |
|---|---|---|---|---|---|---|---|
| vLLM AG | 88.3 | 102.6 | 137.0 | 175.3 | 193.3 | 257.8 | 354.1 |
| vLLM A2A | 76.4 | 72.6 | 86.8 | 95.1 | 99.2 | 105.7 | 129.5 |
| this AG | 61.8 | 81.2 | 99.4 | 137.3 | 158.1 | 224.4 | 308.1 |
| this owner | 53.1 | 54.6 | 58.7 | 63.9 | 68.0 | 83.8 | 116.3 |
| owner vs AG / vs A2A | 1.66x / 1.44x | 1.88x / 1.33x | 2.33x / 1.48x | 2.74x / 1.49x | 2.84x / 1.46x | 3.08x / 1.26x | 3.04x / 1.11x |
| owner ceiling | 50.9 | 51.0 | 54.3 | 55.2 | 57.8 | 61.2 | 68.1 |
| owner % of ceiling | 96 | 93 | 93 | 86 | 85 | 73 | 59 |
| AG % of ceiling | 93 | 83 | 82 | 74 | 73 | 69 | 68 |

The part above the ceiling at M >= 128 is mostly the GEMM (80 us at 192 against a 47 us ceiling: `gemm/bf16_n20480_k7168`'s README);
the exchange itself: owner mode costs 9 us (M 12) to 36 us (M 192) over the plain GEMM, ~87 % of (our GEMM + the exchange
ceiling) at M 192. Skeleton measurements of the earlier pull design (stage 3.6-6.1 us, pull 6.1 / 16.7 / 29.6 us at M 12 / 80 /
192, all-gather pull at 94 % of the measured 8-GPU ingress): the push cut one remote round trip (about -10 us at small M).

## Tests
```bash
python3 -m pytest tests/collectives/all_to_all_bf16_n20480_ranks8 -q               # 8 GPUs
python3 benchmarks/collectives/all_to_all_bf16_n20480_ranks8/benchmark_exchange.py # 8 GPUs
```
