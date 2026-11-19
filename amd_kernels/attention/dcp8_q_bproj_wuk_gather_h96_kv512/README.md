# DCP8 DSA query path: q_b + W_UK + all-gather over the eight head shards

## What it does
The decode query of one DSA layer at TP8 / DCP8 (12 heads per rank, kv_lora 512, NoPE, q_pe 64 wide): `q_a` bf16
[M, 1536] (the RMSNorm'd q_c, identical on all ranks) -> q_b projection [1536 -> 12 x 192] -> per head `q_nope` [M, 128]
x W_UK [128, 512] -> the latent query bf16 [M, 96, 576] on every rank (head 12 r + h from rank r: `[latent 512 | q_pe
64]`), the tensor `triton_sparse_mla_fwd` reads. Stock vLLM runs five launches and an all-gather (skinny GEMM, `torch.bmm`,
`torch.cat`, AITER IPC all-gather of 110 KB per token). Both variants keep its rounding (fp32 accumulate, bf16 after
q_b, bf16 after W_UK; `reference.py`: cosine >= 0.9999, rel. L2 <= 4e-3, bitwise repeatable, no atomic adds into outputs):
- `q` (shipped, one launch): the q_b tiles are pushed (bf16 q [M, 12, 192], 4.5 KB per token and peer) into both
  data areas of all eight ranks by epoch parity; every rank applies W_UK to all 96 heads from a replicated packed weight
  (12.6 MB per layer) into the caller's `out` while the weights stream in behind the exchange.
- `latent`: q_b and the 12 local heads' W_UK, then the latent query is pushed (13.8 KB per token and peer): two
  launches, the result a view of the communicator's buffer. The measured alternative; it moves 3x the bytes.
Weights are packed once at load (`pack_q_b`, `pack_w_uk`: MFMA operand tiles).

## Entry points
- `QueryGatherComm`: the eight-rank communicator (IPC gather buffer, flag words, device-side epoch); created
  collectively before capture, one per stream, closed on all ranks after graph work completes.
- `q_b_allgather_wuk_all_heads`: the `q` variant, graph-safe; what a decode layer calls in place of q_b, W_UK, concat, gather.
- `q_b_wuk_allgather_latent`: the `latent` variant, graph-safe; the result is valid until the next call.
- `pack_q_b`, `pack_w_uk`: load-time packing. `host.config_q` / `host.config_latent` hold the row-count tiers.

## Performance
Eight MI355X, graph replay of 16 calls over rotated weights, min of 15 rounds of the slowest rank, us per call. Stock =
q_b GEMM + `torch.bmm` + `torch.cat` + AITER all-gather in the same harness. Floor = launch gap + q_b weight at 6.4 TB/s +
remote write ack + flag + the push at the 365 GB/s all-to-all rate + the consumer's q read and output write at the
measured HBM rates (`analysis/hw_ceilings.md`), phases in series. A skeleton (same loads, flags and stores, no MFMA)
takes the same time as `q` (within 8%) at every row count: the math is free, the rest is the dependence chain
(timeline in `analysis/kernels/dcp8_q_bproj_wuk_gather_h96_kv512.md`).

| M | stock | latent | q | q vs stock | q floor | % of floor |
|---|---|---|---|---|---|---|
| 12 | 24.0 | 13.3 | 11.1 | 2.16x | 6.1 | 55% |
| 24 | 27.6 | 17.8 | 14.1 | 1.96x | 7.3 | 52% |
| 40 | 32.6 | 23.9 | 16.3 | 2.00x | 9.0 | 55% |
| 65 | 42.0 | 32.7 | 20.8 | 2.02x | 11.7 | 56% |
| 80 | 46.0 | 37.5 | 22.3 | 2.06x | 13.3 | 60% |
| 128 | 58.5 | 54.0 | 30.7 | 1.91x | 18.4 | 60% |
| 192 | 75.6 | 74.7 | 40.9 | 1.85x | 25.3 | 62% |

## Tests
- `python3 -m pytest tests/attention/dcp8_q_bproj_wuk_gather_h96_kv512 -q` (8 GPUs, 120 s per call: `-k latent-rows`, `-k q-rows`, `test_compile.py`)
- `python3 benchmarks/attention/dcp8_q_bproj_wuk_gather_h96_kv512/benchmark_query_gather.py` (8 GPUs)
