# DCP8 LSE combine fused with the W_UV up-projection (MLA, 96 heads, kv_lora 512)

## What it does

Decode with DCP8 on 8 GPUs: every rank holds the partial latent attention output `x_r [T, >=96, 512]` (bf16) and the
partial LSE `lse_r [T, >=96]` (fp32) of its KV shard for all 96 heads (T <= 256). One launch replaces vLLM's v_up-first
sequence (`dcp_vup_rs_combine`: Triton LSE mask, AITER LSE all-gather, batched `x @ W_UV` GEMM, Triton scale, AITER
reduce-scatter) and returns this rank's heads 12 d .. 12 d + 11 as `out [T, 12, 128]` bf16, optionally multiplied by
`sigmoid(gate)` (the DSA output gate, replacing `_gate_mul_kernel`).
Per head, `v_s = bf16(x_s @ W_UV)` (fp32 accumulate) is pushed over xGMI straight into the owner's slot; the owner
weighs it by `exp(lse_s - max) / sum` over the 8 sources (LSE masked to -inf for empty shards and padding rows) and sums
in source order in fp32, rounded once. `reference.py` is the contract (docstring: rounding, masks, tolerance: cosine >
0.9999 and max error <= 2^-7 of the row max against the reference; two runs match bit for bit). No atomics.
`W_UV` is read transposed ([96, 128, 512], `prepare_w_uv`); the heads' order is DCP-rank-major.

## Entry points

- `DcpVupCombine`: the communicator (a collective: uncached IPC pools, control blocks, peer table), one per stream, built
  before graph capture; `is_lse_base_on_e` selects exp / exp2.
- `prepare_w_uv`: the one-time weight layout for the entry (replaces the replicated `W_UV_dcp`).
- `dcp8_lse_combine_vup`: the entry (`amd_kernels.warm_up`): one call per DSA layer on every rank in the same order;
  `seq_lens` / `query_start_loc` are optional, `gate` selects the gated output. The first call of each mode compiles
  the 16 token tiers (16 rows each).
The tile count follows `ceil(T / 16)` in `host.py`; the protocol and the layout are in the module docstring of `kernel.py`.

## Performance

8 x MI355X, CUDA-graph replay, inputs rotating over > 300 MB, MAX over ranks, us per call (median / min of 9),
baseline: vLLM's `dcp_vup_rs_combine` with the AITER comm on the same inputs. Ceiling: launch 1.6 us + first-tile load +
max(flag 0.8 us + bytes sent / 343 GB/s, rest of x / 6.4 TB/s) + 0.6 us tail (constants measured on MI355X).

| T | sent MB | ceiling us | stock us | fused us | speedup | % of ceiling | stock + gate us | fused gated us |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.26 | 6.2 | 24.5 / 23.4 | 12.5 / 11.0 | 1.97x | 56% | 26.8 / 25.9 | 13.8 / 12.3 |
| 24 | 0.52 | 7.0 | 25.9 / 25.0 | 14.6 / 12.5 | 1.77x | 56% | | |
| 40 | 0.87 | 8.0 | 28.7 / 27.8 | 17.6 / 15.9 | 1.63x | 50% | | |
| 65 | 1.42 | 9.6 | 31.1 / 29.5 | 17.9 / 16.9 | 1.74x | 57% | | |
| 80 | 1.75 | 10.6 | 32.8 / 31.1 | 17.7 / 17.2 | 1.85x | 61% | 36.0 / 34.5 | 21.0 / 19.2 |
| 128 | 2.80 | 13.6 | 41.7 / 39.3 | 22.9 / 21.2 | 1.82x | 64% | | |
| 192 | 4.19 | 17.7 | 45.9 / 44.7 | 28.4 / 26.3 | 1.62x | 67% | 48.8 / 46.0 | 28.3 / 27.3 |
| 240 | 5.24 | 20.7 | 53.3 / 51.9 | 33.0 / 31.5 | 1.62x | 66% | 55.6 / 54.5 | 32.9 / 30.6 |
| 256 | 5.59 | 21.8 | 52.7 / 51.5 | 32.5 / 31.4 | 1.62x | 69% | 57.7 / 55.7 | 34.7 / 33.5 |

Speedup is median / median, the ceiling share is ceiling / min. Output vs the stock sequence: cosine 1.000000.

## Tests

```bash
python3 -m pytest tests/attention/dcp8_lse_combine_vup_mla_h12_kv512 -q                    # 8 GPUs
AKR_COMPILE_ONLY=1 python3 -m pytest tests/attention/dcp8_lse_combine_vup_mla_h12_kv512/test_compile.py -q   # 1 GPU
python3 benchmarks/attention/dcp8_lse_combine_vup_mla_h12_kv512/benchmark_combine.py [--gate]    # 8 GPUs
```
