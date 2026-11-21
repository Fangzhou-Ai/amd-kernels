# RMS norm + RoPE + bf16 / fp8 KV insert, head dim 64, rotary 64

## What it does

Turns a projection's bf16 output into an attention layer's inputs in one launch: per-head RMS norm (fp32, weight
`[64]`), NeoX RoPE over all 64 dims (halves of 32), and the K / V insert into the cache pages (V is not normed or
rotated). Caller: the Kimi-K3.1 DSpark draft (5 GQA layers, 12 q / 2 KV heads a TP8 rank): one layer's query-block
qkv row, and the context rows' fused K / V row, which feeds all 5 layers' caches. Cache `[pages, kv_heads, page, 128]`
(K 64 | V 64 per token, any page / head / token strides): bf16 as is, or e4m3 of `value / scale` (IEEE divide, RNE,
saturated at +-448) with per-tensor fp32 scales; `cos_sin_cache [rows, 64]` bf16 (cos | sin). Contract:
`reference.py`. The multi-layer kernel rounds the normed K to bf16 before RoPE (vLLM's eager `rms_norm` +
`rotary_embedding` of the context store); the qkv kernel rotates the fp32 normed values (Inductor's fused norm + RoPE
of the compiled draft layer). Every output is within one bf16 ulp / e4m3 step of the reference or `ROW_ATOL` (2^-6)
of its row maximum, at least `EXACT_MIN` (99.9%) bit-equal; cache elements outside the written slots stay unchanged.

## Entry points

- `qkv_norm_rope_insert.qkv_norm_rope_insert`: q / k norm + RoPE of one layer's fused bf16 qkv rows into the caller's
  bf16 q, and the K / V insert of the tokens that have slots (slot < 0 skips); without a cache it writes q only (the
  profile run).
- `qkv_norm_rope_insert.supports_qkv_norm_rope_insert`: whether the inputs fit (gfx950, bf16, the cache layout).
- `qkv_norm_rope_insert.cache_kind`: "bf16", "fp8" or None for a cache tensor.
- `rope_blocker`: why a layer's RoPE does not fit these kernels (None: it fits).
- `kv_norm_rope_insert_multilayer.kv_norm_rope_insert_multilayer`: K norm + RoPE and the K / V inserts of the
  context rows' fused K / V row into every layer's cache (one slot mapping, or one a layer).
- `kv_norm_rope_insert_multilayer.supports_multilayer`: whether it takes these caches and scales.
- `kv_norm_rope_insert_multilayer.capture_blocker`: why a call cannot run in the CUDA graph being captured (the
  first call of a cache set builds its pointer table and must be eager).

Both are entries (`amd_kernels.warm_up`): the first call of each mode (heads, cache dtype or no insert, page size;
layers, KV heads, page size, cache dtype) compiles its kernel with slot -1 calls that write nothing. No row-count
dispatch: rows, positions and slots are runtime arguments. The shape-agnostic entry is
`attention/norm_rope_insert_common`.

## Performance

One MI355X; one graph of 20 calls with their own inputs and disjoint slots, min of 15 medians, us per call. Stock:
the compiled draft layer's norm + RoPE (4 Inductor launches) + `triton_reshape_and_cache_flash`; the context store
after its GEMM (copy, grouped `rms_norm`, repeat, `rotary_embedding`, 5 inserts: 9 launches), and for bf16 caches
the one-launch Triton store of `VLLM_K31_BEAT_CTXKV`. Floor: one copy launch moving the same bytes (1.95-2.09 us; an
empty graph launch is 1.57 us). Multi-layer: cache bytes bit-equal to the stock chain at every row. qkv: q and cache
bytes bit-equal when Inductor's variance reduction runs 8 rows a wave; Inductor autotunes it per process, and other
configs leave ~1e-5 of the q / K values one bf16 ulp off (the stock chain itself varies so between server launches).

| Rows (draft at TP8) | cache | qkv: T -> us (stock) | multi-layer: R -> us (stock; Triton store) |
|---|---|---|---|
| 1 request x 7 | bf16 | 7 -> 2.07 (9.30) | 8 -> 2.09 (18.44; 3.51) |
| 2 x 5 | bf16 | 10 -> 2.07 (9.71) | 12 -> 2.12 (18.41; 3.52) |
| 8 x 4 | e4m3 | 32 -> 2.23 (10.25) | 40 -> 2.22 (18.83) |
| 13 x 4 | e4m3 | 52 -> 2.23 (9.87) | 65 -> 2.20 (19.42) |
| 20 x 4 | e4m3 | 80 -> 2.23 (10.11) | 100 -> 2.22 (20.11) |
| 48 x 4 | e4m3 | 192 -> 2.27 (10.30) | 240 -> 2.27 (20.73) |
| prefill chunk | e4m3 | - | 10240 -> 16.43 (138.41) |

## Tests

```bash
python3 -m pytest tests/attention/norm_rope_insert_d64_rot64 -q
python3 benchmarks/attention/norm_rope_insert_d64_rot64/benchmark_norm_rope_insert.py   # stock columns need vLLM
```
