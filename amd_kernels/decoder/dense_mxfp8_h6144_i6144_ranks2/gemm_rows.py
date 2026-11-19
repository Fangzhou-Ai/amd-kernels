# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The MXFP8 GEMM rows of this block's down projection (6144 x 6144, read through ``gemm.mxfp8_common.tuned_rows``;
the shared table's 6144 x 6144 rows are a dense gate_up's). Split-K decode at 33-256 beats the decode rows at every
row count, its 225-256 row looks 2 W steps ahead (3 spills a VGPR, 1% slower); K-wave 2 decode rows at 257-320,
385-448 and 641-1024, 4wave_nt 256 x 128 at 1025-1280 (benchmarks/gemm/mxfp8_common/benchmark_decode_splitk.py,
sweep_rows.py)."""

DOWN_ROWS = """\
M,N,K,out_dtype,kernel,tile_m,tile_n,tile_k,waves_m,waves_n,waves_per_eu,xcd_swizzle,split_k,la_w,la_a,min_m,w_cm
8,6144,6144,bf16,decode_fq,8,32,128,2,0,0,0,1,6,3
16,6144,6144,bf16,decode,16,32,128,1,0,0,0,1,7,7
32,6144,6144,bf16,decode,32,32,128,1,0,0,0,1,,
64,6144,6144,bf16,decode_splitk,32,96,128,3,3,0,0,2,3,0,,
96,6144,6144,bf16,decode_splitk,48,96,128,3,3,0,0,2,3,0,,
128,6144,6144,bf16,decode_splitk,64,96,128,3,3,0,0,2,3,0,,
160,6144,6144,bf16,decode_splitk,80,96,128,3,3,0,0,2,3,0,,
192,6144,6144,bf16,decode_splitk,96,96,128,3,3,0,0,2,3,0,,
224,6144,6144,bf16,decode_splitk,112,96,128,3,3,0,0,2,3,0,,
256,6144,6144,bf16,decode_splitk,128,96,128,3,3,0,0,2,2,0,,
320,6144,6144,bf16,decode,48,96,128,2,0,0,0,1,,,,0
384,6144,6144,bf16,decode,96,96,128,3,0,0,0,1,,
448,6144,6144,bf16,decode,64,96,128,2,0,0,0,1,,,,0
512,6144,6144,bf16,decode,128,96,128,4,0,0,0,1,,
640,6144,6144,bf16,decode,160,96,128,3,0,0,0,1,,
1024,6144,6144,bf16,decode,128,96,128,2,0,0,0,1,,,,0
1280,6144,6144,bf16,4wave_nt,256,128,128,2,2,1,1,1,,
65536,6144,6144,bf16,4wave_nt,256,256,128,2,2,1,1,1,,
"""
