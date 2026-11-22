# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The entry's warm-up (a fresh process) keeps no buffer: after the push mode's first call nothing it allocated stays
(its private exchange state lives as long as each warm-up call), and the compile-only calls make no split-K workspace;
the first split-K launch does, ``host.prepare`` ahead of it."""
import json
import os
import subprocess
import sys

import pytest
import torch


def scenario():
    from amd_kernels import warm_up
    from amd_kernels.gemm.bf16_n20480_k7168 import bf16_gemm_n20480_k7168, host
    from amd_kernels.gemm.bf16_n20480_k7168.reference import K, N

    warm_up.set_auto(True)
    dev = torch.device("cuda", torch.cuda.current_device())
    w = (torch.randn(N, K, device=dev) * 0.02).bfloat16()
    x = torch.zeros(256, K, dtype=torch.bfloat16, device=dev)
    recv = torch.zeros(256 * 8 * N, dtype=torch.bfloat16, device=dev)
    flag = torch.zeros(128, dtype=torch.int32, device=dev)
    peers = torch.tensor([recv.data_ptr()] * 16 + [flag.data_ptr()] * 8, dtype=torch.int64, device=dev)
    counter = torch.zeros(1, dtype=torch.int32, device=dev)
    rows = torch.arange(8, device=dev)
    host._dummies(dev)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated(dev)
    bf16_gemm_n20480_k7168(x, w, row_indices=rows, push=((peers, counter, flag.data_ptr(), 0, 0), None))
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated(dev) - base
    ws_after_warm_up = host._exchange_workspace.cache_info().currsize
    out = torch.empty(128, N, dtype=torch.bfloat16, device=dev)
    bf16_gemm_n20480_k7168(x[:128], w, out)
    torch.cuda.synchronize()
    ws_after_split_k = host._exchange_workspace.cache_info().currsize
    host.prepare(dev, 128)
    return {"kept": kept, "ws_after_warm_up": ws_after_warm_up, "ws_after_split_k": ws_after_split_k,
            "ws_after_prepare": host._exchange_workspace.cache_info().currsize}


def test_warm_up_keeps_no_buffer():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run([sys.executable, "-m", __name__], capture_output=True, text=True, timeout=1800,
                          env=dict(os.environ))
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    assert r["kept"] == 0, r
    assert r["ws_after_warm_up"] == 0 and r["ws_after_split_k"] == 1 and r["ws_after_prepare"] == 1, r


if __name__ == "__main__":
    print(json.dumps(scenario()))
