# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""After the first call's warm-up, no served row count compiles."""

import pytest
import torch
import torch.distributed as dist
from amd_kernels import warm_up
from amd_kernels.attention.dcp8_q_bproj_wuk_gather_h96_kv512 import (
    QueryGatherComm,
    pack_q_b,
    pack_w_uk,
    q_b_allgather_wuk_all_heads,
    q_b_wuk_allgather_latent,
)

from tests.attention.dcp8_q_bproj_wuk_gather_h96_kv512.multi_gpu8 import run_on_8_gpus

ROWS = (1, 5, 6, 10, 12, 15, 16, 17, 24, 30, 40, 45, 48, 49, 65, 80, 96, 97, 128, 160, 192, 193, 256)


def worker(rank, errors, mode):
    warm_up.set_auto(True)
    dev = torch.device("cuda", rank)
    heads = 12 if mode == "latent" else 96
    wq = pack_q_b(torch.zeros(2304, 1536, dtype=torch.bfloat16, device=dev))
    wu = pack_w_uk(torch.zeros(heads, 128, 512, dtype=torch.bfloat16, device=dev))
    comm = QueryGatherComm(rank, dist.group.WORLD, dev, mode)
    call = q_b_wuk_allgather_latent if mode == "latent" else q_b_allgather_wuk_all_heads
    call(comm, torch.zeros(1, 1536, dtype=torch.bfloat16, device=dev), wq, wu)
    before = warm_up.late_compiles()
    for m in ROWS:
        call(comm, torch.zeros(m, 1536, dtype=torch.bfloat16, device=dev), wq, wu)
    torch.cuda.synchronize()
    if warm_up.late_compiles() != before:
        errors.append(f"rank {rank}: {warm_up.late_compiles() - before} compiles after warm-up ({mode})")
    comm.close()


@pytest.mark.parametrize("mode", ["latent", "q"])
def test_no_compile_after_warm_up(mode):
    run_on_8_gpus(worker, mode, timeout=900)
