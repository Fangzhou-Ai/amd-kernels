# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Run a test body on 8 GPUs with ``tests/multi_gpu.py``."""

from tests.multi_gpu import run_on_gpus

WORLD = 8


def run_on_8_gpus(fn, *args, timeout=900):
    """``fn(rank, errors, *args)`` on 8 GPUs."""
    run_on_gpus(fn, *args, world=WORLD, timeout=timeout)
