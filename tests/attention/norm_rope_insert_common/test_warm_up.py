# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Through the entries: the first call warms the routed package up (``amd_kernels.warm_up``), every served row count
after it compiles nothing (``compile_watch``), and a CUDA graph of the entry call, captured with imports disabled,
replays the eager call bit for bit."""

import importlib

import pytest
import torch

from amd_kernels import compile_watch, warm_up
from amd_kernels.attention import norm_rope_insert_common as E

from . import cases

QKV_ROWS = [1, 7, 10, 32, 52, 80, 192, 240, 2048]
ML_ROWS = [1, 8, 12, 40, 65, 100, 240, 4096, 10240]


@pytest.fixture
def auto_warm_up():
    if not E.available():
        pytest.skip("needs gfx950 with FlyDSL")
    warm_up.set_auto(True)
    yield
    warm_up.set_auto(False)


def walk(name, builds):
    builds[0]()[0].run(True)
    torch.cuda.synchronize()
    assert any(entry == name for entry, _ in warm_up.warmed()), warm_up.warmed()
    warned, late = [], warm_up.late_compiles()
    watch = compile_watch.late_compiles()
    assert compile_watch.arm(warned.append)
    try:
        for build in builds[1:]:
            build()[0].run(True)
        torch.cuda.synchronize()
    finally:
        compile_watch.disarm()
    assert not warned and compile_watch.late_compiles() == watch and warm_up.late_compiles() == late, warned


def graph_replays_eager(build, monkeypatch):
    a, _ = build()
    b, _ = build()
    a.run(True)
    b.run(True)  # the multi-layer pointer table is built on a first eager call, as in serving

    def no_import(name, *args, **kwargs):
        raise AssertionError(f"import of {name} inside capture")

    for x in b.outputs():
        x.zero_()
    g = torch.cuda.CUDAGraph()
    with monkeypatch.context() as m:
        m.setattr(importlib, "import_module", no_import)
        with torch.cuda.graph(g):
            b.run(True)
    g.replay()
    torch.cuda.synchronize()
    for x, y in zip(a.outputs(), b.outputs(), strict=True):
        assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@torch.inference_mode()
def test_qkv(auto_warm_up, monkeypatch, kind):
    walk("attention/norm_rope_insert_d64_rot64:qkv", [lambda n=n: cases.qkv(n, kind=kind, seed=n) for n in QKV_ROWS])
    graph_replays_eager(lambda: cases.qkv(52, kind=kind, seed=3), monkeypatch)


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@torch.inference_mode()
def test_multilayer(auto_warm_up, monkeypatch, kind):
    walk("attention/norm_rope_insert_d64_rot64:multilayer",
         [lambda n=n: cases.multilayer(n, kind=kind, seed=n) for n in ML_ROWS])
    graph_replays_eager(lambda: cases.multilayer(100, kind=kind, seed=3), monkeypatch)
