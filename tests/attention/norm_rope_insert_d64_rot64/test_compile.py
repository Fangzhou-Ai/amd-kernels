# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Nothing compiles after warm-up (root README, rule 4).

``qkv_norm_rope_insert`` compiles one variant per (query heads, KV heads, cache dtype or no insert, page size),
``kv_norm_rope_insert_multilayer`` one per (layers, KV heads, page size, cache dtype); a deployment fixes all of them,
and the token count, the slots given and the per-layer slot mappings are runtime arguments. Warm each variant once,
then probe the served token counts; a probe that compiles means the token count reaches a compile key.
"""

import flydsl.compiler as flyc
import pytest
import torch

from amd_kernels.attention.norm_rope_insert_d64_rot64.kv_norm_rope_insert_multilayer import (
    kv_norm_rope_insert_multilayer,
)
from amd_kernels.attention.norm_rope_insert_d64_rot64.qkv_norm_rope_insert import (
    qkv_norm_rope_insert,
)

from .test_norm_rope_insert import (
    EPS,
    NKV,
    NQ,
    make_multilayer,
    random_cache,
    require_gfx950,
    rope_cache,
    scale_pair,
)

D = 64
QKV_SERVED = [1, 7, 10, 32, 52, 80, 192, 240, 2048]
ML_SERVED = [1, 8, 12, 40, 65, 100, 240, 4096, 10240]


def count_compiles(monkeypatch):
    compiled = []
    real = flyc.compile

    def counting(fn, *args, **kwargs):
        compiled.append(getattr(fn, "kernel_name", type(fn).__name__))
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(flyc, "compile", counting)
    return compiled


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@torch.inference_mode()
def test_qkv_no_compile_after_warmup(kind, monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(1)
    n_max, page = QKV_SERVED[-1], 1152
    qkv = torch.randn((n_max, (NQ + 2 * NKV) * D), generator=gen).bfloat16().cuda()
    pos = torch.randint(0, 4096, (n_max,), generator=gen).cuda()
    slots = torch.randperm(2 * n_max, generator=gen).cuda()
    w = torch.randn(D, generator=gen).bfloat16().cuda()
    cs = rope_cache(gen)
    cache = random_cache(((2 * n_max + page - 1) // page, NKV, page, 2 * D), kind, gen)
    q = torch.empty((n_max, NQ * D), dtype=torch.bfloat16, device="cuda")
    ks, vs = scale_pair(1.0)

    def run(n, insert):
        args = (qkv[:n], pos[:n], w, w, cs, EPS, NQ, NKV, q[:n])
        if insert:
            qkv_norm_rope_insert(*args, cache, slots[:n], ks, vs)
        else:
            qkv_norm_rope_insert(*args)

    run(QKV_SERVED[0], False)  # the profile run (no slots)
    run(QKV_SERVED[0], True)  # the graph captures
    torch.cuda.synchronize()
    compiled = count_compiles(monkeypatch)
    for n in QKV_SERVED[1:]:
        run(n, True)
        run(n, False)
        assert not compiled, f"{n} tokens compiled {compiled}"


@pytest.mark.parametrize("kind", ["bf16", "fp8"])
@torch.inference_mode()
def test_multilayer_no_compile_after_warmup(kind, monkeypatch):
    require_gfx950()
    gen = torch.Generator().manual_seed(2)
    kv, pos, weights, cs, caches, slots, ks, vs = make_multilayer(ML_SERVED[-1], 1152, kind, True, gen)
    shared = [slots[0]] * len(slots)

    def run(n, layer_slots, n_insert=None):
        maps = [s[:n] for s in (slots if layer_slots else shared)]
        kv_norm_rope_insert_multilayer(kv[:n], pos[:n], weights, cs, caches, maps, ks, vs, EPS, n_insert=n_insert)

    run(ML_SERVED[0], False, 0)  # the dummy run (no valid slots)
    torch.cuda.synchronize()
    compiled = count_compiles(monkeypatch)
    for n in ML_SERVED[1:]:
        run(n, False)
        run(n, True)
        run(n, False, n // 2)
        assert not compiled, f"{n} rows compiled {compiled}"
