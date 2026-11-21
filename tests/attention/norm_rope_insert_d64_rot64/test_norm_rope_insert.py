# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Both kernels against the contract in ``norm_rope_insert_d64_rot64/reference.py``.

Served shapes (README): a draft layer at TP8 (12 query heads, 2 KV heads, head dim 64), 5 layers, bf16 or e4m3
caches. The qkv kernel sees the draft's query block (requests x k rows, padded to the graph sizes); the multi-layer
kernel the context rows of a step (requests x (k + 1) in decode, a prefill chunk). Every case: q and every cache
element against the reference, the elements the kernel must not write unchanged (skipped slots, other tokens),
three launches equal bit for bit, and a CUDA graph replayed on refilled outputs.
"""

import functools

import pytest
import torch

from amd_kernels.attention.norm_rope_insert_d64_rot64 import (
    kv_norm_rope_insert_multilayer as ml_host,
)
from amd_kernels.attention.norm_rope_insert_d64_rot64.kv_norm_rope_insert_multilayer import (
    kv_norm_rope_insert_multilayer,
    supports_multilayer,
)
from amd_kernels.attention.norm_rope_insert_d64_rot64.qkv_norm_rope_insert import (
    qkv_norm_rope_insert,
    supports_qkv_norm_rope_insert,
)
from amd_kernels.attention.norm_rope_insert_d64_rot64.reference import (
    D,
    check_bf16,
    check_cache,
    kv_norm_rope_insert_multilayer_reference,
    qkv_norm_rope_insert_reference,
)

EPS = 1e-5  # the draft's rms_norm_eps
CS_ROWS = 4096
LAYERS = 5
NQ, NKV = 12, 2
CACHE_DTYPES = {"bf16": torch.bfloat16, "fp8": torch.uint8, "e4m3": torch.float8_e4m3fn}


def require_gfx950():
    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        pytest.skip("requires gfx950")


def rope_cache(gen, rows=CS_ROWS):
    ang = torch.rand((rows, 32), generator=gen, dtype=torch.float64) * 6.3
    return torch.cat([ang.cos(), ang.sin()], dim=1).bfloat16().cuda()


@functools.cache
def scale_pair(scale):
    """Device scalars, made once (a CUDA graph cannot copy them in)."""
    return torch.tensor(scale, device="cuda"), torch.tensor(scale * 1.3, device="cuda")


def random_cache(shape, kind, gen):
    if kind == "bf16":
        return (torch.randn(shape, generator=gen) * 3).bfloat16().cuda()
    return torch.randint(0, 256, shape, generator=gen, dtype=torch.uint8).cuda().view(CACHE_DTYPES[kind])


def graph_replay(fn, refill):
    """Capture ``fn``, refill its outputs, replay twice (yields after each replay)."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    for _ in range(2):
        refill()
        graph.replay()
        torch.cuda.synchronize()
        yield


def same(a, b):
    return torch.equal(a.view(torch.uint8), b.view(torch.uint8))


# (id, tokens, page, cache (None: no insert), scale, slots given (None: all))
QKV_CASES = [
    ("t1_p16_bf16", 1, 16, "bf16", 1.0, None),
    ("t7_p1280_bf16", 7, 1280, "bf16", 1.0, None),
    ("t10_p1280_bf16", 10, 1280, "bf16", 1.0, None),
    ("t32_p1152_fp8", 32, 1152, "fp8", 1.0, None),
    ("t52_p1152_e4m3", 52, 1152, "e4m3", 0.37, None),
    ("t80_p16_fp8", 80, 16, "fp8", 1.0, None),
    ("t192_p1152_fp8", 192, 1152, "fp8", 0.37, None),
    ("t240_p1152_fp8_fewer_slots", 240, 1152, "fp8", 1.0, 192),
    ("t240_p64_bf16_fewer_slots", 240, 64, "bf16", 1.0, 200),
    ("t2048_p64_bf16", 2048, 64, "bf16", 1.0, None),
    ("t240_noinsert", 240, 16, None, 1.0, None),
]


@pytest.mark.parametrize("name,n,page,kind,scale,n_slots", QKV_CASES, ids=[c[0] for c in QKV_CASES])
@torch.inference_mode()
def test_qkv_norm_rope_insert(name, n, page, kind, scale, n_slots):
    require_gfx950()
    gen = torch.Generator().manual_seed(n * 10 + page)
    n_pages = (2 * n + page - 1) // page + 2
    qkv = (torch.randn((n, (NQ + 2 * NKV) * D), generator=gen) * 2).bfloat16().cuda()
    pos = torch.randint(0, CS_ROWS, (n,), generator=gen).cuda()
    pos[::5] -= CS_ROWS  # negative positions wrap
    slots = torch.randperm(n_pages * page, generator=gen)[: n_slots or n].cuda()
    slots[::7] = -1
    qw = (torch.randn(D, generator=gen) * 0.5).bfloat16().cuda()
    kw = (torch.randn(D, generator=gen) * 0.5).bfloat16().cuda()
    cs = rope_cache(gen)
    ks, vs = scale_pair(scale)
    insert = kind is not None
    cache0 = random_cache((n_pages, NKV, page, 2 * D), kind or "bf16", gen)
    if insert:
        assert supports_qkv_norm_rope_insert(qkv, cs, cache0)
    args = (qkv, pos, qw, kw, cs, EPS, NQ, NKV)
    q_ref, ins = qkv_norm_rope_insert_reference(*args, cache0 if insert else None, slots, ks, vs)

    caches = [cache0.clone() for _ in range(3)]
    qs = [torch.full((n, NQ * D), float("nan"), dtype=torch.bfloat16, device="cuda") for _ in range(3)]
    for q, cache in zip(qs, caches):
        if insert:
            qkv_norm_rope_insert(*args, q, cache, slots, ks, vs)
        else:
            qkv_norm_rope_insert(*args, q)
    torch.cuda.synchronize()
    frac_q = check_bf16(qs[0], q_ref, name)
    frac_c = check_cache(caches[0], ins[0], cache0, ins[1], name) if insert else 1.0
    if not insert:
        assert same(caches[0], cache0), name
    print(f"{name}: q bit-equal {frac_q:.6f}, cache bit-equal {frac_c:.6f}")
    for q, cache in zip(qs[1:], caches[1:]):
        assert same(q, qs[0]) and same(cache, caches[0]), name

    q, cache = qs[1], caches[1]

    def run():
        if insert:
            qkv_norm_rope_insert(*args, q, cache, slots, ks, vs)
        else:
            qkv_norm_rope_insert(*args, q)

    def refill():
        q.fill_(float("nan"))
        cache.copy_(cache0)

    for _ in graph_replay(run, refill):
        assert same(q, qs[0]) and same(cache, caches[0]), name


@torch.inference_mode()
def test_qkv_strided_rows():
    """qkv as a column slice of a wider row (the projection output with padding columns)."""
    require_gfx950()
    gen = torch.Generator().manual_seed(11)
    n, page = 40, 16
    wide = (torch.randn((n, (NQ + 2 * NKV) * D + 64), generator=gen) * 2).bfloat16().cuda()
    qkv = wide[:, : (NQ + 2 * NKV) * D]
    pos = torch.randint(0, CS_ROWS, (n,), generator=gen).cuda()
    slots = torch.randperm(8 * page, generator=gen)[:n].cuda()
    w = (torch.randn(D, generator=gen) * 0.5).bfloat16().cuda()
    cs = rope_cache(gen)
    cache0 = random_cache((8, NKV, page, 2 * D), "bf16", gen)
    q_ref, (c_ref, mask) = qkv_norm_rope_insert_reference(qkv, pos, w, w, cs, EPS, NQ, NKV, cache0, slots)
    q = torch.empty((n, NQ * D), dtype=torch.bfloat16, device="cuda")
    cache = cache0.clone()
    qkv_norm_rope_insert(qkv, pos, w, w, cs, EPS, NQ, NKV, q, cache, slots)
    torch.cuda.synchronize()
    check_bf16(q, q_ref, "strided")
    check_cache(cache, c_ref, cache0, mask, "strided")


# (id, context rows, page, cache, per-layer slot mappings, rows inserted (None: all))
ML_CASES = [
    ("n1_p16_bf16", 1, 16, "bf16", False, None),
    ("n8_p1280_bf16", 8, 1280, "bf16", False, None),
    ("n12_p1280_bf16", 12, 1280, "bf16", True, None),
    ("n40_p1152_fp8", 40, 1152, "fp8", False, None),
    ("n65_p1152_e4m3", 65, 1152, "e4m3", False, None),
    ("n100_p1152_fp8", 100, 1152, "fp8", True, None),
    ("n240_p1152_fp8", 240, 1152, "fp8", False, None),
    ("n240_p16_bf16", 240, 16, "bf16", False, 200),
    ("n4096_p1152_fp8", 4096, 1152, "fp8", True, 4000),
    ("n10240_p1152_fp8", 10240, 1152, "fp8", False, None),
    ("n100_p1152_fp8_dummy", 100, 1152, "fp8", False, 0),
]


def make_multilayer(n, page, kind, layer_slots, gen):
    n_pages = (2 * n + page - 1) // page + 2
    kv = (torch.randn((n, LAYERS * 2 * NKV * D), generator=gen) * 2).bfloat16().cuda()
    pos = torch.randint(0, CS_ROWS, (n,), generator=gen).cuda()
    weights = (torch.randn((LAYERS, D), generator=gen) * 0.5).bfloat16().cuda()
    cs = rope_cache(gen)
    shared = torch.randperm(n_pages * page, generator=gen)[:n].cuda()
    shared[::7] = -1
    if layer_slots:
        slots = [torch.randperm(n_pages * page, generator=gen)[:n].cuda() for _ in range(LAYERS)]
        for s in slots:
            s[::5] = -1
    else:
        slots = [shared] * LAYERS
    caches0 = [random_cache((n_pages, NKV, page, 2 * D), kind, gen) for _ in range(LAYERS)]
    if kind == "bf16":
        ks = vs = None
    else:
        ks = [torch.tensor(0.75 + 0.1 * i, device="cuda") for i in range(LAYERS)]
        vs = [torch.tensor(1.25 - 0.1 * i, device="cuda") for i in range(LAYERS)]
    return kv, pos, weights, cs, caches0, slots, ks, vs


@pytest.mark.parametrize("name,n,page,kind,layer_slots,n_insert", ML_CASES, ids=[c[0] for c in ML_CASES])
@torch.inference_mode()
def test_kv_norm_rope_insert_multilayer(name, n, page, kind, layer_slots, n_insert):
    require_gfx950()
    gen = torch.Generator().manual_seed(n + page)
    kv, pos, weights, cs, caches0, slots, ks, vs = make_multilayer(n, page, kind, layer_slots, gen)
    assert supports_multilayer(caches0, ks, vs)
    refs = kv_norm_rope_insert_multilayer_reference(kv, pos, weights, cs, caches0, slots, ks, vs, EPS, n_insert)
    runs = [[c.clone() for c in caches0] for _ in range(3)]
    for caches in runs:
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, slots, ks, vs, EPS, n_insert=n_insert)
    torch.cuda.synchronize()
    fracs = [
        check_cache(got, ref, before, mask, f"{name} layer {i}")
        for i, (got, (ref, mask), before) in enumerate(zip(runs[0], refs, caches0))
    ]
    print(f"{name}: cache bit-equal (per layer) {', '.join(f'{f:.6f}' for f in fracs)}")
    for caches in runs[1:]:
        assert all(same(a, b) for a, b in zip(caches, runs[0])), name

    caches = runs[1]

    def run():
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, slots, ks, vs, EPS, n_insert=n_insert)

    def refill():
        for c, c0 in zip(caches, caches0):
            c.copy_(c0)

    for _ in graph_replay(run, refill):
        assert all(same(a, b) for a, b in zip(caches, runs[0])), name


@torch.inference_mode()
def test_multilayer_pointer_table_cache():
    """The per-layer cache / scale pointer table is cached by the persistent pointers only; new eager slot views at
    other offsets reuse it."""
    require_gfx950()
    gen = torch.Generator().manual_seed(3)
    kv, pos, weights, cs, caches, slots, ks, vs = make_multilayer(68, 16, "fp8", False, gen)
    kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, slots, ks, vs, EPS)
    misses = ml_host._pointer_table.cache_info().misses
    for offset in (1, 3):
        views = [torch.cat([torch.zeros(offset, device="cuda", dtype=s.dtype), s])[offset:] for s in slots]
        kv_norm_rope_insert_multilayer(kv, pos, weights, cs, caches, views, ks, vs, EPS)
        assert ml_host._pointer_table.cache_info().misses == misses


def test_supports_multilayer():
    def caches(n=LAYERS, shape=(4, 2, 16, 128), dtype=torch.uint8):
        return [torch.zeros(shape, dtype=dtype) for _ in range(n)]

    scales = [torch.ones(1) for _ in range(LAYERS)]
    assert supports_multilayer(caches(), scales, scales)
    assert supports_multilayer(caches(dtype=torch.float8_e4m3fn), scales, scales)
    assert supports_multilayer(caches(dtype=torch.bfloat16), None, None)
    strided = torch.zeros(4, 2, 16, 256, dtype=torch.uint8)[..., :128]
    half = [torch.ones(1, dtype=torch.float16)] * LAYERS
    for bad, ks in (
        ([], []),
        (caches(dtype=torch.float16), scales),
        (caches(shape=(4, 2, 16, 256)), scales),
        (caches(shape=(0, 2, 16, 128)), scales),
        (caches(LAYERS - 1) + caches(1, (4, 1, 16, 128)), scales),
        (caches(LAYERS - 1) + caches(1, dtype=torch.float8_e4m3fn), scales),
        (caches(LAYERS - 1) + [strided], scales),
        (caches(), half),
        (caches(), scales[:-1]),
        (caches(), None),
    ):
        assert not supports_multilayer(bad, ks, None if ks is None else scales[: len(bad)])


def test_rope_blocker():
    from amd_kernels.attention.norm_rope_insert_d64_rot64 import rope_blocker

    cs = torch.empty(4096, 64, dtype=torch.bfloat16)
    assert rope_blocker(64, 64, True, cs) is None
    for hs, rot, neox, c in ((64, 32, True, cs), (128, 64, True, cs), (64, 64, False, cs), (64, 64, True, cs.float()),
                             (64, 64, True, torch.empty(4096, 128, dtype=torch.bfloat16)),
                             (64, 64, True, torch.empty(64, 4096, dtype=torch.bfloat16).T)):
        assert rope_blocker(hs, rot, neox, c) is not None, (hs, rot, neox, tuple(c.shape))
