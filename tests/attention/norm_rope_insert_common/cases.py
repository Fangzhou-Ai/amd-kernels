# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Served-shape calls of the norm + RoPE + insert entries and of the shape package they route to (tests and
benchmark): each builder returns fresh, seeded inputs, so two builds are equal bit for bit."""

import importlib

import torch

PKG = "amd_kernels.attention.norm_rope_insert_d64_rot64"
NQ, NKV, D, LAYERS, EPS, CS_ROWS = 12, 2, 64, 5, 1e-5, 4096


class Call:
    """One call: ``run(entry)`` launches it through the entry (True) or the shape package (False); ``outputs`` are
    every tensor the call writes."""

    def __init__(self, run, outputs):
        self.run, self.outputs = run, outputs


def _common(n, page, kind, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    pages = (2 * n + page - 1) // page + 2
    ang = torch.rand((CS_ROWS, 32), generator=gen, device="cuda") * 6.3
    cs = torch.cat([ang.cos(), ang.sin()], dim=1).bfloat16()
    pos = torch.randint(0, CS_ROWS, (n,), generator=gen, device="cuda")
    slots = torch.randperm(pages * page, generator=gen, device="cuda")[:n]
    dtype = torch.bfloat16 if kind == "bf16" else torch.uint8
    return gen, pages, cs, pos, slots, dtype


def qkv(n, page=16, kind="fp8", seed=0):
    """One draft layer's qkv rows at TP8 (12 q / 2 KV heads)."""
    from amd_kernels.attention import norm_rope_insert_common as E

    gen, pages, cs, pos, slots, dtype = _common(n, page, kind, seed)
    x = (torch.randn((n, (NQ + 2 * NKV) * D), generator=gen, device="cuda") * 2).bfloat16()
    qw, kw = ((torch.randn(D, generator=gen, device="cuda") * 0.5).bfloat16() for _ in range(2))
    cache = torch.zeros((pages, NKV, page, 2 * D), dtype=dtype, device="cuda")
    q = torch.zeros((n, NQ * D), dtype=torch.bfloat16, device="cuda")
    ks, vs = torch.tensor(0.8, device="cuda"), torch.tensor(1.1, device="cuda")
    args = (x, pos, qw, kw, cs, EPS, NQ, NKV, q, cache, slots, ks, vs)
    pkg = importlib.import_module(PKG + ".qkv_norm_rope_insert")

    def run(entry):
        (E.qkv_norm_rope_insert if entry else pkg.qkv_norm_rope_insert)(*args)

    return Call(run, lambda: [q, cache]), args


def multilayer(n, page=16, kind="fp8", seed=0):
    """The context rows of a step, 5 layers x 2 KV heads."""
    from amd_kernels.attention import norm_rope_insert_common as E

    gen, pages, cs, pos, slots, dtype = _common(n, page, kind, seed)
    kv = (torch.randn((n, LAYERS * 2 * NKV * D), generator=gen, device="cuda") * 2).bfloat16()
    kw = (torch.randn((LAYERS, D), generator=gen, device="cuda") * 0.5).bfloat16()
    caches = [torch.zeros((pages, NKV, page, 2 * D), dtype=dtype, device="cuda") for _ in range(LAYERS)]
    scales = [torch.tensor(0.75 + 0.1 * i, device="cuda") for i in range(LAYERS)]
    ks, vs = (scales, scales[::-1]) if kind != "bf16" else (None, None)
    args = (kv, pos, kw, cs, caches, [slots] * LAYERS, ks, vs, EPS)
    pkg = importlib.import_module(PKG + ".kv_norm_rope_insert_multilayer")

    def run(entry):
        (E.kv_norm_rope_insert_multilayer if entry else pkg.kv_norm_rope_insert_multilayer)(*args)

    return Call(run, lambda: list(caches)), args
