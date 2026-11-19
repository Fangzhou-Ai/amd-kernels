# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Contract against reference.py on 8 GPUs: eager and graph replay, changed inputs, masks, padding rows, strides."""
import os

import torch
import torch.distributed as dist

from .harness import run_on_8_gpus

W = 8


def make_inputs(t, step, dev, empty_ranks=(), pad_heads=96):
    """x, lse of all 8 ranks, w_uv, and per-rank (seq_lens, query_start_loc) with ``t`` rows of 1..3 per request."""
    g = torch.Generator(device=dev).manual_seed(1234 + t * 7 + step * 100003)
    xs = [(torch.randn((t, pad_heads, 512), generator=g, device=dev) * 0.5).bfloat16() for _ in range(W)]
    lses = [torch.randn((t, pad_heads), generator=g, device=dev) * 2 + 3 for _ in range(W)]
    w_uv = (torch.randn((96, 512, 128), generator=g, device=dev) * 0.05).bfloat16()
    nreq = max(1, (t * 2) // 3)
    bounds = torch.linspace(0, max(t - 2, 1), nreq + 1).round().int().tolist()  # the last rows are graph padding
    qsl = torch.tensor(bounds, dtype=torch.int32, device=dev)
    sls = []
    for r in range(W):
        sl = torch.randint(1, 5000, (nreq,), generator=g, device=dev).int()
        if r in empty_ranks:
            sl[:] = 0
        elif nreq > 1:
            sl[(r + step) % nreq] = 0
        sls.append(sl)
    return xs, lses, w_uv, sls, [qsl] * W


def _worker(rank, errors, tokens, base_e=True):
    from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512 import DcpVupCombine, dcp8_lse_combine_vup, prepare_w_uv
    from amd_kernels.attention.dcp8_lse_combine_vup_mla_h12_kv512.reference import combine_reference

    dev = torch.device("cuda", rank)
    import os
    comm = DcpVupCombine(rank, dist.group.WORLD, dev, is_lse_base_on_e=base_e)
    if os.environ.get('DCP_VARIANT'):
        comm.variant = eval(os.environ['DCP_VARIANT'])

    def check(tag, got, xs, lses, w_uv, sls, qsls, gate=None):
        want = combine_reference(xs, lses, w_uv, rank, sls, qsls, base_e, gate).float()
        g = got.float()
        err = (g - want).abs().max().item()
        scale = want.abs().max().item() + 1e-6
        cos = torch.nn.functional.cosine_similarity(g.flatten(), want.flatten(), dim=0).item()
        if not (g == g).all() or err > scale * 2**-7 or cos < 0.9999:
            errors.append(f"rank {rank} {tag}: max err {err:.4g} (scale {scale:.4g}) cos {cos:.6f}")

    for t in tokens:
        for step, pad in ((0, 96), (1, 128)):
            xs, lses, w_uv, sls, qsls = make_inputs(t, step, dev, empty_ranks=(3,) if step == 1 else (), pad_heads=pad)
            wt = prepare_w_uv(w_uv)
            out = torch.full((t, 12, 128), float("nan"), dtype=torch.bfloat16, device=dev)
            dcp8_lse_combine_vup(comm, xs[rank], lses[rank], wt, out, sls[rank], qsls[rank])
            torch.cuda.synchronize()
            check(f"t{t} pad{pad} eager", out, xs, lses, w_uv, sls, qsls)
            gate = torch.randn((t, 1536), device=dev).bfloat16() * 3
            dcp8_lse_combine_vup(comm, xs[rank], lses[rank], wt, out, sls[rank], qsls[rank], gate=gate)
            torch.cuda.synchronize()
            check(f"t{t} pad{pad} gate", out, xs, lses, w_uv, sls, qsls, gate)
        out.fill_(float("nan"))
        cx, cl, cs = xs[rank], lses[rank], sls[rank]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dcp8_lse_combine_vup(comm, cx, cl, wt, out, cs, qsls[rank])
        for step in range(2, 2 + int(os.environ.get('DCP_REPS', '4'))):
            xs2, lses2, w2, sls2, qsls2 = make_inputs(t, step, dev, pad_heads=128)
            cx.copy_(xs2[rank]); cl.copy_(lses2[rank]); cs.copy_(sls2[rank])
            out.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(f"t{t} replay{step}", out, xs2, lses2, w_uv, sls2, qsls2)
            first = out.clone()
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(first, out):
                errors.append(f"rank {rank} t{t} replay{step}: two runs differ")
        del graph


def test_combine_contract():
    run_on_8_gpus(_worker, [int(t) for t in os.environ.get('DCP_TOKENS', '1,12,16,17,40,65,128,129,192').split(',')])
