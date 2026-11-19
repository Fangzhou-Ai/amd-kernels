# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Communicator, row-count dispatch and entry of the eight-rank latent all-reduce + RMSNorm + up projection."""
import ctypes
import functools

import torch
import torch.distributed as dist

from amd_kernels import warm_up
from amd_kernels.common import hip_runtime as ops

from . import tail_kernel as T
from .ipc import alloc_handle, close_handle, exchange, f32_bits, open_handle, run_compiled
from .norm_math import K, N, WORLD

NAME = "moe/allreduce_latent_rmsnorm_up_proj_h3584_n896_ranks8"
MAX_ROWS = T.MAX_ROWS


@functools.cache
def _kernel(mode, mpg, rb, ts=False):
    return T.compile_tail(mode, mpg, rb, ts)


def _release(ptr, peer):
    hip = ops._hip()
    fn = hip.hipIpcCloseMemHandle if peer else hip.hipFree
    fn.argtypes = [ctypes.c_void_p]
    rc = fn(ctypes.c_void_p(ptr))
    assert rc == 0, f"HIP buffer release failed: {rc}"


def _mode(comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """What a call fixes about its kernels besides the row count (``amd_kernels.warm_up``)."""
    return ()


def _sizes(mode):
    """Row counts that reach every (rows per tile group, rows per all-reduce workgroup) variant."""
    return (64, 96, 128, 192, 256)


def _inputs(mode, n, comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    dev = partial.device
    z = lambda *s: torch.zeros(*s, dtype=torch.bfloat16, device=dev)  # noqa: E731
    return (comm, z(n, K), norm_weight, up_weight, z(n, N), None, eps), {}


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def allreduce_latent_rmsnorm_up_proj(comm, partial, norm_weight, up_weight, shared, out=None, eps=1e-6):
    """``out = shared + rmsnorm(allreduce(partial)) @ up_weight^T`` (see reference.py); ``out`` defaults to ``shared``
    (in place). Returns ``out``."""
    return comm._launch(partial, norm_weight, up_weight, shared, shared if out is None else out, eps)


class LatentTailComm:
    """One communicator per stream, created collectively on eight ranks (``device``: this rank's GPU).

    All ranks make the same calls in the same order; close only after the work completes. ``fused=False`` runs the
    all-reduce and the up projection as two launches (the same kernels in their single-role modes).
    """

    def __init__(self, rank, group, device, fused=True):
        if dist.get_world_size(group) != WORLD:
            raise ValueError("eight ranks required")
        device = torch.device(device)
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.current_device() != device.index or dist.get_rank(group) != rank:
            raise ValueError("select this rank device before creating the communicator")
        self.rank, self.group, self.device, self.fused = rank, group, device, fused
        ptr, handle = ops.allocate_shared_buffer_and_handle(T.BUF_BYTES)
        peers = exchange(handle, rank, group)
        self.bufs = [ptr if r == rank else ops.open_mem_handle(peers[r]) for r in range(WORLD)]
        self.lat_buf = torch.zeros(T.LAT_ALLOC // 2, dtype=torch.bfloat16, device=device)
        self.lat = self.lat_buf[:MAX_ROWS * K].view(MAX_ROWS, K)
        lat_peers = exchange(alloc_handle(self.lat_buf.data_ptr()), rank, group)
        self.lat_opened = [open_handle(raw) for r, (raw, _) in enumerate(lat_peers) if r != rank]
        opened = iter(self.lat_opened)
        lats = [self.lat_buf.data_ptr() if r == rank else next(opened) + lat_peers[r][1] for r in range(WORLD)]
        self.buf_table = torch.tensor(self.bufs + lats, dtype=torch.int64, device=device)
        self.counts = torch.zeros(T.CNT_WORDS, dtype=torch.int32, device=device)
        self.rb_override = 0  # experiments: rows per all-reduce workgroup (2, 4 or 8) instead of ``tail_kernel.rb_of``
        self.ts = None  # a [workgroup, wave, 8] int32 tensor of phase stamps (a profiling build), set by the caller
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def _launch(self, x, norm_weight, w, shared, out, eps, modes=None):
        if not self.bufs:
            raise RuntimeError("communicator is closed")
        m = x.shape[0]
        dev = self.device
        if not (0 < m <= MAX_ROWS and tuple(x.shape) == (m, K)):
            raise ValueError(f"expected 1..{MAX_ROWS} rows of width {K}")
        for t in (x, norm_weight, w, shared, out):
            if t.dtype != torch.bfloat16 or t.device != dev or t.stride(-1) != 1:
                raise ValueError("operands must be bf16 on the communicator device with unit inner stride")
        if x.stride(0) % 8 or shared.stride(0) % 4 or out.stride(0) % 4:
            raise ValueError("row strides must keep 16 B (latent) and 8 B (shard) alignment")
        if tuple(norm_weight.shape) != (K,) or tuple(w.shape) != (N, K) or not w.is_contiguous():
            raise ValueError(f"norm weight [{K}] and contiguous up shard [{N}, {K}] required")
        if tuple(shared.shape) != (m, N) or tuple(out.shape) != (m, N):
            raise ValueError(f"shared and out must be [{m}, {N}]")
        stream = torch.cuda.current_stream()
        mpg = T.mpg_of((m + 15) // 16)
        rb = self.rb_override or T.rb_of(m)
        ts = 0 if self.ts is None else self.ts.data_ptr()
        args = (m, (m + rb - 1) // rb, self.rank, self.buf_table.data_ptr(), self.counts.data_ptr(), x.data_ptr(),
                x.stride(0), norm_weight.data_ptr(), w.data_ptr(), shared.data_ptr(), shared.stride(0),
                out.data_ptr(), out.stride(0), f32_bits(eps), 0, ts, stream)
        for mode in modes or (("fused",) if self.fused else ("ar", "up")):
            run_compiled(_kernel(mode, mpg, rb, bool(ts)), *args)
        return out

    def close(self):
        if not self.bufs:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for r, ptr in enumerate(self.bufs):
            _release(ptr, r != self.rank)
        for ptr in self.lat_opened:
            close_handle(ptr)
        self.bufs = []
        self.lat_opened = []
