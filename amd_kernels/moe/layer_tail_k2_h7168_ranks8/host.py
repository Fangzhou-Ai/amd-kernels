# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the layer tail K2: the launch, and the per-rank state (scratch, epoch, health words, peer memory)."""

import torch

from amd_kernels.common.mono.abi import PEER_HEAD
from amd_kernels.common.mono.peer import PeerBuffer
from amd_kernels.common.mono.plan import HEALTH_WORDS

from . import kernel as K
from .kernel import ABI, K2Build, build

TP = K.TP_


def _run_compiled(launcher, *args) -> None:
    import flydsl.compiler as flyc
    from flydsl.utils import env

    if env.compile.compile_only:
        flyc.compile(launcher, *args)
        return
    compiled = getattr(launcher, "_cf", None)
    if compiled is not None:
        compiled(*args)
        return
    launcher._cf = flyc.compile(launcher, *args)


def _ptr(t):
    return 0 if t is None else t.data_ptr()


class LayerTailRunner:
    """Collective over the TP8 group: every rank builds it at the same point.

    ``scratch`` is shared by every launch of a step (their mailbox tags differ); ``min_scratch`` is the bytes a
    caller's other launches need, so the tagged-row region lies past them.
    """

    def __init__(self, device, group, rank, min_scratch: int = 0):
        self.rank = rank
        nbytes = max(K.tag_end(), min_scratch, K.scratch_bytes(K2Build(tokens=K.k2b.MAX_S, nblocks=8)))
        self.scratch = torch.zeros(nbytes, dtype=torch.uint8, device=device)
        self.epoch = torch.zeros(1, dtype=torch.int32, device=device)
        self.health = torch.zeros(HEALTH_WORDS, dtype=torch.int32, device=device)
        self.peers = PeerBuffer(PEER_HEAD + K.extra_peer_end(), group, rank, TP, device)
        self.peers.bytes.zero_()
        lo = PEER_HEAD + K.sent_offset()
        self.peers.bytes[lo : lo + K.sent_bytes()].fill_(0xFF)
        torch.accelerator.synchronize()
        torch.distributed.barrier(group=group)
        table = self.peers.addresses
        self.peer_table = (table.repeat(TP) if table.numel() == 1 else table) + PEER_HEAD
        self.alarm = int(self.peers.addresses[0].item()) + 4 * rank
        self.flags = self.peers.bytes[: 4 * TP].view(torch.int32)

    def step_begin(self) -> None:
        """Once a forward, before its first launch (a CUDA graph captures it)."""
        self.epoch.add_(1)

    def launch_args(self) -> dict:
        return {"scratch": self.scratch, "epoch": self.epoch, "health": self.health, "alarm": self.alarm}

    def peer_args(self) -> dict:
        return {"peers": self.peer_table, "rank": self.rank}

    def close(self) -> None:
        self.peers.close()


def k2_launch(
    key: K2Build,
    *,
    core,
    w_o,
    prefix,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_gate,
    bias,
    w_ld,
    w_sgu,
    w_sd,
    w13,
    w13s,
    w2,
    w2s,
    ln_w,
    w_up,
    out,
    scratch,
    peers,
    rank,
    epoch,
    layer,
    health,
    alarm,
    tl=None,
):
    """``core`` [S, 1536] (o_proj's input), ``prefix`` updated in place (written when ``key.reset``), ``out`` the
    MoE output [S, 7168]."""
    f = build(key)
    _run_compiled(
        f,
        core.data_ptr(),
        w_o.data_ptr(),
        prefix.data_ptr(),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        w_gate.data_ptr(),
        bias.data_ptr(),
        w_ld.data_ptr(),
        w_sgu.data_ptr(),
        w_sd.data_ptr(),
        w13.data_ptr(),
        w13s.data_ptr(),
        w2.data_ptr(),
        w2s.data_ptr(),
        ln_w.data_ptr(),
        w_up.data_ptr(),
        out.data_ptr(),
        scratch.data_ptr(),
        peers.data_ptr(),
        rank,
        epoch.data_ptr(),
        layer,
        health.data_ptr(),
        alarm,
        _ptr(tl),
        torch.cuda.current_stream(),
    )
