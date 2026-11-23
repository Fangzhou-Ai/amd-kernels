# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host side of the drafter's dense MLP: tier choice, workspace, launches, warm-up."""
import torch
from amd_kernels import warm_up

from .kernel import CNT_STRIDE, compile_gemm
from .pack import GP, pack_down, pack_gate_up, pack_rows, unpack_rows
from .reference import HIDDEN, INTER

NAME = "decoder/bf16_dense_mlp_h7168_i1792"
KS_IN = HIDDEN // 32
KS_ACT = INTER // 32
GROUPS_UP = 2 * INTER // 16
GROUPS_DOWN = HIDDEN // 16
MAX_ROWS = 256

# (rows up to, row tiles of 16, gate_up and down (mt, g, split, depth, wr, wk)): mt row tiles per workgroup (the grid
# covers tiles / mt row blocks), weight groups, K split, loads in flight, wave grid (rows x K)
TIERS = (
    (16, 1, (1, 4, 4, 4, 1, 4), (1, 2, 1, 4, 1, 4)),
    (32, 2, (2, 4, 4, 4, 1, 2), (2, 2, 1, 4, 1, 4)),
    (48, 3, (3, 4, 4, 4, 1, 4), (3, 2, 1, 4, 1, 2)),
    (64, 4, (4, 4, 4, 4, 1, 4), (4, 2, 1, 4, 1, 2)),
    (80, 5, (5, 4, 4, 4, 1, 4), (5, 2, 1, 4, 1, 2)),
    (96, 6, (6, 4, 4, 4, 1, 4), (6, 2, 1, 4, 1, 2)),
    (128, 8, (8, 4, 4, 4, 1, 4), (8, 2, 1, 4, 1, 2)),
    (144, 9, (9, 4, 4, 4, 1, 4), (9, 2, 1, 4, 1, 2)),
    # two row blocks: 43.1 us at 240 rows against 50.5 for the best one-block pair (gate_up + down, graph loop)
    (160, 10, (5, 4, 2, 3, 1, 4), (5, 4, 1, 3, 1, 4)),
    (192, 12, (6, 4, 2, 3, 1, 4), (6, 4, 1, 3, 1, 4)),
    (224, 14, (7, 4, 2, 3, 1, 4), (7, 4, 1, 3, 1, 4)),
    (256, 16, (8, 4, 2, 3, 1, 4), (8, 4, 1, 3, 1, 4)),
)


def tier_for(rows: int) -> tuple:
    for tier in TIERS:
        if rows <= tier[0]:
            return tier
    raise ValueError(f"at most {MAX_ROWS} rows")


def packed_tiles(rows: int) -> int:
    """Row tiles of 16 in the packed activations the kernels read and write for ``rows`` rows."""
    return tier_for(rows)[1]


def packed_rows_buffer(rows: int, device) -> torch.Tensor:
    return torch.empty(packed_tiles(rows), KS_IN, 64, 8, dtype=torch.bfloat16, device=device)


def _blocks(tiles, groups, cfg):
    """(row blocks, tiles of the grid, workgroups)."""
    mt, g, split = cfg[:3]
    mblocks = tiles // mt
    n = mblocks * (groups // g)
    return mblocks, n, (n + 7) // 8 * 8 * split


def _items(cfg, gate_up):
    mt, g, _, _, wr, _ = cfg
    return wr * (mt // wr) * (g // 2 if gate_up else g) * 64


class MlpWorkspace:
    """Partial-sum slabs, ticket counters and the packed intermediate of one stream (allocated once; graph safe)."""

    def __init__(self, device):
        slab = cnt_up = cnt_down = 0
        for _, tiles, up, down in TIERS:
            _, n_up, grid_up = _blocks(tiles, GROUPS_UP, up)
            _, n_down, grid_down = _blocks(tiles, GROUPS_DOWN, down)
            slab = max(slab, grid_up * _items(up, True) * 8, grid_down * _items(down, False) * 4)
            cnt_up, cnt_down = max(cnt_up, n_up), max(cnt_down, n_down)
        self.slab = torch.zeros(slab + 1024, dtype=torch.int32, device=device)
        self.cnt_up = torch.zeros(cnt_up * CNT_STRIDE, dtype=torch.int32, device=device)
        self.cnt_down = torch.zeros(cnt_down * CNT_STRIDE, dtype=torch.int32, device=device)
        self.act = torch.zeros(max(t for _, t, _, _ in TIERS), KS_ACT, 64, 8, dtype=torch.bfloat16, device=device)


def _run(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


def swiglu_gate_up(ws: MlpWorkspace, x_packed: torch.Tensor, gate_up: torch.Tensor, rows: int) -> torch.Tensor:
    """bf16(silu(gate) * up) of the packed activations, in the packed layout of ``down_proj``'s input."""
    _, tiles, cfg, _ = tier_for(rows)
    mblocks, _, grid = _blocks(tiles, GROUPS_UP, cfg)
    exe = compile_gemm(HIDDEN, GROUPS_UP, *cfg, 2)
    _run(exe, x_packed.data_ptr(), gate_up.data_ptr(), ws.act.data_ptr(), ws.slab.data_ptr(), ws.cnt_up.data_ptr(),
         rows, mblocks, grid, torch.cuda.current_stream())
    return ws.act


def down_proj(ws: MlpWorkspace, act: torch.Tensor, down: torch.Tensor, rows: int, out: torch.Tensor) -> torch.Tensor:
    _, tiles, _, cfg = tier_for(rows)
    mblocks, _, grid = _blocks(tiles, GROUPS_DOWN, cfg)
    exe = compile_gemm(INTER, GROUPS_DOWN, *cfg, 1)
    _run(exe, act.data_ptr(), down.data_ptr(), out.data_ptr(), ws.slab.data_ptr(), ws.cnt_down.data_ptr(), rows,
         mblocks, grid, torch.cuda.current_stream())
    return out


def _mode(ws, x_packed, gate_up, down, rows, out=None):
    return ()


def _sizes(mode):
    return [tier[0] for tier in TIERS]


def _inputs(mode, n, ws, x_packed, gate_up, down, rows, out=None):
    return (ws, packed_rows_buffer(n, gate_up.device), gate_up, down, n), {}


@warm_up.entry(NAME, mode=_mode, sizes=_sizes, inputs=_inputs)
def dense_mlp(ws: MlpWorkspace, x_packed: torch.Tensor, gate_up: torch.Tensor, down: torch.Tensor, rows: int,
              out: torch.Tensor | None = None) -> torch.Tensor:
    """This rank's partial MLP output [rows, 7168] bf16 = down(silu(gate) * up) of the packed activations
    ``x_packed`` ([packed_tiles(rows), 224, 64, 8], ``pack_rows``) and the packed weights (``pack_gate_up``,
    ``pack_down``); the caller all-reduces it. 1 <= rows <= 256. Graph safe; two launches."""
    assert 1 <= rows <= MAX_ROWS
    if out is None:
        out = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=gate_up.device)
    assert out.shape == (rows, HIDDEN) and out.is_contiguous()
    act = swiglu_gate_up(ws, x_packed, gate_up, rows)
    return down_proj(ws, act, down, rows, out)


__all__ = ["MlpWorkspace", "dense_mlp", "pack_down", "pack_gate_up", "pack_rows", "unpack_rows", "swiglu_gate_up",
           "down_proj", "packed_rows_buffer", "packed_tiles", "GP"]
