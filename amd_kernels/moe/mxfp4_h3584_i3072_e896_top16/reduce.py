# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Sum of the 16 slot partials of every token in slot order (fp32), rounded once to BF16."""
import functools
import flydsl.compiler as flyc
from amd_kernels.warm_up import launcher
import flydsl.expr as fx
from amd_kernels.common import buffer_ops
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Vector as Vec

from .common import H, TOPK
from .sync import poll_first, wait_count

HALF = H // 2  # columns per tile: 224 threads x 8


@functools.cache
def compile_reduce_tile(*, bm=16, tiles_per_block=14):
    """Tile (token, half) of the reduction (256 threads, 224 used). With ``done`` != 0 it first waits until the counter
    shows every GEMM2 tile of the call stored (``tiles_per_block`` per active m-block); the last of the 2 m tiles
    resets it."""

    @flyc.jit
    def tile(rb, SL, OUT, NV, REV, m, rows, done):
        tx = fx.thread_idx.x
        t, half = rb // 2, rb % 2
        if done != fx.Int64(0):
            need = fx.Int32(NV[0]) // bm * tiles_per_block
            wait_count(done, need, fx.Int32(2) * m, poll_first(done))
        if tx < HALF // 8:
            sr = buffer_ops.create_buffer_resource(SL, max_size=False, num_records_bytes=rows * H * 2)
            orr = buffer_ops.create_buffer_resource(OUT, max_size=False, num_records_bytes=m * H * 2)
            rv = [fx.Int32(REV[t * TOPK + s]) for s in range_constexpr(TOPK)]
            vs = [Vec(buffer_ops.buffer_load(sr, (rv[s] * H + half * HALF) // 2 + tx * 4, vec_width=4,
                                             dtype=fx.Int32)) for s in range_constexpr(TOPK)]
            acc = [fx.Float32(0.0)] * 8
            for s in range_constexpr(TOPK):
                for i in range_constexpr(4):
                    w = fx.Int32(vs[s][i])
                    acc[2 * i] = acc[2 * i] + (w << 16).bitcast(fx.Float32)
                    acc[2 * i + 1] = acc[2 * i + 1] + (w & -65536).bitcast(fx.Float32)
            pk = [Vec.from_elements([acc[2 * i], acc[2 * i + 1]], fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)[0]
                  for i in range_constexpr(4)]
            buffer_ops.buffer_store(Vec.from_elements(pk, fx.Int32), orr, (t * H + half * HALF) // 2 + tx * 4)

    return tile


@functools.cache
@launcher
def compile_reduce(*, name_suffix=""):
    tile = compile_reduce_tile()
    name = f"moe_reduce_slots_h{H}_top{TOPK}{name_suffix}"

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def kernel(SL: fx.Tensor, OUT: fx.Tensor, NV: fx.Tensor, REV: fx.Tensor, m: fx.Int32, rows: fx.Int32):
        tile(fx.Int32(fx.block_idx.x), SL, OUT, NV, REV, m, rows, fx.Int64(0))

    @flyc.jit
    def launch(SL: fx.Tensor, OUT: fx.Tensor, NV: fx.Tensor, REV: fx.Tensor, m: fx.Int32, rows: fx.Int32,
               stream: fx.Stream):
        kernel(SL, OUT, NV, REV, m, rows).launch(grid=(m * 2, 1, 1), block=(256, 1, 1), stream=stream)

    launch.kernel_name = name
    return launch
