# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Host launcher of the KDA decode kernel K1."""

import torch

from .kernel import ABI, KdaPreBuild, build


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


def ptr(t):
    return 0 if t is None else t.data_ptr()


def kda_pre(
    key: KdaPreBuild,
    *,
    prefix,
    delta,
    blocks,
    ares_nw,
    ares_qk,
    in_nw,
    w_in,
    w_fb,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    on_w,
    rstate,
    st_idx,
    num_acc,
    core_out,
    scratch,
    layer,
    epoch,
    health,
    alarm,
    tl=None,
):
    """Host launcher. ``conv_state`` [slots, dim, state_len] or a transposed view
    of [slots, state_len, dim] (strides read); ``rstate`` [slots, 12, 128, 128]
    fp32; ``st_idx`` [R, >= L] int32; ``scratch`` zeroed before the step's first
    layer (tags are a layer's); ``epoch`` a device int32 bumped once a step and
    ``layer`` this launch's slot below ``TAG_SLOTS``."""
    f = build(key)
    _run_compiled(
        f,
        prefix.data_ptr(),
        ptr(delta),
        blocks.data_ptr(),
        blocks.stride(0),
        blocks.stride(1),
        ares_nw.data_ptr(),
        ares_qk.data_ptr(),
        in_nw.data_ptr(),
        w_in.data_ptr(),
        w_fb.data_ptr(),
        conv_w.data_ptr(),
        conv_state.data_ptr(),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        on_w.data_ptr(),
        rstate.data_ptr(),
        rstate.stride(0),
        st_idx.data_ptr(),
        st_idx.stride(0),
        num_acc.data_ptr(),
        core_out.data_ptr(),
        scratch.data_ptr(),
        layer,
        epoch.data_ptr(),
        health.data_ptr(),
        alarm,
        ptr(tl),
        torch.cuda.current_stream(),
    )
