# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Launch wrapper of the KDA spec-verify decode kernel."""

import functools
import torch

from amd_kernels.warm_up import entry

from .kernel import COL_BETA, LOG2E, NH, D, LP, compile_kda_decode


def _run_compiled(exe, *args):
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)




LPR = 8
STORE_CM = 2
WAVES_SMALL = 8
WAVES_LARGE = 4
LARGE_ROWS = 100


def waves_for(rows):
    return WAVES_LARGE if rows >= LARGE_ROWS else WAVES_SMALL


@functools.cache
def _get_kernel(max_tok, nw_f32, snap, waves):
    return compile_kda_decode(max_tok, nw_f32, waves, LPR, STORE_CM, snap)


def _kda_decode_fb_gate_spec(
    proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens, state_indices,
    num_accepted, norm_weight, norm_eps, out, snapshot_token=None,
):
    n_seq = cu_seqlens.shape[0] - 1
    if n_seq <= 0 or proj.shape[0] == 0:
        return
    max_tok = state_indices.shape[1]
    ps = proj.stride(0)
    assert proj.dtype == torch.bfloat16 and proj.stride(1) == 1 and ps % 2 == 0 and proj.shape[1] >= COL_BETA + NH
    assert f_b_weight.shape == (LP, D) and f_b_weight.dtype == torch.bfloat16 and f_b_weight.is_contiguous()
    assert state.dtype == torch.float32 and state.shape[1:] == (NH, D, D) and state.stride(1) == D * D
    assert out.shape[1:] == (NH, D) and out.dtype == torch.bfloat16 and out.stride(1) == D and out.stride(2) == 1
    assert state_indices.dtype == num_accepted.dtype == cu_seqlens.dtype == torch.int32 and state_indices.stride(1) == 1
    assert norm_weight.is_contiguous() and norm_weight.shape == (D,)
    nw_f32 = norm_weight.dtype == torch.float32
    cs = conv_state
    _run_compiled(
        _get_kernel(max_tok, nw_f32, snapshot_token is not None, waves_for(proj.shape[0])),
        proj.data_ptr(),
        ps,
        proj.shape[0] * ps * 2,
        f_b_weight.data_ptr(),
        A_log.data_ptr(),
        dt_bias.data_ptr(),
        conv_weight.data_ptr(),
        cs.data_ptr(),
        cs.stride(0) * cs.element_size(),
        cs.stride(1),
        cs.stride(2),
        state.data_ptr(),
        state.stride(0) * 4,
        cu_seqlens.data_ptr(),
        state_indices.data_ptr(),
        state_indices.stride(0),
        num_accepted.data_ptr(),
        (snapshot_token if snapshot_token is not None else num_accepted).data_ptr(),
        norm_weight.data_ptr(),
        out.data_ptr(),
        out.stride(0),
        out.shape[0] * out.stride(0) * 2,
        float(lower_bound) * LOG2E,
        float(norm_eps),
        n_seq,
        torch.cuda.current_stream(),
    )


def _mode(proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens, state_indices,
          num_accepted, norm_weight, norm_eps, out, snapshot_token=None):
    """What a call fixes about its kernel (``amd_kernels.warm_up``): the tokens per sequence and the norm weight
    dtype; the token and sequence counts are runtime arguments."""
    return state_indices.shape[1], norm_weight.dtype == torch.float32, snapshot_token is not None


def _inputs(mode, n, proj, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu_seqlens,
            state_indices, num_accepted, norm_weight, norm_eps, out, snapshot_token=None):
    """One sequence of ``n`` tokens (zeros; slot 0 is padding, nothing launches)."""
    dev = proj.device
    x = torch.zeros((n, proj.shape[1]), dtype=proj.dtype, device=dev)
    cu = torch.tensor([0, n], dtype=torch.int32, device=dev)
    idx = torch.zeros((1, state_indices.shape[1]), dtype=torch.int32, device=dev)
    acc = torch.ones(1, dtype=torch.int32, device=dev)
    o = torch.empty((n, NH, D), dtype=torch.bfloat16, device=dev)
    return (x, f_b_weight, A_log, dt_bias, lower_bound, conv_weight, conv_state, state, cu, idx, acc, norm_weight,
            norm_eps, o, acc if mode[2] else None), {}


kda_decode_fb_gate_spec = entry(
    "attention/kda_decode_fb_gate_spec_h12_d128", mode=_mode, sizes=lambda mode: (1, LARGE_ROWS), inputs=_inputs
)(_kda_decode_fb_gate_spec)
