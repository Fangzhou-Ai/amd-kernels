# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""The device check and the reason texts every op of the family shares."""
import functools


@functools.cache
def available() -> bool:
    """gfx950 with FlyDSL importable."""
    import torch

    if not torch.cuda.is_available() or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName:
        return False
    try:
        import flydsl  # noqa: F401
    except ImportError:
        return False
    return True


def dims(names, key) -> str:
    if len(names) != len(key):
        return f"shape {tuple(key)}"
    return ", ".join(f"{n}={v}" for n, v in zip(names, key))


def no_kernel(op, names, key, table) -> str:
    return f"{op}: no kernel for {dims(names, key)} (covered: {', '.join(str(k) for k in sorted(table))})"


def no_kernel_for_type(op, cls, table) -> str:
    return f"{op}: no kernel for a {cls.__qualname__} (covered: {', '.join(str(k) for k in sorted(table))})"


def built_for(op, names, key, call, what="communicator") -> str:
    return f"{op}: {what} built for {dims(names, key)}, call is {dims(names[:len(call)], call)}"


def key_of_type(table, cls):
    """The key whose module defines ``cls``, or None."""
    return next((k for k, name in table.items() if cls.__module__ == name), None)



def swap_stream_keys(workspaces: dict, device_index: int, a: int, b: int) -> None:
    """Exchange the entries of streams ``a`` and ``b`` on a device in a dict keyed (device index, stream, ...)."""
    moved = {}
    for key in [k for k in workspaces if k[0] == device_index and k[1] in (a, b)]:
        moved[(key[0], b if key[1] == a else a, *key[2:])] = workspaces.pop(key)
    workspaces.update(moved)


def loaded_hosts(table) -> list:
    """The shape-package hosts of a ``_SHAPE_HOSTS`` table imported so far (a package nobody called has no
    workspace)."""
    import sys

    return [sys.modules[name] for name in table.values() if name in sys.modules]


def swap_stream_workspaces(device_index: int, stream_a: int, stream_b: int) -> None:
    """Exchange the router's and the experts' per-stream workspaces of two streams on a device (dict keys only:
    nothing is allocated, copied or freed). A graph captured on ``stream_b`` while it holds ``stream_a``'s workspaces
    bakes their addresses, so one set serves a stream's eager calls and the graphs it replays; the calls on one set
    must stay in one stream order (the kernels keep their epochs and tickets on the device)."""
    from . import experts, router

    for host in loaded_hosts(router._SHAPE_HOSTS) + loaded_hosts(experts._SHAPE_HOSTS):
        for name in ("_WORK", "_AUX_WORK"):
            if isinstance(getattr(host, name, None), dict):
                swap_stream_keys(getattr(host, name), device_index, stream_a, stream_b)
