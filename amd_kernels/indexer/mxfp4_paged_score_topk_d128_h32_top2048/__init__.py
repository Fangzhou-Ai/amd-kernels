# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MXFP4 paged index scorer fused with the per-row top-2048 of the DSA decode indexer (gfx950, FlyDSL).

``reference.py`` is the contract; ``host.py`` the entry and its launch configuration; ``lists.py`` the chunk kernel
(scores and per-chunk top lists, the logits never reach memory); ``merge.py`` the row merge.
"""

from .host import cache_views, mxfp4_paged_score_topk
from .reference import TOPK

__all__ = ["TOPK", "cache_views", "mxfp4_paged_score_topk"]
