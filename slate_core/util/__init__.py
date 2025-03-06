"""A collection of util functions."""

from __future__ import annotations

from slate_core.util._decorators import CachedFunction, cached, timed
from slate_core.util._diagonal import build_diagonal, extract_diagonal
from slate_core.util._index import (
    get_max_idx,
    get_position_in_sorted,
    slice_along_axis,
    slice_ignoring_axes,
)
from slate_core.util._pad import (
    Padding,
    Truncation,
    pad_along_axis,
    pad_ft_points,
    truncate_along_axis,
)

__all__ = [
    "CachedFunction",
    "Padding",
    "Truncation",
    "build_diagonal",
    "cached",
    "extract_diagonal",
    "get_max_idx",
    "get_position_in_sorted",
    "pad_along_axis",
    "pad_ft_points",
    "slice_along_axis",
    "slice_ignoring_axes",
    "timed",
    "truncate_along_axis",
]
