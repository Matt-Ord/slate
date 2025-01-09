"""A collection of util functions."""

from __future__ import annotations

from slate.util._decorators import CachedFunction, cached, timed
from slate.util._diagonal import build_diagonal, extract_diagonal
from slate.util._index import (
    get_data_in_axes,
    get_max_idx,
    get_position_in_sorted,
    slice_along_axis,
    slice_ignoring_axes,
)
from slate.util._pad import (
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
    "get_data_in_axes",
    "get_max_idx",
    "get_position_in_sorted",
    "pad_along_axis",
    "pad_ft_points",
    "slice_along_axis",
    "slice_ignoring_axes",
    "timed",
    "truncate_along_axis",
]
