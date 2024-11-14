"""A collection of util functions."""

from __future__ import annotations

from ._decorators import CachedFunction, cached, timed
from ._index import (
    get_data_in_axes,
    get_max_idx,
    get_position_in_sorted,
    slice_along_axis,
    slice_ignoring_axes,
)
from ._pad import (
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
    "cached",
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
