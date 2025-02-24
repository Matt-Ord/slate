from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import EllipsisType


def slice_along_axis(
    slice_at_axis: slice | int | None, axis: int = -1
) -> tuple[EllipsisType | slice | int | None, ...] | tuple[slice | int | None, ...]:
    """Return a slice such that the 1d slice provided by slice_at_axis, slices along the dimension provided."""
    from_end = False
    if axis < 0:  # choosing axis at the end
        from_end = True
        axis = -1 - axis
    # Pad the slice with slice(None)
    slice_padding = axis * (slice(None),)
    if from_end:
        return (Ellipsis, slice_at_axis, *slice_padding)

    return (*slice_padding, slice_at_axis)


def slice_ignoring_axes[A: slice | int | None](
    old_slice: Sequence[A], axes: tuple[int, ...]
) -> tuple[slice | A, ...]:
    """Given a slice, insert slice(None) everywhere given in axes."""
    new_slice = list[slice | A](old_slice)
    for axis in sorted(int(a) for a in axes):
        new_slice.insert(axis, slice(None))
    return tuple(new_slice)


def get_position_in_sorted(axes: tuple[int, ...]) -> tuple[int, ...]:
    """
    Given a list of axes get the index in the sorted list.

    ie 2,4,1,3 -> 1,3,0,2
    """
    return tuple(np.argsort(np.argsort(axes)))


def get_max_idx(
    data: np.ndarray[tuple[int, ...], np.dtype[np.number[Any]]],
    *,
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    """Get the index of the max of the data in the given axes."""
    max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    return tuple(x.item() for (i, x) in enumerate(max_idx) if i not in axes)
