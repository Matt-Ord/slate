from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

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


def slice_ignoring_axes(
    old_slice: Sequence[slice | int | None], axes: tuple[int, ...]
) -> tuple[slice | int | None, ...]:
    """
    Given a slice, insert slice(None) everywhere given in axes.

    Parameters
    ----------
    slice : list[slice  |  _IntLike_co  |  None]
        _description_
    axes : tuple[_IntLike_co]
        _description_

    Returns
    -------
    list[slice | _IntLike_co | None]
        _description_
    """
    old_slice = list(old_slice)
    for axis in sorted(int(a) for a in axes):
        old_slice.insert(axis, slice(None))
    return tuple(old_slice)


def get_position_in_sorted(axes: tuple[int, ...]) -> tuple[int, ...]:
    """
    Given a list of axes get the index in the sorted list.

    ie 2,4,1,3 -> 1,3,0,2

    Parameters
    ----------
    axes : _AX0Inv

    Returns
    -------
    _AX0Inv
    """
    return tuple(np.argsort(np.argsort(axes)))


@overload
def get_data_in_axes[DT: np.generic](
    data: np.ndarray[Any, np.dtype[DT]],
    axes: tuple[int],
    idx: tuple[int, ...],
) -> np.ndarray[tuple[int], np.dtype[DT]]: ...


@overload
def get_data_in_axes[DT: np.generic](
    data: np.ndarray[Any, np.dtype[DT]],
    axes: tuple[int, int],
    idx: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[DT]]: ...


@overload
def get_data_in_axes[DT: np.generic](
    data: np.ndarray[Any, np.dtype[DT]],
    axes: tuple[int, ...],
    idx: tuple[int, ...],
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]: ...


def get_data_in_axes[DT: np.generic](
    data: np.ndarray[Any, np.dtype[DT]],
    axes: tuple[int, ...] | tuple[int],
    idx: tuple[int, ...],
) -> np.ndarray[tuple[int, ...], np.dtype[DT]] | np.ndarray[tuple[int], np.dtype[DT]]:
    """
    Given a slice, insert slice(None) everywhere given in axes.

    Parameters
    ----------
    slice : list[slice  |  _IntLike_co  |  None]
        slice excluding axes
    axes : tuple[_IntLike_co]
        axes to insert slice(None)

    Returns
    -------
    list[slice | _IntLike_co | None]
    """
    return np.transpose(
        data[slice_ignoring_axes(idx, axes)], get_position_in_sorted(axes)
    )  # type: ignore[no-any-return]


def get_max_idx(
    data: np.ndarray[tuple[int, ...], np.dtype[np.number[Any]]],
    axes: tuple[int, ...],
) -> tuple[int, ...]:
    """Get the index of the max of the data in the given axes."""
    max_idx = np.unravel_index(np.argmax(np.abs(data)), data.shape)
    return tuple(x.item() for (i, x) in enumerate(max_idx) if i not in axes)
