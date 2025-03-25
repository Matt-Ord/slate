from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from slate_core.util._index import slice_along_axis

if TYPE_CHECKING:
    from collections.abc import Iterable


def pad_ft_points[DT: np.dtype[np.generic]](
    array: np.ndarray[Any, DT],
    s: Iterable[int],
    axes: Iterable[int],
) -> np.ndarray[Any, DT]:
    """
    Pad the points in the fourier transform with zeros.

    Pad the points in the fourier transform with zeros, keeping the frequencies of
    each point the same in the initial and final grid.

    """
    shape_arr = np.array(array.shape, dtype=np.int_)
    axes_arr = np.asarray(axes)

    padded_shape = shape_arr.copy()
    padded_shape[axes_arr] = tuple(s)
    padded: np.ndarray[Any, DT] = np.zeros(  # type: ignore can't infer dtype
        shape=padded_shape, dtype=array.dtype
    )

    slice_start = np.array([slice(None) for _ in array.shape], dtype=slice)
    slice_start[axes_arr] = np.array(
        [
            slice(1 + min((n - 1) // 2, (s - 1) // 2))
            for (n, s) in zip(shape_arr[axes_arr], s, strict=True)
        ],
        dtype=slice,
    )
    slice_end = np.array([slice(None) for _ in array.shape], dtype=slice)
    slice_end[axes_arr] = np.array(
        [
            slice(start, None)
            if (start := max((-n + 1) // 2, (-s + 1) // 2)) < 0
            # else no negative frequencies
            else slice(0, 0)
            for (n, s) in zip(shape_arr[axes_arr], s, strict=True)
        ],
        dtype=slice,
    )
    # For each combination of start/end region of the array
    # add in the corresponding values to the padded array
    for slices in itertools.product(*np.array([slice_start, slice_end]).T):
        padded[slices] = array[slices]

    return padded


@dataclass(frozen=True)
class Truncation:
    """Truncation of the basis."""

    n: int
    step: int
    offset: int


def assert_unique_indices(size: int, n: int, step: int) -> None:
    assert n <= size // math.gcd(step, size), "Indices must be unique"


def _get_truncated_indices(
    size: int,
    truncation: Truncation,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    n = truncation.n
    step = truncation.step
    offset = truncation.offset

    assert_unique_indices(size, n, step)
    return (offset + step * np.arange(n)) % size


def _truncate_along_axis_continuous[DT: np.dtype[np.generic]](
    vectors: np.ndarray[Any, DT],
    truncation: Truncation,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], DT]:
    step = truncation.step
    offset = truncation.offset
    n = truncation.n
    # We could alternatively roll after we take the slice
    # but this more complex (ie not worth performance)
    rolled = np.roll(vectors, -offset, axis=axis)
    return rolled[slice_along_axis(slice(0, n * step, step), axis)]  # type: ignore index type wrong


def _is_contiguous_truncate(size: int, truncation: Truncation) -> bool:
    return size >= truncation.step * truncation.n


def truncate_along_axis[DT: np.dtype[np.generic]](
    vectors: np.ndarray[Any, DT],
    truncation: Truncation,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], DT]:
    """Truncate Data along an axis."""
    if _is_contiguous_truncate(vectors.shape[axis], truncation):
        return _truncate_along_axis_continuous(vectors, truncation, axis)

    data_length = vectors.shape[axis]
    indices = _get_truncated_indices(data_length, truncation)
    return np.take(vectors, indices.astype(int), axis=axis)  # type: ignore dtype is DT


@dataclass(frozen=True)
class Padding:
    """Padding of the basis."""

    n: int
    step: int
    offset: int


def _pad_along_axis_continuous[DT: np.dtype[np.generic]](
    vectors: np.ndarray[Any, DT],
    padding: Padding,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], DT]:
    final_shape = np.array(vectors.shape)
    final_shape[axis] = padding.n
    out = np.zeros(final_shape, dtype=vectors.dtype)

    # We could alternatively roll and then slice as this si more performant
    # but adds complexity
    step = padding.step
    s = slice_along_axis(slice(0, step * vectors.shape[axis], step), axis)
    out[s] = vectors

    offset = padding.offset
    return np.roll(out, offset, axis=axis)  # type: ignore[no-any-return]


def _is_contiguous_pad(size: int, padding: Padding) -> bool:
    return padding.n >= padding.step * size


def pad_along_axis[DT: np.dtype[np.generic]](
    vectors: np.ndarray[Any, DT],
    padding: Padding,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], DT]:
    """Pad Data along an axis."""
    if _is_contiguous_pad(vectors.shape[axis], padding):
        return _pad_along_axis_continuous(vectors, padding, axis)
    step = padding.step
    offset = padding.offset
    reverse_truncation = Truncation(vectors.shape[axis], step, offset)
    # The indices we took from the original array
    indices = _get_truncated_indices(padding.n, reverse_truncation)

    final_shape = np.array(vectors.shape)
    final_shape[axis] = padding.n
    out = np.zeros(final_shape, dtype=vectors.dtype)
    out[indices] = vectors
    return out
