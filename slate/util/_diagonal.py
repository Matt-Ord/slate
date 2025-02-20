from __future__ import annotations

from typing import Any, cast

import numpy as np

from slate.util._pad import Padding, Truncation, pad_along_axis, truncate_along_axis


def _spec_from_indices[DT: int | np.signedinteger](
    indices: tuple[DT, ...],
) -> str:
    return "".join(chr(cast("int", 97 + i)) for i in indices)


def extract_diagonal[DT: np.dtype[np.generic]](
    array: np.ndarray[Any, DT], axes: tuple[int, ...], out_axis: int = -1
) -> np.ndarray[Any, DT]:
    input_indices = np.arange(array.ndim)
    input_indices[list(axes)] = input_indices[axes[0]]

    output_indices = np.delete(np.arange(array.ndim), list(axes))
    output_indices = np.insert(output_indices, out_axis, input_indices[axes[0]])

    square_slice = np.array([slice(None)] * array.ndim)
    n_out = np.min(np.array(array.shape)[list(axes)])
    square_slice[list(axes)] = slice(n_out)

    subscripts = f"{_spec_from_indices(tuple(input_indices))}->{_spec_from_indices(tuple(output_indices))}"
    return np.einsum(subscripts, array[tuple(square_slice)])  # type: ignore unknown


def build_diagonal[DT: np.dtype[np.generic]](
    array: np.ndarray[Any, DT],
    axis: int = -1,
    out_axes: tuple[int, ...] = (-1, -2),
    out_shape: tuple[int, ...] | None = None,
) -> np.ndarray[Any, DT]:
    out_shape = out_shape or (array.shape[axis],) * len(out_axes)
    assert len(out_shape) == len(out_axes)
    n_dim_out = array.ndim - 1 + len(out_axes)

    eye_indices = np.mod(out_axes, n_dim_out)

    output_indices = np.arange(n_dim_out)

    input_indices = np.delete(np.arange(n_dim_out), list(out_axes))
    input_indices = np.insert(input_indices, axis, eye_indices[0])

    eye = np.zeros(out_shape, dtype=array.dtype)
    np.fill_diagonal(eye, 1)

    if array.shape[axis] == out_shape[0]:
        padded = array
    elif array.shape[axis] < out_shape[0]:
        padded = pad_along_axis(array, Padding(out_shape[0], 0, 0), axis)
    else:
        padded = truncate_along_axis(array, Truncation(out_shape[0], 0, 0), axis)

    subscripts = f"{_spec_from_indices(tuple(input_indices))},{_spec_from_indices(tuple(eye_indices))}->{_spec_from_indices(tuple(output_indices))}"
    return np.einsum(subscripts, padded, eye)  # type: ignore unknown
