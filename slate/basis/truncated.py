from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Callable, Iterable, Never, Self, override

import numpy as np

from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis import Basis


def pad_ft_points[DT: np.generic](
    array: np.ndarray[Any, np.dtype[DT]],
    s: Iterable[int],
    axes: Iterable[int],
) -> np.ndarray[Any, np.dtype[DT]]:
    """
    Pad the points in the fourier transform with zeros.

    Pad the points in the fourier transform with zeros, keeping the frequencies of
    each point the same in the initial and final grid.

    Parameters
    ----------
    array : NDArray
        The array to pad
    s : Sequence[int]
        The length along each axis to pad or truncate to
    axes : NDArray
        The list of axis to pad

    Returns
    -------
    NDArray
        The padded array
    """
    shape_arr = np.array(array.shape, dtype=np.int_)
    axes_arr = np.asarray(axes)

    padded_shape = shape_arr.copy()
    padded_shape[axes_arr] = s
    padded: np.ndarray[Any, np.dtype[DT]] = np.zeros(  # type: ignore can't infer dtype
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
    for slices in itertools.product(*np.array([slice_start, slice_end]).T.tolist()):
        padded[tuple(slices)] = array[tuple(slices)]

    return padded


class TruncatedBasis[M: BasisMetadata, DT: np.generic](WrappedBasis[M, DT]):
    """Represents a truncated basis."""

    def __init__(self: Self, size: int, inner: Basis[M, DT]) -> None:
        self._size = size
        super().__init__(inner)

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._size

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TruncatedBasis):
            return self._size == value._size and value._inner == self._inner  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash((self._size, self._inner))

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return pad_ft_points(vectors, s=(self._inner.size,), axes=(axis,))

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return pad_ft_points(vectors, s=(self._size,), axes=(axis,))

    @override
    def __convert_vector_into__[
        DT1: np.generic,
    ](  # [DT1: DT, B1: Basis[M1, DT]]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        basis: Basis[BasisMetadata, Never],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors

        if isinstance(basis, TruncatedBasis) and self.inner == basis.inner:
            return pad_ft_points(vectors, s=(basis.size,), axes=(axis,))

        return super().__convert_vector_into__(vectors, basis, axis)

    @override
    def with_rewrapped_inner(
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M, DT]]
    ) -> TruncatedBasis[M, DT]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        TruncatedBasis[M, DT]
        """
        return TruncatedBasis(self.size, wrapper(self.inner))
