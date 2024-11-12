from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Self, override

import numpy as np

from slate.basis import Basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util import slice_along_axis


@dataclass(frozen=True)
class Truncation:
    """Spacing of the basis."""

    n: int
    step: int
    offset: int


def _assert_unique_indices(size: int, n: int, step: int) -> None:
    assert n <= size // math.gcd(step, size), "Indices must be unique"


def _get_truncated_indices(
    size: int,
    truncation: Truncation,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    n = truncation.n
    step = truncation.step
    offset = truncation.offset

    _assert_unique_indices(size, n, step)
    return (offset + step * np.arange(n)) % size


def _truncate_along_axis_continuous[DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    truncation: Truncation,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]:
    step = truncation.step
    offset = truncation.offset
    n = truncation.n
    # We could alternatively roll after we take the slice
    # but this more complex (ie not worth performance)
    rolled = np.roll(vectors, -offset, axis=axis)
    return rolled[slice_along_axis(slice(0, n * step, step), axis)]  # type: ignore index type wrong


def _is_contiguous_truncate(size: int, truncation: Truncation) -> bool:
    return size >= truncation.step * truncation.n


def truncate_along_axis[DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    truncation: Truncation,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]:
    """Truncate Data along an axis."""
    if _is_contiguous_truncate(vectors.shape[axis], truncation):
        return _truncate_along_axis_continuous(vectors, truncation, axis)

    data_length = vectors.shape[axis]
    indices = _get_truncated_indices(data_length, truncation)
    return np.take(vectors, indices.astype(int), axis=axis)


@dataclass(frozen=True)
class Padding:
    """Spacing of the basis."""

    n: int
    step: int
    offset: int


def _pad_along_axis_continuous[DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    padding: Padding,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]:
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


def pad_along_axis[DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    padding: Padding,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]:
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


class TruncatedBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]]
):
    """Represents a basis sampled evenly along an axis."""

    def __init__(self: Self, truncation: Truncation, inner: Basis[M, DT]) -> None:
        self._truncation = truncation
        super().__init__(inner)
        _assert_unique_indices(
            self._inner.size, self._truncation.n, self._truncation.step
        )

    def __hash__(self) -> int:
        return hash((self._inner, self._truncation))

    @property
    def truncation(self: Self) -> Truncation:
        """Spacing of the basis."""
        return self._truncation

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._truncation.n

    @override
    def conjugate_basis(self) -> TruncatedBasis[M, DT]:
        return self

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TruncatedBasis):
            return self._truncation == value._truncation and value._inner == self._inner  # type: ignore unknown
        return False

    @property
    def _inner_padding(self: Self) -> Padding:
        return Padding(self._inner.size, self._truncation.step, self._truncation.offset)

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return pad_along_axis(vectors, self._inner_padding, axis)

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return truncate_along_axis(vectors, self.truncation, axis)

    @override
    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](self: Self, inner: Basis[M1, DT1]) -> TruncatedBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]]
    ) -> TruncatedBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return TruncatedBasis(self.truncation, wrapper(self.inner))
