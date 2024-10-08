from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Self, override

import numpy as np

from slate.basis.basis import Basis
from slate.basis.metadata import BasisMetadata
from slate.util.util import slice_along_axis


def _pad_sample_axis[_DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[_DT]],
    step: int,
    offset: int,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[_DT]]:
    final_shape = np.array(vectors.shape)
    final_shape[axis] = step * final_shape[axis]
    padded = np.zeros(final_shape, dtype=vectors.dtype)
    # We could alternatively slice starting on zero
    # and roll at the end but this is worse for performance
    vectors = np.roll(vectors, offset // step, axis=axis)  # type: ignore cannot infer dtype
    padded[slice_along_axis(slice(offset % step, None, step), axis)] = vectors

    return padded  # type: ignore[no-any-return]


def _truncate_sample_axis[_DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[_DT]],
    step: int,
    offset: int,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[_DT]]:
    truncated = vectors[slice_along_axis(slice(offset % step, None, step), axis)]  # type: ignore index type wrong
    # We could alternatively roll before we take the slice
    # and slice(0, None, ns) but this is worse for performance
    return np.roll(truncated, -(offset // step), axis=axis)  # type: ignore[no-any-return]


@dataclass(frozen=True)
class Spacing:
    """Spacing of the basis."""

    n: int
    step: int
    offset: int


class EvenlySpacedBasis[_M: BasisMetadata, _DT: np.generic](Basis[_M, _DT]):
    """Represents a basis sampled evenly along an axis."""

    def __init__(self: Self, spacing: Spacing, inner: Basis[_M, _DT]) -> None:
        self._inner = inner
        self._metadata = inner.metadata
        self._spacing = spacing
        assert self._inner.size == self._spacing.step * self._spacing.n

    def __hash__(self) -> int:
        return hash((self._inner, self._spacing))

    @property
    def spacing(self: Self) -> Spacing:
        """Spacing of the basis."""
        return self._spacing

    @property
    def inner(self: Self) -> Basis[_M, _DT]:
        """Inner basis."""
        return self._inner

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._spacing.n

    def __eq__(self, value: object) -> bool:
        if isinstance(value, EvenlySpacedBasis):
            return self._spacing == value._spacing and value._inner == self._inner  # type: ignore unknown
        return False

    @override
    def __into_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        inner = _pad_sample_axis(
            vectors, self._spacing.step, self._spacing.offset, axis
        )
        return self._inner.__into_fundamental__(inner, axis=axis)

    @override
    def __from_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        inner = self._inner.__from_fundamental__(vectors, axis=axis)
        return _truncate_sample_axis(
            inner, self._spacing.step, self._spacing.offset, axis
        )

    @override
    def __convert_vector_into__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        basis: Basis[_M, _DT],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors
        inner = _pad_sample_axis(
            vectors, self._spacing.step, self._spacing.offset, axis
        )
        if isinstance(basis, EvenlySpacedBasis) and self.inner == basis.inner:
            return _truncate_sample_axis(
                inner,
                basis.spacing.step,
                basis.spacing.offset,
                axis,
            )

        return self._inner.__convert_vector_into__(inner, basis, axis=axis)
