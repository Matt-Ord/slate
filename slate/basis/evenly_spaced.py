from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Self, override

import numpy as np

from slate.basis import Basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util.util import slice_along_axis


def _pad_sample_axis[DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    step: int,
    offset: int,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]:
    final_shape = np.array(vectors.shape)
    final_shape[axis] = step * final_shape[axis]
    padded = np.zeros(final_shape, dtype=vectors.dtype)
    # We could alternatively slice starting on zero
    # and roll at the end but this is worse for performance
    vectors = np.roll(vectors, offset // step, axis=axis)  # type: ignore cannot infer dtype
    padded[slice_along_axis(slice(offset % step, None, step), axis)] = vectors

    return padded  # type: ignore[no-any-return]


def _truncate_sample_axis[DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    step: int,
    offset: int,
    axis: int = -1,
) -> np.ndarray[tuple[int, ...], np.dtype[DT]]:
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


class EvenlySpacedBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]]
):
    """Represents a basis sampled evenly along an axis."""

    def __init__(self: Self, spacing: Spacing, inner: Basis[M, DT]) -> None:
        self._spacing = spacing
        super().__init__(inner)
        assert self._inner.size == self._spacing.step * self._spacing.n

    def __hash__(self) -> int:
        return hash((self._inner, self._spacing))

    @property
    def spacing(self: Self) -> Spacing:
        """Spacing of the basis."""
        return self._spacing

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._spacing.n

    def __eq__(self, value: object) -> bool:
        if isinstance(value, EvenlySpacedBasis):
            return self._spacing == value._spacing and value._inner == self._inner  # type: ignore unknown
        return False

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return _pad_sample_axis(vectors, self._spacing.step, self._spacing.offset, axis)

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return _truncate_sample_axis(
            vectors, self._spacing.step, self._spacing.offset, axis
        )

    @override
    def with_rewrapped_inner(
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M, DT]]
    ) -> EvenlySpacedBasis[M, DT]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        TruncatedBasis[M, DT]
        """
        return EvenlySpacedBasis(self.spacing, wrapper(self.inner))
