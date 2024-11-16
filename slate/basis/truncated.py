from __future__ import annotations

from typing import Any, Callable, Self, override

import numpy as np

from slate.basis import Basis
from slate.basis._basis import SimpleBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util._pad import (
    Padding,
    Truncation,
    assert_unique_indices,
    pad_along_axis,
    truncate_along_axis,
)


class TruncatedBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]], SimpleBasis
):
    """Represents a basis sampled evenly along an axis."""

    def __init__(self: Self, truncation: Truncation, inner: Basis[M, DT]) -> None:
        self._truncation = truncation
        super().__init__(inner)
        assert isinstance(self.inner, SimpleBasis)
        assert_unique_indices(
            self._inner.size, self._truncation.n, self._truncation.step
        )

    def __hash__(self) -> int:
        return hash((self._inner, self._truncation))

    @property
    def truncation(self: Self) -> Truncation:
        """Truncation of the basis."""
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
