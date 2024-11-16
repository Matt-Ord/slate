from __future__ import annotations

from typing import Any, Callable, Never, Self, override

import numpy as np

from slate.basis import Basis, SimpleBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util import pad_ft_points


class CroppedBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]], SimpleBasis
):
    """Represents a cropped basis."""

    def __init__(self: Self, size: int, inner: Basis[M, DT]) -> None:
        self._size = size
        super().__init__(inner)
        assert isinstance(self.inner, SimpleBasis)

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._size

    def __eq__(self, value: object) -> bool:
        if isinstance(value, CroppedBasis):
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

        if isinstance(basis, CroppedBasis) and self.inner == basis.inner:
            return pad_ft_points(vectors, s=(basis.size,), axes=(axis,))

        return super().__convert_vector_into__(vectors, basis, axis)

    @override
    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](self: Self, inner: Basis[M1, DT1]) -> CroppedBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]]
    ) -> CroppedBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return CroppedBasis(self.size, wrapper(self.inner))

    @override
    def conjugate_basis(self) -> CroppedBasis[M, DT]:
        return CroppedBasis(self.size, self.inner.conjugate_basis())
