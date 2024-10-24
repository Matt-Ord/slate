from __future__ import annotations

from typing import Any, Callable, Self, override

import numpy as np

from slate.basis import Basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata


class CoBasis[M: BasisMetadata, DT: np.generic](WrappedBasis[M, DT, Basis[M, DT]]):
    """Represents a fourier transformed basis."""

    def __eq__(self, value: object) -> bool:
        if isinstance(value, CoBasis):
            return self.size == value.size and value.inner == self.inner  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash((1, self.inner))

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self.inner.size

    @override
    def __into_inner__[DT1: np.generic](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return np.conj(vectors)

    @override
    def __from_inner__[DT1: np.generic](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return np.conj(vectors)

    @override
    def with_inner[M1: BasisMetadata, DT1: np.generic](
        self: Self, inner: Basis[M1, DT1]
    ) -> CoBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[M1: BasisMetadata, DT1: np.generic](
        self: Self,
        wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]],
    ) -> CoBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        TruncatedBasis[M, DT]
        """
        return CoBasis(wrapper(self.inner))
