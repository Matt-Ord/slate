from __future__ import annotations

from typing import Any, Callable, Self, override

import numpy as np

from slate.basis import Basis, SimpleBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata._metadata import BasisMetadata


class RecastBasis[M0: BasisMetadata, M1: BasisMetadata, DT: np.generic](
    WrappedBasis[M0, DT, Basis[M0, DT]], SimpleBasis
):
    """Represents a truncated basis."""

    def __init__(
        self: Self,
        inner: Basis[M0, DT],
        inner_recast: Basis[M1, DT],
        outer_recast: Basis[M1, DT],
    ) -> None:
        self._inner_recast = inner_recast
        self._outer_recast = outer_recast
        super().__init__(inner)
        assert self._inner_recast.size == self.inner.size

    @property
    def size(self: Self) -> int:
        """The size of the basis."""
        return self.outer_recast.size

    @property
    def inner_recast(self: Self) -> Basis[M1, DT]:
        """The basis the inner was recast to."""
        return self._inner_recast

    @property
    def outer_recast(self: Self) -> Basis[M1, DT]:
        """The basis the inner recast was transformed to."""
        return self._outer_recast

    @override
    def conjugate_basis(self) -> Basis[M0, DT]:
        return self

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return self._outer_recast.__convert_vector_into__(
            vectors, self._inner_recast, axis
        )

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return self._inner_recast.__convert_vector_into__(
            vectors, self._outer_recast, axis
        )

    @override
    def with_inner[
        M2: BasisMetadata,
    ](self: Self, inner: Basis[M2, DT]) -> RecastBasis[M2, M1, DT]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M2: BasisMetadata,
    ](
        self: Self, wrapper: Callable[[Basis[M0, DT]], Basis[M2, DT]]
    ) -> RecastBasis[M2, M1, DT]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return RecastBasis(wrapper(self.inner), self.inner_recast, self.outer_recast)
