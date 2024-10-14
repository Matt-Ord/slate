from __future__ import annotations

from typing import Any, Self, cast

import numpy as np

from slate.basis import Basis
from slate.basis.metadata import BasisMetadata
from slate.basis.wrapped import WrappedBasis

from ._tuple_basis import TupleMetadata, VariadicTupleBasis


class DiagonalBasis[
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
    DT: np.generic,
](
    WrappedBasis[TupleMetadata[BasisMetadata, E], DT],
):
    """Represents a diagonal basis."""

    def __init__(self: Self, inner: VariadicTupleBasis[B0, B1, E, DT]) -> None:
        super().__init__(inner)
        assert self.inner.children[0].size == self.inner.children[1].size

    @property
    def inner(self: Self) -> VariadicTupleBasis[B0, B1, E, DT]:
        return cast(VariadicTupleBasis[B0, B1, E, DT], self._inner)

    @property
    def size(self) -> int:
        """Number of elements in the basis."""
        return self.inner.children[0].size

    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(self.size, *swapped.shape[1:])

        return (
            cast(
                np.ndarray[Any, np.dtype[DT]],
                np.einsum(  # type: ignore lib
                    "i...,ij->ij...",
                    stacked,  # type: ignore lib
                    np.eye(self.inner.children[0].size, self.inner.children[1].size),
                ),
            )
            .reshape(-1, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(*self.inner.shape, *swapped.shape[1:])

        return (
            cast(
                np.ndarray[Any, np.dtype[DT]],
                np.einsum("ii...->i...", stacked),  # type: ignore lib
            )
            .reshape(self.size, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiagonalBasis) and (other.inner == self.inner)  # type: ignore unknown

    def __hash__(self) -> int:
        return hash((2, self.inner))
