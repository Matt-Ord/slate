from __future__ import annotations

from typing import Any, Callable, Self, cast, override

import numpy as np

from slate.basis import Basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import StackedMetadata

from ._tuple_basis import VariadicTupleBasis


class DiagonalBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[StackedMetadata[Any, E], DT, VariadicTupleBasis[DT, B0, B1, E]],
):
    """Represents a diagonal basis."""

    def __init__(self: Self, inner: VariadicTupleBasis[DT, B0, B1, E]) -> None:
        super().__init__(inner)
        assert self.inner.children[0].size == self.inner.children[1].size

    @property
    def inner(self: Self) -> VariadicTupleBasis[DT, B0, B1, E]:
        return cast(VariadicTupleBasis[DT, B0, B1, E], self._inner)

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

    @override
    def with_rewrapped_inner(
        self: Self,
        wrapper: Callable[
            [VariadicTupleBasis[DT, B0, B1, E]], VariadicTupleBasis[DT, B0, B1, E]
        ],
    ) -> DiagonalBasis[DT, B0, B1, E]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        TruncatedBasis[M, DT]
        """
        return DiagonalBasis(wrapper(self.inner))
