from __future__ import annotations

from typing import Any, Iterator, Self, cast, override

import numpy as np

from slate.basis.basis import Basis
from slate.basis.metadata import BasisMetadata

from .tuple_basis import TupleBasis, TupleMetadata


class DiagonalBasis[_M: BasisMetadata, _E, _DT: np.generic](
    Basis[TupleMetadata[_M, _E], _DT]
):
    """Represents a diagonal basis."""

    def __init__(self: Self, inner: TupleBasis[_M, _E, _DT]) -> None:
        self._inner = inner
        assert inner.n_dim == 2  # noqa: PLR2004
        self._metadata = inner.metadata

    @property
    def inner(self) -> TupleBasis[_M, _E, _DT]:
        """Inner basis."""
        return self._inner

    @property
    def size(self) -> int:
        """Number of elements in the basis."""
        return self._inner[0].size

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the inner data."""
        return tuple(x.size for x in self._inner)

    def __iter__(self) -> Iterator[Basis[_M, _DT]]:
        return self._inner.__iter__()

    def __getitem__(self: Self, index: int) -> Basis[_M, _DT]:
        return self._inner[index]

    @override
    def __into_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(self.size, *swapped.shape[1:])

        inner = (
            cast(
                np.ndarray[Any, np.dtype[_DT]],
                np.einsum(  # type: ignore lib
                    "i...,ij->ij...",
                    stacked,  # type: ignore lib
                    np.eye(self._inner[0].size, self.inner[1].size),
                ),
            )
            .reshape(-1, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )
        return self._inner.__into_fundamental__(inner, axis)

    @override
    def __from_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        inner = self._inner.__from_fundamental__(vectors, axis)
        swapped = inner.swapaxes(axis, 0)
        stacked = swapped.reshape(*self.inner.shape, *swapped.shape[1:])

        return (
            cast(
                np.ndarray[Any, np.dtype[_DT]],
                np.einsum("ii...->i...", stacked),  # type: ignore lib
            )
            .reshape(self.size, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiagonalBasis) and (other.inner == self.inner)  # type: ignore unknown

    def __hash__(self) -> int:
        return hash((2, self.inner))
