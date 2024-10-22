from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator, Never, Self, override

import numpy as np

from slate.basis import Basis
from slate.metadata import BasisMetadata


class WrappedBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Never] = Basis[M, DT],  # : Basis[M, DT]  # noqa: E251
](Basis[M, DT]):
    """Represents a truncated basis."""

    def __init__(self: Self, inner: Basis[M, DT]) -> None:
        self._inner = inner
        self._metadata = inner.metadata

    @property
    def inner(self: Self) -> Basis[M, DT]:
        """Inner basis."""
        return self._inner

    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
        B1: Basis[BasisMetadata, Never] = Basis[M1, DT1],
    ](self: Self, inner: Basis[M1, DT1]) -> WrappedBasis[M1, DT1, B1]:
        """Get the wrapped basis with the inner set to inner.

        Parameters
        ----------
        self : Self
        inner : B

        Returns
        -------
        WrappedBasis[M, DT, B]
        """
        return self.with_modified_inner(lambda _: inner)

    @abstractmethod
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
        B1: Basis[BasisMetadata, Never] = Basis[M1, DT1],
    ](self: Self, wrapper: Callable[[B], Basis[M1, DT1]]) -> WrappedBasis[M1, DT1, B1]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        WrappedBasis[M, DT]
        """

    @abstractmethod
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]: ...

    @abstractmethod
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]: ...

    def __into_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = self.__into_inner__(vectors, axis)
        return self._inner.__into_fundamental__(transformed, axis=axis)

    def __from_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = self._inner.__from_fundamental__(vectors, axis=axis)
        return self.__from_inner__(transformed, axis)

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

        as_inner = self.__into_inner__(vectors, axis)

        if isinstance(basis, WrappedBasis) and self.inner == basis.inner:
            return basis.__from_inner__(as_inner, axis)
        return self._inner.__convert_vector_into__(as_inner, basis, axis=axis)  # type: ignore unknown


def wrapped_basis_iter_inner[
    M: BasisMetadata,
    DT: np.generic,
](basis: WrappedBasis[M, DT, Any]) -> Iterator[Basis[M, DT]]:
    """Return successive calls to basis.inner until the basis is not a WrappedBasis."""
    yield basis.inner
    if isinstance(basis.inner, WrappedBasis):
        yield from wrapped_basis_iter_inner(basis.inner)  # type: ignore unknown


def get_wrapped_basis_super_inner[
    M: BasisMetadata,
    DT: np.generic,
](basis: WrappedBasis[M, DT, Any]) -> Basis[M, DT]:
    """Get the `super inner` of a wrapped basis.

    If the inner is itself a wrapped basis, return the super inner of that basis
    """
    *_, last = wrapped_basis_iter_inner(basis)
    return last
