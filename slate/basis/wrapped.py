from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator, Never, Self, override

import numpy as np

from slate.basis import Basis
from slate.basis.fundamental import FundamentalBasis
from slate.metadata import BasisMetadata


class WrappedBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Never] = Basis[M, DT],  # : Basis[M, DT]
](Basis[M, DT]):
    """Represents a truncated basis."""

    def __init__(self: Self, inner: Basis[M, DT]) -> None:
        self._inner = inner
        self._metadata = inner.metadata()

    @property
    def inner(self: Self) -> Basis[M, DT]:
        """Inner basis."""
        return self._inner

    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
        B1: Basis[BasisMetadata, Never] = Basis[M1, DT1],
    ](self: Self, inner: Basis[M1, DT1]) -> WrappedBasis[M1, DT1, B1]:
        """Get the wrapped basis with the inner set to inner."""
        return self.with_modified_inner(lambda _: inner)

    @abstractmethod
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
        B1: Basis[BasisMetadata, Never] = Basis[M1, DT1],
    ](self: Self, wrapper: Callable[[B], Basis[M1, DT1]]) -> WrappedBasis[M1, DT1, B1]:
        """Get the wrapped basis after wrapper is applied to inner."""

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
        assert self.metadata() == basis.metadata()

        if self == basis:
            return vectors

        as_inner = self.__into_inner__(vectors, axis)

        if isinstance(basis, WrappedBasis) and self.inner == basis.inner:
            return basis.__from_inner__(as_inner, axis)
        return self._inner.__convert_vector_into__(as_inner, basis, axis=axis)  # type: ignore unknown


def wrapped_basis_iter_inner[
    M: BasisMetadata,
    DT: np.generic,
](basis: Basis[M, DT]) -> Iterator[Basis[M, DT]]:
    """Return successive calls to basis.inner until the basis is not a WrappedBasis."""
    yield basis
    if isinstance(basis, WrappedBasis):
        yield from wrapped_basis_iter_inner(basis.inner)  # type: ignore unknown


def get_wrapped_basis_super_inner[
    M: BasisMetadata,
    DT: np.generic,
](basis: Basis[M, DT]) -> Basis[M, DT]:
    """Get the `super inner` of a wrapped basis.

    If the inner is itself a wrapped basis, return the super inner of that basis
    """
    *_, last = wrapped_basis_iter_inner(basis)
    return last


def as_add_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports addition.

    If the basis is already an add basis, return it.
    If it wraps an add basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if "ADD" in b.features),
        FundamentalBasis(basis.metadata()),
    )


def as_sub_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports subtraction.

    If the basis is already a sub basis, return it.
    If it wraps a sub basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if "SUB" in b.features),
        FundamentalBasis(basis.metadata()),
    )


def as_mul_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports multiplication.

    If the basis is already a mul basis, return it.
    If it wraps a mul basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if "MUL" in b.features),
        FundamentalBasis(basis.metadata()),
    )
