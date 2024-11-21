from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Iterator, Never, Self, cast, override

import numpy as np

from slate.basis import Basis, BasisFeature, FundamentalBasis
from slate.metadata import BasisMetadata


class WrappedBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Any] = Basis[M, DT],  # : Basis[M, DT]
](Basis[M, DT]):
    """Represents a truncated basis."""

    def __init__(self: Self, inner: Basis[M, DT]) -> None:
        self._inner = cast(B, inner)
        self._metadata = inner.metadata()

    @property
    def inner(self: Self) -> B:
        """Inner basis."""
        return self._inner

    def with_inner[
        _M: BasisMetadata,
        _DT: np.generic,
        _B: Basis[BasisMetadata, Any] = Basis[_M, _DT],
    ](self: WrappedBasis[_M, _DT, B], inner: _B) -> WrappedBasis[_M, _DT, _B]:
        """Get the wrapped basis with the inner set to inner."""
        return self.with_modified_inner(lambda _: inner)

    @abstractmethod
    def with_modified_inner[
        _M: BasisMetadata,
        _DT: np.generic,
        _B: Basis[BasisMetadata, Any] = Basis[_M, _DT],
    ](
        self: WrappedBasis[_M, _DT, B], wrapper: Callable[[B], _B]
    ) -> WrappedBasis[_M, _DT, _B]:
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

    @override
    def __into_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = self.__into_inner__(vectors, axis)
        return self._inner.__into_fundamental__(transformed, axis=axis)

    @override
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

        if isinstance(basis, WrappedBasis) and self.inner == basis.inner:  # type: ignore unknown
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


def as_feature_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT], features: set[BasisFeature]
) -> Basis[M, DT]:
    """Get the closest basis that supports the feature set."""
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if features <= b.features),
        FundamentalBasis(basis.metadata()),
    )


def as_add_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports addition.

    If the basis is already an ADD basis, return it.
    If it wraps an ADD basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"ADD"})


def as_sub_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports subtraction.

    If the basis is already a SUB basis, return it.
    If it wraps a SUB basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"SUB"})


def as_mul_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports MUL.

    If the basis is already a MUL basis, return it.
    If it wraps a MUL basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"MUL"})


def as_index_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports INDEX.

    If the basis is already an INDEX basis, return it.
    If it wraps a INDEX basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"INDEX"})
