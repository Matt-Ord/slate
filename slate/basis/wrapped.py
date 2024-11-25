from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Never, Self, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis.fundamental import FundamentalBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator


class WrappedBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Any] = Basis[M, DT],  # : Basis[M, DT]
](Basis[M, DT]):
    """A wrapped basis, represents some transformation over an underlying 'inner' basis."""

    def __init__(self: Self, inner: Basis[M, DT]) -> None:
        self._inner = cast(B, inner)
        self._metadata = inner.metadata()
        super().__init__(inner.metadata(), is_dual=inner.is_dual)

    @property
    @override
    def is_dual(self: Self) -> bool:
        return self.inner.is_dual

    @override
    def dual_basis(self) -> Self:
        copied = copy(self)
        copied._inner = self._inner.dual_basis()  # noqa: SLF001
        copied._is_dual = copied.inner.is_dual  # noqa: SLF001
        return copied

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
        _B: Basis[BasisMetadata, Any] = Basis[_M, Any],
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
        return self.inner.__convert_vector_into__(as_inner, basis, axis=axis)  # type: ignore unknown


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


def get_common_basis[M: BasisMetadata, E, DT: np.generic](
    rhs: Basis[M, DT],
    lhs: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest common basis of two bases."""
    assert lhs.metadata() == rhs.metadata()
    lhs_rev = reversed(list(wrapped_basis_iter_inner(lhs)))
    rhs_rev = reversed(list(wrapped_basis_iter_inner(rhs)))

    last_common = FundamentalBasis(lhs.metadata())
    for a, b in zip(lhs_rev, rhs_rev):
        if a != b:
            return last_common
        last_common = a
    return last_common


def as_feature_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT], features: set[BasisFeature]
) -> Basis[M, DT]:
    """Get the closest basis that supports the feature set."""
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if features <= b.features),
        FundamentalBasis(basis.metadata()),
    )


def as_add_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports addition.

    If the basis is already an ADD basis, return it.
    If it wraps an ADD basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"ADD"})


def as_sub_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports subtraction.

    If the basis is already a SUB basis, return it.
    If it wraps a SUB basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"SUB"})


def as_mul_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports MUL.

    If the basis is already a MUL basis, return it.
    If it wraps a MUL basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"MUL"})


def as_index_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports INDEX.

    If the basis is already an INDEX basis, return it.
    If it wraps a INDEX basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"INDEX"})
