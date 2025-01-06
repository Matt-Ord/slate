from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    TypeGuard,
    cast,
    overload,
    override,
)

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis._tuple import TupleBasis2D, tuple_basis
from slate.basis.wrapped import WrappedBasis, wrapped_basis_iter_inner
from slate.metadata import Metadata2D
from slate.metadata._metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.metadata.stacked import StackedMetadata


class DiagonalBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[Metadata2D[Any, Any, E], DT, TupleBasis2D[DT, Any, Any, E]],
):
    """Represents a diagonal basis."""

    def __init__[_DT: np.generic, _B0: Basis[Any, Any], _B1: Basis[Any, Any], _E](
        self: DiagonalBasis[_DT, _B0, _B1, _E],
        inner: TupleBasis2D[_DT, _B0, _B1, _E],
    ) -> None:
        super().__init__(cast("Any", inner))
        assert self.inner.children[0].size == self.inner.children[1].size

    @property
    @override
    def inner(self) -> TupleBasis2D[DT, B0, B1, E]:
        return cast("TupleBasis2D[DT, B0, B1, E]", self._inner)

    @property
    @override
    def size(self) -> int:
        return self.inner.children[0].size

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(self.size, *swapped.shape[1:])

        return (
            cast(
                "np.ndarray[Any, np.dtype[DT]]",
                np.einsum(  # type: ignore lib
                    "i...,ij->ij...",
                    stacked,  # type: ignore lib
                    np.eye(self.inner.children[0].size, self.inner.children[1].size),
                ),
            )
            .reshape(-1, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(*self.inner.shape, *swapped.shape[1:])

        return (
            cast(
                "np.ndarray[Any, np.dtype[DT]]",
                np.einsum("ii...->i...", stacked),  # type: ignore lib
            )
            .reshape(self.size, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DiagonalBasis)
            and (other.inner == self.inner)  # type: ignore unknown
            and self.is_dual == other.is_dual
        )

    @override
    def __hash__(self) -> int:
        return hash((2, self.inner, self.is_dual))

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        DT1: np.generic,
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](self, inner: TupleBasis2D[DT1, B01, B11, E1]) -> DiagonalBasis[DT1, B01, B11, E1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        DT1: np.generic,
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self,
        wrapper: Callable[
            [TupleBasis2D[DT, Any, Any, E]], TupleBasis2D[DT1, B01, B11, E1]
        ],
    ) -> DiagonalBasis[DT1, B01, B11, E1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return DiagonalBasis[DT1, B01, B11, E1](wrapper(self.inner))

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "SIMPLE_ADD" in self.inner.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.inner.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.inner.features:
            out.add("SUB")
            out.add("SIMPLE_SUB")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)
        return self.__from_inner__(self.inner.points)


@overload
def diagonal_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any]](
    children: tuple[_B0, _B1], extra_metadata: None = None
) -> DiagonalBasis[Any, _B0, _B1, None]: ...


@overload
def diagonal_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any], E](
    children: tuple[_B0, _B1], extra_metadata: E
) -> DiagonalBasis[Any, _B0, _B1, E]: ...


def diagonal_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any], E](
    children: tuple[_B0, _B1], extra_metadata: E | None = None
) -> DiagonalBasis[Any, _B0, _B1, E | None]:
    """Build a VariadicTupleBasis from a tuple."""
    return DiagonalBasis(tuple_basis(children, extra_metadata))


def _is_diagonal_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[StackedMetadata[M, E], DT],
) -> TypeGuard[DiagonalBasis[DT, Basis[M, DT], Basis[M, DT], E]]:
    return isinstance(basis, DiagonalBasis)


@overload
def as_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[Metadata2D[M0, M1, E], DT],
) -> DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E] | None: ...
@overload
def as_diagonal_basis[
    M: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[StackedMetadata[M, E], DT],
) -> DiagonalBasis[DT, Basis[M, DT], Basis[M, DT], E] | None: ...
@overload
def as_diagonal_basis[
    DT: np.generic,
](
    basis: Basis[BasisMetadata, DT],
) -> (
    DiagonalBasis[DT, Basis[BasisMetadata, DT], Basis[BasisMetadata, DT], Any] | None
): ...


def as_diagonal_basis(
    basis: Any,
) -> Any:
    """Get the closest basis that supports the feature set."""
    assert len(basis.metadata().fundamental_shape) == 2  # noqa: PLR2004
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if _is_diagonal_basis(b)),
        None,
    )
