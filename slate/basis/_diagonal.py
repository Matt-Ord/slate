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

    def __init__[DT_: np.generic, B0_: Basis[Any, Any], B1_: Basis[Any, Any], E_](
        self: DiagonalBasis[DT_, B0_, B1_, E_],
        inner: TupleBasis2D[DT_, B0_, B1_, E_],
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
        if vectors.size == 0:
            return vectors
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
        if vectors.size == 0:
            return vectors
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

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
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
def diagonal_basis[B0: Basis[Any, Any], B1: Basis[Any, Any]](
    children: tuple[B0, B1], extra_metadata: None = None
) -> DiagonalBasis[Any, B0, B1, None]: ...


@overload
def diagonal_basis[B0: Basis[Any, Any], B1: Basis[Any, Any], E](
    children: tuple[B0, B1], extra_metadata: E
) -> DiagonalBasis[Any, B0, B1, E]: ...


def diagonal_basis[B0: Basis[Any, Any], B1: Basis[Any, Any], E](
    children: tuple[B0, B1], extra_metadata: E | None = None
) -> DiagonalBasis[Any, B0, B1, E | None]:
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
