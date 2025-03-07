from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    TypeGuard,
    cast,
    overload,
    override,
)

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, ctype
from slate_core.basis._tuple import TupleBasis, TupleBasis2D
from slate_core.basis._wrapped import WrappedBasis, wrapped_basis_iter_inner
from slate_core.metadata import BasisMetadata

from ._upcast import AsUpcast

if TYPE_CHECKING:
    from slate_core.metadata import TupleMetadata


class DiagonalBasis[
    B: TupleBasis2D = TupleBasis2D,
    DT: ctype[Never] = ctype[Never],
](
    WrappedBasis[B, DT],
):
    """Represents a diagonal basis."""

    def __init__[B_: TupleBasis[tuple[Basis, Basis], Any]](
        self: DiagonalBasis[B_, ctype[Never]], inner: B_
    ) -> None:
        super().__init__(cast("B", inner))
        assert self.inner.children[0].size == self.inner.children[1].size

    @override
    def resolve_ctype[DT_: ctype[Never]](
        self: DiagonalBasis[TupleBasis[tuple[Basis, Basis], Any, DT_], Any],
    ) -> DiagonalBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("DiagonalBasis[B, DT_]", self)

    @override
    def upcast[M0: BasisMetadata, M1: BasisMetadata, E](
        self: DiagonalBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> AsUpcast[DiagonalBasis[B, DT], TupleMetadata[tuple[M0, M1], E], DT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

    @property
    @override
    def size(self) -> int:
        return self.inner.children[0].size

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic](
        self: DiagonalBasis[Any, ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT1]:
        def fn() -> np.ndarray[Any, np.dtype[DT2]]:
            if vectors.size == 0:
                return vectors
            swapped = vectors.swapaxes(axis, 0)
            stacked = swapped.reshape(self.size, *swapped.shape[1:])

            return (
                cast(
                    "np.ndarray[Any, np.dtype[DT2]]",
                    np.einsum(  # type: ignore lib
                        "i...,ij->ij...",
                        stacked,  # type: ignore lib
                        np.eye(
                            self.inner.children[0].size, self.inner.children[1].size
                        ),
                    ),
                )
                .reshape(-1, *swapped.shape[1:])
                .swapaxes(axis, 0)
            )

        return BasisConversion(fn)

    @override
    def __from_inner__[DT2: np.generic, DT3: np.generic](
        self: DiagonalBasis[Any, ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT3, DT2, DT3]:
        def fn() -> np.ndarray[Any, np.dtype[DT2]]:
            if vectors.size == 0:
                return vectors
            swapped = vectors.swapaxes(axis, 0)
            stacked = swapped.reshape(*self.inner.shape, *swapped.shape[1:])

            return (
                cast(
                    "np.ndarray[Any, np.dtype[DT2]]",
                    np.einsum("ii...->i...", stacked),  # type: ignore lib
                )
                .reshape(self.size, *swapped.shape[1:])
                .swapaxes(axis, 0)
            )

        return BasisConversion(fn)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            is_diagonal_basis(other)
            and (other.inner == self.inner)
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
    def add_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number](
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

        return (
            cast("WrappedBasis[Any, ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


@overload
def is_diagonal_basis[M1: BasisMetadata, M2: BasisMetadata, E, DT: ctype[Never]](  # type: ignore not overlapping
    basis: Basis[TupleMetadata[tuple[M1, M2], E], DT],
) -> TypeGuard[
    DiagonalBasis[TupleBasis[tuple[Basis[M1, DT], Basis[M2, DT]], E], DT]
]: ...
@overload
def is_diagonal_basis[M1: BasisMetadata, M2: BasisMetadata, E, DT: ctype[Never]](
    basis: Basis[BasisMetadata, DT],
) -> TypeGuard[
    DiagonalBasis[
        TupleBasis[tuple[Basis[BasisMetadata, DT], Basis[BasisMetadata, DT]], Never], DT
    ]
]: ...
@overload
def is_diagonal_basis(basis: object) -> TypeGuard[DiagonalBasis]: ...


def is_diagonal_basis(basis: object) -> TypeGuard[DiagonalBasis]:
    return isinstance(basis, DiagonalBasis)


@overload
def as_diagonal_basis[M1: BasisMetadata, M2: BasisMetadata, E, DT: ctype[Never]](  # type: ignore not overlapping
    basis: Basis[TupleMetadata[tuple[M1, M2], E], DT],
) -> DiagonalBasis[TupleBasis[tuple[Basis[M1, DT], Basis[M2, DT]], E], DT] | None: ...
@overload
def as_diagonal_basis[DT: ctype[Never]](
    basis: Basis[BasisMetadata, DT],
) -> (
    DiagonalBasis[
        TupleBasis[tuple[Basis[BasisMetadata, DT], Basis[BasisMetadata, DT]], Never], DT
    ]
    | None
): ...


def as_diagonal_basis[DT: ctype[Never]](
    basis: Basis[BasisMetadata, DT],
) -> DiagonalBasis[Any, DT] | None:
    """Get the closest basis that supports the feature set."""
    shape = basis.metadata().fundamental_shape
    if isinstance(shape, int) or len(shape) != 2:  # noqa: PLR2004
        return None
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if is_diagonal_basis(b)), None
    )
