from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, TypeGuard, cast, overload, override

import numpy as np

from slate_core.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    Ctype,
    UnionCtype,
)
from slate_core.basis._diagonal import DiagonalBasis, as_diagonal
from slate_core.basis._wrapped import WrappedBasis
from slate_core.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate_core.basis._tuple import TupleBasis, TupleBasisLike
    from slate_core.basis._wrapped import AsUpcast


class RecastBasis[
    BInner: Basis = Basis,
    BInnerRecast: Basis = Basis,
    BOuterRecast: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
](WrappedBasis[BInner, CT]):
    """Represents a truncated basis."""

    def __init__[BInner_: Basis, BInnerRecast_: Basis, BOuterRecast_: Basis](
        self: RecastBasis[BInner_, BInnerRecast_, BOuterRecast_, Ctype[Never]],
        inner: BInner_,
        inner_recast: BInnerRecast_,
        outer_recast: BOuterRecast_,
    ) -> None:
        self._inner_recast = cast("BInnerRecast", inner_recast)
        self._outer_recast = cast("BOuterRecast", outer_recast)
        super().__init__(cast("BInner", inner))

        assert self._inner_recast.size == self.inner.size
        assert self._outer_recast.metadata() == self.inner_recast.metadata()

    @property
    @override
    def ctype(self) -> CT:
        return cast(
            "CT",
            UnionCtype(
                (self.inner.ctype, self.inner_recast.ctype, self.outer_recast.ctype)
            ),
        )

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: RecastBasis[Basis[Any, DT_], Basis[Any, DT_], Basis[Any, DT_], Any],
    ) -> RecastBasis[BInner, BInnerRecast, BOuterRecast, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("RecastBasis[BInner, BInnerRecast, BOuterRecast, DT_]", self)

    @override
    def upcast[M: BasisMetadata](
        self: RecastBasis[Basis[M, Any]],
    ) -> AsUpcast[RecastBasis[BInner, BInnerRecast, BOuterRecast, CT], M, CT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast(
            "AsUpcast[RecastBasis[BInner, BInnerRecast, BOuterRecast, CT], M, CT]",
            super().upcast(),
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            is_recast_basis(other)
            and self.inner == other.inner
            and self.inner_recast == other.inner_recast
            and self.outer_recast == other.outer_recast
        )

    @override
    def __hash__(self) -> int:
        return hash((self.inner, self.inner_recast, self.outer_recast))

    @property
    @override
    def size(self) -> int:
        """The size of the basis."""
        return self.outer_recast.size

    @property
    def inner_recast(self) -> BInnerRecast:
        """The basis the inner was recast to."""
        return self._inner_recast

    @property
    def outer_recast(self) -> BOuterRecast:
        """The basis the inner recast was transformed to."""
        return self._outer_recast

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: RecastBasis[Basis, Basis[BasisMetadata, Ctype[DT3]], Basis, Ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion[DT1, DT2, DT3](
            lambda: self._outer_recast.__convert_vector_into__(
                vectors, self._inner_recast, axis
            ).ok()  # type: ignore safe
        )

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: RecastBasis[Basis, Basis, Basis[BasisMetadata, Ctype[DT1]], Ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion[DT1, DT2, DT3](
            lambda: self._inner_recast.__convert_vector_into__(
                vectors, self._outer_recast, axis
            ).ok()  # type: ignore safe
        )

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if (
            "LINEAR_MAP" in self.outer_recast.features
            and "LINEAR_MAP" in self.inner.features
        ):
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
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


def recast_basis_from_diagonal[
    B0: Basis,
    B1: Basis,
    E,
    CT: Ctype[np.generic],
](
    basis: DiagonalBasis[TupleBasis[tuple[B0, B1], E], CT],
) -> RecastBasis[DiagonalBasis[TupleBasis[tuple[B0, B1], E], CT], B1, B1, CT]:
    """Recast a diagonal basis to the lhs basis."""
    return RecastBasis(basis, basis.inner.children[1], basis.inner.children[1])  # type: ignore it does support DT actually


def as_recast_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[np.generic],
](
    basis: TupleBasisLike[tuple[M0, M1], E, CT],
) -> (
    RecastBasis[
        DiagonalBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], CT],
        Basis[M1, CT],
        Basis[M1, CT],
        CT,
    ]
    | None
):
    """Get the basis as a recast diagonal basis."""
    basis_as_diagonal = as_diagonal(basis)
    if basis_as_diagonal is None:
        return None
    return recast_basis_from_diagonal(basis_as_diagonal)


@overload
def is_recast_basis[CT: Ctype[np.generic]](  # type: ignore is compatible
    basis: Basis[BasisMetadata, CT],
) -> TypeGuard[RecastBasis[Basis, Basis, Basis, CT]]: ...


@overload
def is_recast_basis(
    basis: object,
) -> TypeGuard[RecastBasis]: ...


def is_recast_basis(
    basis: object,
) -> TypeGuard[RecastBasis[Basis, Basis, Basis, Any]]:
    """Check if the basis is a recast basis."""
    return isinstance(basis, RecastBasis)
