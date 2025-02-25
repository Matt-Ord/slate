from __future__ import annotations

from typing import Any, Never, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature, ctype
from slate.basis._diagonal import DiagonalBasis, as_diagonal_basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata


class RecastBasis[
    BInner: Basis,
    BOuter: Basis,
    DT: ctype[Never] = ctype[Never],
](WrappedBasis[BInner, DT]):
    """Represents a truncated basis."""

    def __init__[
        BInner_: Basis,
        BOuter_: Basis,
    ](
        self: RecastBasis[BInner_, BOuter_],
        inner: BInner_,
        inner_recast: Basis[M1_, DT_],
        outer_recast: BOuter_,
    ) -> None:
        self._inner_recast: Basis[M1, DT] = inner_recast
        self._outer_recast: BOuter = outer_recast
        super().__init__(inner)

        assert self._inner_recast.size == self.inner.size

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RecastBasis)
            and self.inner == other.inner  # type: ignore unknown
            and self.inner_recast == other.inner_recast  # type: ignore unknown
            and self.outer_recast == other.outer_recast  # type: ignore unknown
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
    def inner_recast(self) -> Basis[M1, DT]:
        """The basis the inner was recast to."""
        return self._inner_recast

    @property
    def outer_recast(self) -> BOuter:
        """The basis the inner recast was transformed to."""
        return self._outer_recast

    @override
    def __into_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        return self._outer_recast.__convert_vector_into__(
            vectors, self._inner_recast, axis
        )

    @override
    def __from_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        return self._inner_recast.__convert_vector_into__(
            vectors, self._outer_recast, axis
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


def recast_basis_from_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    basis: DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E],
) -> RecastBasis[
    Metadata2D[M0, M1, E], M1, DT, DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E]
]:
    """Recast a diagonal basis to the lhs basis."""
    return RecastBasis(basis, basis.inner[1], basis.inner[1])


def as_recast_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    basis: Basis[Metadata2D[M0, M1, E], DT],
) -> (
    RecastBasis[
        Metadata2D[M0, M1, E],
        M1,
        DT,
        DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E],
    ]
    | None
):
    """Get the basis as a recast diagonal basis."""
    basis_as_diagonal = as_diagonal_basis(basis)
    if basis_as_diagonal is None:
        return None
    return recast_basis_from_diagonal(basis_as_diagonal)


type RecastDiagonalBasis[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
] = RecastBasis[
    Metadata2D[M, M, None], M, DT, DiagonalBasis[Any, BInner, BInner, None], BOuter
]


def recast_diagonal_basis[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    inner_basis: BInner, outer_basis: BOuter
) -> RecastDiagonalBasis[Any, Any, BInner, BOuter]:
    """Recast a basis which is diagonal in the inner basis."""
    return cast(
        "RecastDiagonalBasis[M, DT, BInner, BOuter]",
        RecastBasis(
            DiagonalBasis(inner_basis.dual_basis(), inner_basis)),
            inner_basis,
            outer_basis,
        ),
    )
