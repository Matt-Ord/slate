from __future__ import annotations

from typing import Any, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis._diagonal import DiagonalBasis, as_diagonal_basis, diagonal_basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.metadata.stacked import Metadata2D


class RecastBasis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
    BInner: Basis[BasisMetadata, Any] = Basis[M0, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M1, DT],
](WrappedBasis[M0, DT, BInner]):
    """Represents a truncated basis."""

    def __init__[
        _M1: BasisMetadata,
        _DT: np.generic,
        _BInner: Basis[BasisMetadata, Any],
        _BOuter: Basis[BasisMetadata, Any] = Basis[_M1, _DT],
    ](
        self: RecastBasis[Any, _M1, _DT, _BInner, _BOuter],
        inner: _BInner,
        inner_recast: Basis[_M1, _DT],
        outer_recast: _BOuter,
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
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return self._outer_recast.__convert_vector_into__(
            vectors, self._inner_recast, axis
        )

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return self._inner_recast.__convert_vector_into__(
            vectors, self._outer_recast, axis
        )

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "SIMPLE_ADD" in self.outer_recast.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.outer_recast.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.outer_recast.features:
            out.add("SUB")
            out.add("SIMPLE_SUB")
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


def recast_basis_from_diagonal[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    basis: DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E],
) -> RecastBasis[
    Metadata2D[M0, M1, E], M1, DT, DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E]
]:
    """Recast a diagonal basis to the lhs basis."""
    return RecastBasis(basis, basis.inner[1], basis.inner[1])


def as_recast_diagonal_basis[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
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
    DT: np.generic,
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
] = RecastBasis[
    Metadata2D[M, M, None], M, DT, DiagonalBasis[Any, BInner, BInner, None], BOuter
]


def recast_diagonal_basis[
    M: BasisMetadata,
    DT: np.generic,
    BInner: Basis[BasisMetadata, Any] = Basis[M, DT],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    inner_basis: BInner, outer_basis: BOuter
) -> RecastDiagonalBasis[Any, Any, BInner, BOuter]:
    """Recast a basis which is diagonal in the inner basis."""
    return cast(
        "RecastDiagonalBasis[M, DT, BInner, BOuter]",
        RecastBasis(
            diagonal_basis((inner_basis.dual_basis(), inner_basis)),
            inner_basis,
            outer_basis,
        ),
    )
