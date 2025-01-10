from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np

from slate.array import Array
from slate.basis import Basis, DiagonalBasis, diagonal_basis
from slate.basis._basis_state_metadata import BasisStateMetadata
from slate.basis._fundamental import FundamentalBasis
from slate.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata, SimpleMetadata

if TYPE_CHECKING:
    from slate.metadata import Metadata2D


class DiagonalExplicitBasis[M: BasisMetadata, DT: np.generic, B: Basis[Any, Any]](
    ExplicitUnitaryBasis[M, DT, B]
):
    def __init__[DT1: np.generic, B1: Basis[Any, Any]](
        self: DiagonalExplicitBasis[Any, DT1, B1],
        matrix: Array[
            Metadata2D[SimpleMetadata, BasisStateMetadata[B1], Any],
            DT1,
            DiagonalBasis[
                DT1,
                Basis[SimpleMetadata, Any],
                Basis[BasisStateMetadata[B1], Any],
                None,
            ],
        ],
    ) -> None:
        super().__init__(cast("Any", matrix), data_id=uuid.UUID(int=0))


class TrivialExplicitBasis[M: BasisMetadata, DT: np.generic, B: Basis[Any, Any]](
    ExplicitUnitaryBasis[M, DT, B]
):
    def __init__[B1: Basis[Any, Any]](
        self: TrivialExplicitBasis[Any, Any, B1], inner: B1
    ) -> None:
        super().__init__(
            Array(
                diagonal_basis(
                    (
                        FundamentalBasis(BasisStateMetadata(inner)),
                        FundamentalBasis(BasisStateMetadata(inner)),
                    )
                ),
                cast("np.ndarray[Any, np.dtype[DT]]", np.ones(inner.size)),
            ),
            data_id=uuid.UUID(int=0),
        )

    @override
    def __into_inner__[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        # TODO: use sparse-aware einsum to do this automatically  # noqa: FIX002
        return vectors

    @override
    def __from_inner__[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return vectors
