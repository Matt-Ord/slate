from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate import basis
from slate.array._array import SlateArray
from slate.basis import Basis, BasisFeature, DiagonalBasis, TupleBasis, tuple_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis.recast import RecastBasis
    from slate.metadata.stacked import Metadata2D, StackedMetadata


def with_basis[
    M: BasisMetadata,
    DT: np.generic,
    B1: Basis[Any, Any],
](
    array: SlateArray[M, DT],
    basis: B1,
) -> SlateArray[M, DT, B1]:
    """Convert the array to the given basis."""
    return array.with_basis(basis)


def as_feature_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT], features: set[BasisFeature]
) -> SlateArray[M, DT]:
    return array.with_basis(basis.as_feature_basis(array.basis, features))


def as_index_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    return array.with_basis(basis.as_index_basis(array.basis))


def as_mul_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    return array.with_basis(basis.as_mul_basis(array.basis))


def as_sub_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    return array.with_basis(basis.as_sub_basis(array.basis))


def as_add_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    return array.with_basis(basis.as_add_basis(array.basis))


def as_diagonal_basis[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: SlateArray[Metadata2D[M0, M1, E], DT],
) -> (
    SlateArray[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[Any, Basis[M0, Any], Basis[M1, Any], E],
    ]
    | None
):
    b = basis.as_diagonal_basis(array.basis)
    if b is None:
        return None

    return array.with_basis(b)


def as_tuple_basis[M: BasisMetadata, E, DT: np.generic](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[
    StackedMetadata[M, E], DT, TupleBasis[M, E, Any, StackedMetadata[M, E]]
]:
    return array.with_basis(basis.as_tuple_basis(array.basis))


def as_flatten_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[
        StackedMetadata[StackedMetadata[M, Any], Any],
        DT,
    ],
) -> SlateArray[StackedMetadata[M, None], DT, TupleBasis[M, None, DT]]:
    basis_flat = basis.flatten_basis(array.basis)
    children = basis.as_tuple_basis(array.basis).children
    final_basis = tuple_basis(
        tuple(basis.as_tuple_basis(c) for c in children),
        array.basis.metadata().extra,
    )
    converted = array.with_basis(final_basis)
    return SlateArray(basis_flat, converted.raw_data)


def as_outer_basis[
    M: BasisMetadata,
    DT: np.generic,
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    array: SlateArray[Any, DT, RecastBasis[Any, M, DT, Any, BOuter]],
) -> SlateArray[M, DT, BOuter]:
    return SlateArray(array.basis.outer_recast, array.raw_data)
