from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from slate import basis
from slate.array._array import SlateArray
from slate.basis import Basis, BasisFeature, DiagonalBasis, TupleBasis, tuple_basis
from slate.basis._basis_state_metadata import BasisStateMetadata
from slate.basis._fundamental import FundamentalBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis._tuple import TupleBasis1D
    from slate.basis.recast import RecastBasis
    from slate.metadata.stacked import Metadata1D, Metadata2D, StackedMetadata


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


def nest[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: SlateArray[M, DT, B],
) -> SlateArray[Metadata1D[M, None], DT, TupleBasis1D[DT, B, None]]:
    return SlateArray(tuple_basis((array.basis,)), array.raw_data)


@overload
def flatten[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: SlateArray[Metadata1D[M, Any], DT, TupleBasis1D[DT, B, Any]],
) -> SlateArray[M, DT, B]: ...


@overload
def flatten[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: SlateArray[Metadata1D[M, Any], DT],
) -> SlateArray[M, DT, B]: ...


@overload
def flatten[M: BasisMetadata, DT: np.generic](
    array: SlateArray[
        StackedMetadata[StackedMetadata[M, Any], Any],
        DT,
    ],
) -> SlateArray[StackedMetadata[M, None], DT]: ...


def flatten[DT: np.generic](
    array: SlateArray[
        StackedMetadata[StackedMetadata[BasisMetadata, Any], Any],
        DT,
    ],
) -> SlateArray[Any, DT, Any]:
    basis_as_tuple = basis.as_tuple_basis(array.basis)
    if len(basis_as_tuple.children) == 1:
        converted = array.with_basis(basis_as_tuple)
    else:
        final_basis = tuple_basis(
            tuple(basis.as_tuple_basis(c) for c in basis_as_tuple.children),
            array.basis.metadata().extra,
        )
        converted = array.with_basis(final_basis)
    return SlateArray(basis.flatten(array.basis), converted.raw_data)


def as_outer_array[
    M: BasisMetadata,
    DT: np.generic,
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    array: SlateArray[Any, DT, RecastBasis[Any, M, DT, Any, BOuter]],
) -> SlateArray[M, DT, BOuter]:
    return SlateArray(array.basis.outer_recast, array.raw_data)


def as_diagonal_array[M: BasisMetadata, E, DT: np.generic](
    array: SlateArray[
        Metadata2D[M, M, E], DT, DiagonalBasis[DT, Basis[M, DT], Basis[M, DT], E]
    ],
) -> SlateArray[M, DT, Basis[M, Any]]:
    return SlateArray(array.basis.inner[1], array.raw_data)


def as_raw_array[DT: np.generic, B: Basis[Any, Any]](
    array: SlateArray[Any, DT, B],
) -> SlateArray[BasisStateMetadata[B], DT, FundamentalBasis[BasisStateMetadata[B]]]:
    return SlateArray(FundamentalBasis(BasisStateMetadata(array.basis)), array.raw_data)
