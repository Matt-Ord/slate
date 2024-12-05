from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np

from slate import basis
from slate.array._array import Array
from slate.basis import (
    Basis,
    BasisFeature,
    BasisStateMetadata,
    DiagonalBasis,
    FundamentalBasis,
    RecastBasis,
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    tuple_basis,
)
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata import Metadata1D, Metadata2D, Metadata3D, StackedMetadata


def with_basis[
    M: BasisMetadata,
    DT: np.generic,
    B1: Basis[Any, Any],
](
    array: Array[M, DT],
    basis: B1,
) -> Array[M, DT, B1]:
    """Convert the array to the given basis."""
    return array.with_basis(basis)


def cast_basis[M: BasisMetadata, DT: np.generic, B: Basis[Any, Any]](
    array: Array[M, DT], basis: B
) -> Array[Any, DT, B]:
    assert array.basis.size == basis.size
    return Array(basis, array.raw_data)


def as_feature_basis[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT], features: set[BasisFeature]
) -> Array[M, DT]:
    return array.with_basis(basis.as_feature_basis(array.basis, features))


def as_index_basis[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, DT]:
    return array.with_basis(basis.as_index_basis(array.basis))


def as_mul_basis[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, DT]:
    return array.with_basis(basis.as_mul_basis(array.basis))


def as_sub_basis[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, DT]:
    return array.with_basis(basis.as_sub_basis(array.basis))


def as_add_basis[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, DT]:
    return array.with_basis(basis.as_add_basis(array.basis))


def as_diagonal_basis[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M0, M1, E], DT],
) -> (
    Array[
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


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    E,
    DT: np.generic,
](
    array: Array[Any, DT, Basis[Metadata1D[M0, E], DT]],
) -> Array[
    Metadata1D[M0, E],
    DT,
    TupleBasis1D[np.generic, Basis[M0, Any], E],
]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.generic,
](
    array: Array[Any, DT, Basis[Metadata2D[M0, M1, E], DT]],
) -> Array[
    Metadata2D[M0, M1, E],
    DT,
    TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], E],
]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    array: Array[Any, DT, Basis[Metadata3D[M0, M1, M2, E], DT]],
) -> Array[
    Metadata3D[M0, M1, M2, E],
    DT,
    TupleBasis3D[np.generic, Basis[M0, Any], Basis[M1, Any], Basis[M2, Any], E],
]: ...


@overload
def as_tuple_basis[M: BasisMetadata, E, DT: np.generic](
    array: Array[StackedMetadata[M, E], DT],
) -> Array[StackedMetadata[M, E], DT, TupleBasis[M, E, Any, StackedMetadata[M, E]]]: ...


def as_tuple_basis[M: BasisMetadata, E, DT: np.generic](
    array: Array[StackedMetadata[M, E], DT],
) -> Array[StackedMetadata[M, E], DT, TupleBasis[M, E, Any, StackedMetadata[M, E]]]:
    return array.with_basis(basis.as_tuple_basis(array.basis))


def nest[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: Array[M, DT, B],
) -> Array[Metadata1D[M, None], DT, TupleBasis1D[DT, B, None]]:
    return cast_basis(array, tuple_basis((array.basis,)))


@overload
def flatten[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: Array[Metadata1D[M, Any], DT, TupleBasis1D[DT, B, Any]],
) -> Array[M, DT, B]: ...


@overload
def flatten[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: Array[Metadata1D[M, Any], DT],
) -> Array[M, DT, B]: ...


@overload
def flatten[M: BasisMetadata, DT: np.generic](
    array: Array[
        StackedMetadata[StackedMetadata[M, Any], Any],
        DT,
    ],
) -> Array[StackedMetadata[M, None], DT]: ...


def flatten[DT: np.generic](
    array: Array[
        StackedMetadata[StackedMetadata[BasisMetadata, Any], Any],
        DT,
    ],
) -> Array[Any, DT, Any]:
    basis_as_tuple = basis.as_tuple_basis(array.basis)
    if len(basis_as_tuple.children) == 1:
        converted = array.with_basis(basis_as_tuple)
    else:
        final_basis = tuple_basis(
            tuple(basis.as_tuple_basis(c) for c in basis_as_tuple.children),
            array.basis.metadata().extra,
        )
        converted = array.with_basis(final_basis)
    return cast_basis(converted, basis.flatten(array.basis))


def as_outer_array[
    M: BasisMetadata,
    DT: np.generic,
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    array: Array[Any, DT, RecastBasis[Any, M, DT, Any, BOuter]],
) -> Array[M, DT, BOuter]:
    return cast_basis(array, array.basis.outer_recast)


def as_diagonal_array[M: BasisMetadata, E, DT: np.generic](
    array: Array[
        Metadata2D[M, M, E], DT, DiagonalBasis[DT, Basis[M, DT], Basis[M, DT], E]
    ],
) -> Array[M, DT, Basis[M, Any]]:
    return cast_basis(array, array.basis.inner[1])


def as_raw_array[DT: np.generic, B: Basis[Any, Any]](
    array: Array[Any, DT, B],
) -> Array[BasisStateMetadata[B], DT, FundamentalBasis[BasisStateMetadata[B]]]:
    return cast_basis(array, FundamentalBasis(BasisStateMetadata(array.basis)))
