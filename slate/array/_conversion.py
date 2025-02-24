from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, overload

import numpy as np

from slate import basis
from slate.array._array import Array, ArrayBuilder
from slate.basis import (
    Basis,
    BasisFeature,
    BasisStateMetadata,
    DiagonalBasis,
    FundamentalBasis,
    RecastBasis,
    TupleBasis,
)
from slate.basis._basis import ctype
from slate.metadata import BasisMetadata
from slate.metadata.stacked import AnyMetadata

if TYPE_CHECKING:
    from slate.basis._tuple import TupleBasisMetadata
    from slate.metadata import TupleMetadata


def cast_basis[B: Basis[Any, Any], DT: np.dtype[np.generic]](
    array: Array[Any, DT], basis: B
) -> ArrayBuilder[B, DT]:
    assert array.basis.size == basis.size
    return ArrayBuilder(basis, array.raw_data)


def as_feature_basis[
    M: BasisMetadata,
    DT: np.generic,
    DT1: ctype[np.generic],
](
    array: Array[Basis[M, DT1], np.dtype[DT]], features: set[BasisFeature]
) -> Array[Basis[M, DT1], np.dtype[DT]]:
    return array.with_basis(basis.as_feature_basis(array.basis, features)).ok()


def as_index_basis[M: BasisMetadata, DT: np.generic, DT1: ctype[np.generic]](
    array: Array[Basis[M, DT1], np.dtype[DT]],
) -> Array[Basis[M, DT1], np.dtype[DT]]:
    return as_feature_basis(array, {"INDEX"})


def as_mul_basis[M: BasisMetadata, DT: np.generic, DT1: ctype[np.generic]](
    array: Array[Basis[M, DT1], np.dtype[DT]],
) -> Array[Basis[M, DT1], np.dtype[DT]]:
    return as_feature_basis(array, {"MUL"})


def as_sub_basis[M: BasisMetadata, DT: np.generic, DT1: ctype[np.generic]](
    array: Array[Basis[M, DT1], np.dtype[DT]],
) -> Array[Basis[M, DT1], np.dtype[DT]]:
    return as_feature_basis(array, {"SUB"})


def as_add_basis[M: BasisMetadata, DT: np.generic, DT1: ctype[np.generic]](
    array: Array[Basis[M, DT1], np.dtype[DT]],
) -> Array[Basis[M, DT1], np.dtype[DT]]:
    return as_feature_basis(array, {"ADD"})


def as_linear_map_basis[M: BasisMetadata, DT: np.generic, DT1: ctype[np.generic]](
    array: Array[Basis[M, DT1], np.dtype[DT]],
) -> Array[Basis[M, DT1], np.dtype[DT]]:
    return as_feature_basis(array, {"LINEAR_MAP"})


def as_diagonal_basis[
    B0: Basis,
    B1: Basis,
    E,
    DT: np.dtype[np.generic],
    DT1: ctype[Never],
](
    array: Array[Basis[TupleBasisMetadata[tuple[B0, B1], E], DT1], DT],
) -> Array[DiagonalBasis[tuple[B0, B1], E, DT1], DT] | None:
    b = basis.as_diagonal_basis(array.basis)
    if b is None:
        return None
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return array.with_basis(b).ok()  # type: ignore see above


def as_tuple_basis[
    C: tuple[Basis[BasisMetadata, ctype[Never]], ...],
    E,
    DT: ctype[Never],
    DT1: np.dtype[np.generic],
](
    array: Array[Basis[TupleBasisMetadata[C, E], DT], DT1],
) -> Array[TupleBasis[C, E, DT], DT1] | None:
    b = basis.as_tuple_basis(array.basis)
    if b is None:
        return None
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return array.with_basis(b).ok()  # type: ignore see above


def as_fundamental_basis[M: AnyMetadata, DT: np.generic](
    array: Array[Basis[M, ctype[Never]], np.dtype[DT]],
) -> Array[Basis[M, ctype[np.generic]], np.dtype[DT]]:
    return array.with_basis(basis.as_fundamental(array.basis)).ok()


def nest[
    B: Basis[Any, ctype[Never]],
    DT: np.dtype[np.generic],
](
    array: Array[B, DT],
) -> Array[TupleBasis[tuple[B], None, ctype[Never]], DT]:
    # Since the basis supports the same dtype as the original basis
    # it is safe to call ok on the builder.
    # Unfortunately is not possible to express this invariant in the type system.
    return cast_basis(array, TupleBasis((array.basis,))).ok()  # type: ignore see above


@overload
def flatten[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: Array[Metadata1D[M, Any], DT, TupleBasis1D[DT, B, Any]],
) -> Array[M, DT, B]: ...


@overload
def flatten[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    B: Basis[Any, Any] = Basis[M, DT],
](
    array: Array[Metadata1D[M, Any], DT],
) -> Array[M, DT, B]: ...


@overload
def flatten[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: Array[
        TupleMetadata[TupleMetadata[M, Any], Any],
        DT,
    ],
) -> Array[TupleMetadata[M, None], DT]: ...


def flatten[DT: np.dtype[np.generic]](
    array: Array[
        TupleMetadata[TupleMetadata[BasisMetadata, Any], Any],
        DT,
    ],
) -> Array[Any, DT, Any]:
    basis_as_tuple = basis.as_tuple_basis(array.basis)
    if len(basis_as_tuple.children) == 1:
        converted = array.with_basis(basis_as_tuple)
    else:
        final_basis = TupleBasis(
            tuple(basis.as_tuple_basis(c) for c in basis_as_tuple.children),
            array.basis.metadata().extra,
        )
        converted = array.with_basis(final_basis)
    return cast_basis(converted, basis.flatten(array.basis))


def as_outer_array[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    array: Array[Any, DT, RecastBasis[Any, M, DT, Any, BOuter]],
) -> Array[M, DT, BOuter]:
    return cast_basis(array, array.basis.outer_recast)


def as_diagonal_array[M: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[
        Metadata2D[M, M, E], DT, DiagonalBasis[DT, Basis[M, DT], Basis[M, DT], E]
    ],
) -> Array[M, DT, Basis[M, Any]]:
    return cast_basis(array, array.basis.inner[1])


def as_raw_array[DT: np.dtype[np.generic], B: Basis[Any, Any]](
    array: Array[Any, DT, B],
) -> Array[BasisStateMetadata[B], DT, FundamentalBasis[BasisStateMetadata[B]]]:
    return cast_basis(array, FundamentalBasis(BasisStateMetadata(array.basis)))
