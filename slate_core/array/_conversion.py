from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, overload

import numpy as np

from slate_core import basis
from slate_core.array._array import build
from slate_core.basis import (
    Basis,
    BasisFeature,
    BasisStateMetadata,
    Ctype,
    DiagonalBasis,
    FundamentalBasis,
    RecastBasis,
    TupleBasis,
    TupleBasisLike,
    is_tuple_basis_like,
)
from slate_core.metadata import AnyMetadata, BasisMetadata

if TYPE_CHECKING:
    from slate_core.array._array import Array, ArrayBuilder
    from slate_core.metadata import TupleMetadata


def cast_basis[B: Basis, DT: np.dtype[np.generic]](
    array: Array[Any, DT], basis: B
) -> ArrayBuilder[B, DT]:
    assert array.basis.size == basis.size
    return build(basis, array.raw_data)


def as_feature_basis[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    DT1: Ctype[Never],
](
    array: Array[Basis[M, DT1], DT], features: set[BasisFeature]
) -> Array[Basis[M, DT1], DT]:
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return array.with_basis(basis.as_feature_basis(array.basis, features)).ok()  # type: ignore see above


def as_index_basis[M: BasisMetadata, DT: np.dtype[np.generic], DT1: Ctype[Never]](
    array: Array[Basis[M, DT1], DT],
) -> Array[Basis[M, DT1], DT]:
    return as_feature_basis(array, {"INDEX"})


def as_mul_basis[M: BasisMetadata, DT: np.dtype[np.generic], DT1: Ctype[Never]](
    array: Array[Basis[M, DT1], DT],
) -> Array[Basis[M, DT1], DT]:
    return as_feature_basis(array, {"MUL"})


def as_sub_basis[M: BasisMetadata, DT: np.dtype[np.generic], DT1: Ctype[Never]](
    array: Array[Basis[M, DT1], DT],
) -> Array[Basis[M, DT1], DT]:
    return as_feature_basis(array, {"SUB"})


def as_add_basis[M: BasisMetadata, DT: np.dtype[np.generic], DT1: Ctype[Never]](
    array: Array[Basis[M, DT1], DT],
) -> Array[Basis[M, DT1], DT]:
    return as_feature_basis(array, {"ADD"})


def as_linear_map_basis[M: BasisMetadata, DT: np.dtype[np.generic], DT1: Ctype[Never]](
    array: Array[Basis[M, DT1], DT],
) -> Array[Basis[M, DT1], DT]:
    return as_feature_basis(array, {"LINEAR_MAP"})


def as_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
    DT1: Ctype[Never],
](
    array: Array[Basis[TupleMetadata[tuple[M0, M1], E], DT1], DT],
) -> (
    Array[DiagonalBasis[TupleBasis[tuple[Basis[M0, DT1], Basis[M1, DT1]], E], DT1], DT]
    | None
):
    b = basis.as_diagonal_basis(array.basis)
    if b is None:
        return None
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return array.with_basis(b).ok()  # type: ignore see above


@overload
def as_tuple_basis[M0: BasisMetadata, E, CT: Ctype[Never], DT1: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[M0], E, CT], DT1],
) -> Array[TupleBasis[tuple[Basis[M0, CT]], E, CT], DT1]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
    DT1: np.dtype[np.generic],
](
    array: Array[TupleBasisLike[tuple[M0, M1], E, CT], DT1],
) -> Array[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E, CT], DT1]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    CT: Ctype[Never],
    DT1: np.dtype[np.generic],
](
    array: Array[TupleBasisLike[tuple[M0, M1, M2], E, CT], DT1],
) -> Array[
    TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT], Basis[M2, CT]], E, CT], DT1
]: ...


@overload
def as_tuple_basis[M: BasisMetadata, E, CT: Ctype[Never], DT1: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[M, ...], E, CT], DT1],
) -> Array[TupleBasis[tuple[Basis[M, CT], ...], E, CT], DT1]: ...


def as_tuple_basis[M: BasisMetadata, E, CT: Ctype[Never], DT1: np.dtype[np.generic]](
    array: Array[Basis[TupleMetadata[tuple[M, ...], E], CT], DT1],
) -> Array[TupleBasis[tuple[Basis[M, CT], ...], E, CT], DT1]:
    b = basis.as_tuple_basis(array.basis)
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return array.with_basis(b).ok()  # type: ignore see above


def as_fundamental_basis[M: AnyMetadata, DT: np.dtype[np.generic]](
    array: Array[Basis[M, Ctype[Never]], DT],
) -> Array[Basis[M, Ctype[np.generic]], DT]:
    return array.with_basis(basis.as_fundamental(array.basis)).ok()


def as_transformed_basis[M: AnyMetadata, DT: np.dtype[np.complexfloating]](
    array: Array[Basis[M, Ctype[Never]], DT],
) -> Array[Basis[M, Ctype[np.complexfloating]], DT]:
    return array.with_basis(basis.as_transformed(array.basis)).ok()


def as_outer_array[B: Basis, DT: np.dtype[np.generic]](
    array: Array[RecastBasis[Basis, B, Basis, Ctype[Never]], DT],
) -> Array[B, DT]:
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return cast_basis(array, array.basis.outer_recast).ok()  # type: ignore safe, since outer_recast must support DT


def as_diagonal_array[B: Basis, DT: np.dtype[np.generic]](
    array: Array[DiagonalBasis[TupleBasis[tuple[Basis, B], Never], Ctype[Never]], DT],
) -> Array[B, DT]:
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return cast_basis(array, array.basis.inner[1]).ok()  # type: ignore safe, since inner[1] must support DT


def nest[
    B: Basis[Any, Ctype[Never]],
    DT: np.dtype[np.generic],
](
    array: Array[B, DT],
) -> Array[TupleBasis[tuple[B], None, Ctype[Never]], DT]:
    # Since the basis supports the same dtype as the original basis
    # it is safe to call ok on the builder.
    # Unfortunately is not possible to express this invariant in the type system.
    return cast_basis(array, TupleBasis((array.basis,))).ok()  # type: ignore see above


@overload
def flatten[B: Basis, DT: np.dtype[np.generic]](
    array: Array[TupleBasis[tuple[B], Never], DT],
) -> Array[B, DT]: ...


@overload
def flatten[M: BasisMetadata, DT: np.dtype[np.generic], DT1: Ctype[Never]](
    array: Array[TupleBasisLike[tuple[M], Never, DT1], DT],
) -> Array[Basis[M, DT1], DT]: ...


@overload
def flatten[M: BasisMetadata, E, DT1: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[TupleMetadata[tuple[M, ...]], ...], E, DT1], DT],
) -> Array[TupleBasis[tuple[Basis[M], ...], E, DT1], DT]: ...


def flatten[DT: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[BasisMetadata, ...], Any], DT],
) -> Array[Any, DT]:
    basis_as_tuple = basis.as_tuple_basis(array.basis).resolve_ctype()
    if len(basis_as_tuple.children) == 1:
        converted = as_tuple_basis(array)
        # Since the basis supports the same dtype as the original basis
        # it is safe to call ok on the builder.
        # Unfortunately is not possible to express this invariant in the type system.
        return cast_basis(converted, basis_as_tuple.children[0]).ok()  # type: ignore see above
    children = tuple(
        basis.as_tuple_basis(c) if is_tuple_basis_like(c) else c
        for c in basis_as_tuple.children
    )
    final_basis = TupleBasis(children, array.basis.metadata().extra)
    converted = array.with_basis(final_basis)
    return cast_basis(converted, basis.flatten(basis_as_tuple)).ok()  # type: ignore see above


def as_raw_array[DT: np.dtype[np.generic], B: Basis](
    array: Array[B, DT],
) -> Array[FundamentalBasis[BasisStateMetadata[B]], DT]:
    return cast_basis(array, FundamentalBasis(BasisStateMetadata(array.basis))).ok()
