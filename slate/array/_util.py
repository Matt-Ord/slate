from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate import basis
from slate.array._array import SlateArray
from slate.basis import (
    Basis,
    TupleBasis,
    tuple_basis,
)
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis.recast import RecastBasis
    from slate.metadata.stacked import StackedMetadata


def conjugate[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    """Conjugate a slate array."""
    converted = array.with_basis(basis.as_index_basis(array.basis))
    return SlateArray(converted.basis, np.conj(converted.raw_data)).with_basis(
        array.basis
    )


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
