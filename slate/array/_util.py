from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array._array import SlateArray
from slate.basis import as_index_basis
from slate.basis._basis import Basis
from slate.basis._tuple import TupleBasis, as_tuple_basis, flatten_basis, tuple_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis.recast import RecastBasis
    from slate.metadata.stacked import StackedMetadata


def conjugate[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    """Conjugate a slate array."""
    converted = array.with_basis(as_index_basis(array.basis))
    return SlateArray(converted.basis, np.conj(converted.raw_data)).with_basis(
        array.basis
    )


def array_as_tuple_basis[M: BasisMetadata, E, DT: np.generic](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[
    StackedMetadata[M, E], DT, TupleBasis[M, E, Any, StackedMetadata[M, E]]
]:
    basis = as_tuple_basis(array.basis)
    return array.with_basis(basis)


def array_as_flatten_basis[M: BasisMetadata, DT: np.generic](
    array: SlateArray[
        StackedMetadata[StackedMetadata[M, Any], Any],
        DT,
    ],
) -> SlateArray[StackedMetadata[M, None], DT, TupleBasis[M, None, DT]]:
    basis = flatten_basis(array.basis)
    final_basis = tuple_basis(
        tuple(as_tuple_basis(c) for c in as_tuple_basis(array.basis).children),
        array.basis.metadata().extra,
    )
    converted = array.with_basis(final_basis)
    return SlateArray(basis, converted.raw_data)


def array_as_outer_basis[
    M: BasisMetadata,
    DT: np.generic,
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    array: SlateArray[Any, DT, RecastBasis[Any, M, DT, Any, BOuter]],
) -> SlateArray[M, DT, BOuter]:
    return SlateArray(array.basis.outer_recast, array.raw_data)
