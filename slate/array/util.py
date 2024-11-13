from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.stacked._tuple_basis import TupleBasis, as_tuple_basis, tuple_basis
from slate.metadata._metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.metadata.stacked.stacked import StackedMetadata


def _transpose_from_tuple[DT: np.generic, M: BasisMetadata, E](
    array: SlateArray[DT, TupleBasis[M, E, Any]],
) -> SlateArray[DT, TupleBasis[M, E, Any]]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    return SlateArray[DT, TupleBasis[M, E, Any]](
        tuple_basis((array.basis[1], array.basis[0]), array.basis.metadata.extra),
        array.raw_data.reshape(array.basis.shape).transpose(),
    )


def transpose[M: BasisMetadata, E, DT: np.generic](
    array: SlateArray[DT, Basis[StackedMetadata[M, E], Any]],
) -> SlateArray[DT, Basis[StackedMetadata[M, E], Any]]:
    """Transpose a slate array."""
    tuple_basis = as_tuple_basis(array.basis)
    return _transpose_from_tuple(convert_array(array, tuple_basis))
