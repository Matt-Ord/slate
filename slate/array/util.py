from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.stacked import TupleBasis, as_tuple_basis, tuple_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata.stacked.stacked import StackedMetadata


def _transpose_from_tuple[DT: np.generic, M: BasisMetadata, E](
    array: SlateArray[StackedMetadata[M, E], DT, TupleBasis[M, E, np.generic]],
) -> SlateArray[StackedMetadata[M, E], DT]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    return SlateArray(
        tuple_basis((array.basis[1], array.basis[0]), array.basis.metadata.extra),
        array.raw_data.reshape(array.basis.shape).transpose(),
    )


def transpose[M: BasisMetadata, E, DT: np.generic](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[StackedMetadata[M, E], DT]:
    """Transpose a slate array."""
    tuple_basis = as_tuple_basis(array.basis)
    return _transpose_from_tuple(convert_array(array, tuple_basis))
