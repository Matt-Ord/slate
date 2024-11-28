from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array._array import SlateArray
from slate.basis import as_index_basis
from slate.basis._basis import Basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis.recast import RecastBasis


def conjugate[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    """Conjugate a slate array."""
    converted = array.with_basis(as_index_basis(array.basis))
    return SlateArray(converted.basis, np.conj(converted.raw_data)).with_basis(
        array.basis
    )


def array_as_outer[
    M: BasisMetadata,
    DT: np.generic,
    BOuter: Basis[BasisMetadata, Any] = Basis[M, DT],
](
    array: SlateArray[Any, DT, RecastBasis[Any, M, DT, Any, BOuter]],
) -> SlateArray[M, DT, BOuter]:
    return SlateArray(array.basis.outer_recast, array.raw_data)
