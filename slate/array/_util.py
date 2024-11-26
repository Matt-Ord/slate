from __future__ import annotations

import numpy as np

from slate.array._array import SlateArray
from slate.basis import as_index_basis
from slate.metadata import BasisMetadata


def conjugate[M: BasisMetadata, DT: np.generic](
    array: SlateArray[M, DT],
) -> SlateArray[M, DT]:
    """Conjugate a slate array."""
    converted = array.with_basis(as_index_basis(array.basis))
    return SlateArray(converted.basis, np.conj(converted.raw_data)).with_basis(
        array.basis
    )
