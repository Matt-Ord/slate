from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate import array as _array
from slate import basis
from slate.array import Array
from slate.basis import DiagonalBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis import Basis
    from slate.metadata import Metadata2D


def extract_diagonal[M: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M, M, E], DT],
) -> Array[M, DT, Basis[M, Any]] | None:
    b = DiagonalBasis(basis.from_metadata(array.basis.metadata()))
    converted = array.with_basis(b)

    return Array(converted.basis.inner[1], converted.raw_data)


def abs[M: BasisMetadata, DT: np.generic](  # noqa: A001
    array: Array[M, DT],
) -> Array[M, np.float64]:
    """Calculate the absolute value of the array."""
    converted = _array.as_index_basis(array)
    return Array(converted.basis, np.abs(converted.raw_data))
