from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate import basis
from slate.array._array import Array
from slate.basis._diagonal import DiagonalBasis
from slate.metadata._metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.metadata.stacked import Metadata2D


def extract_diagonal[M: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M, M, E], DT],
) -> Array[M, DT, Basis[M, Any]] | None:
    b = DiagonalBasis(basis.from_metadata(array.basis.metadata()))
    converted = array.with_basis(b)

    return Array(converted.basis.inner[1], converted.raw_data)
