from __future__ import annotations

import numpy as np

from slate.array._array import Array
from slate.array._conversion import as_index_basis
from slate.metadata._metadata import BasisMetadata


def real[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.real(converted.raw_data)).with_basis(array.basis)


def imag[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.imag(converted.raw_data)).with_basis(array.basis)


def angle[M: BasisMetadata, DT: np.complexfloating](
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.angle(converted.raw_data)).with_basis(array.basis)


def abs[M: BasisMetadata, DT: np.generic](  # noqa: A001
    array: Array[M, DT],
) -> Array[M, np.floating]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.abs(converted.raw_data)).with_basis(array.basis)
