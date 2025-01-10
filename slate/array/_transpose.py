from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array._array import Array
from slate.array._conversion import as_diagonal_basis, as_index_basis, as_tuple_basis
from slate.basis import (
    Basis,
    TupleBasis2D,
    tuple_basis,
)
from slate.basis._diagonal import DiagonalBasis, diagonal_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata.stacked import Metadata2D


def conjugate[M: BasisMetadata, DT: np.generic](
    array: Array[M, DT],
) -> Array[M, DT]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    return Array(converted.basis, np.conj(converted.raw_data)).with_basis(array.basis)


def _transpose_from_diagonal[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    return Array(
        diagonal_basis(
            (array.basis.inner[1], array.basis.inner[0]), array.basis.metadata().extra
        ),
        array.raw_data,
    )


def _transpose_from_tuple[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    return Array(
        tuple_basis((array.basis[1], array.basis[0]), array.basis.metadata().extra),
        array.raw_data.reshape(array.basis.shape).transpose(),
    )


def transpose[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M2, M1, E], DT]:
    """Transpose a slate array."""
    as_diagonal = as_diagonal_basis(array)
    if as_diagonal is not None:
        return _transpose_from_diagonal(as_diagonal)

    return _transpose_from_tuple(as_tuple_basis(array))


def _inv_from_diagonal[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    return Array(
        diagonal_basis(
            (array.basis.inner[1].dual_basis(), array.basis.inner[0].dual_basis()),
            array.basis.metadata().extra,
        ),
        np.divide(1.0, array.raw_data),
    )


def _inv_from_tuple[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    raw_data = array.raw_data.reshape(array.basis.shape)
    return Array(
        tuple_basis(
            (array.basis[1].dual_basis(), array.basis[0].dual_basis()),
            array.basis.metadata().extra,
        ),
        np.linalg.inv(raw_data),  # type: ignore unknown
    )


def inv[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M2, M1, E], DT]:
    """Inverse a slate array."""
    as_diagonal = as_diagonal_basis(array)
    if as_diagonal is not None:
        return _inv_from_diagonal(as_diagonal)

    return _inv_from_tuple(as_tuple_basis(array))


def dagger[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M2, M1, E], DT]:
    """Conjugate Transpose a slate array."""
    return conjugate(transpose(array))
