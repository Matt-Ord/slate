from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array._array import SlateArray
from slate.basis import (
    Basis,
    TupleBasis2D,
    as_diagonal_basis,
    as_tuple_basis,
    tuple_basis,
)
from slate.basis._diagonal import DiagonalBasis, diagonal_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata.stacked import Metadata2D


def _transpose_from_diagonal[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: SlateArray[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> SlateArray[Metadata2D[M0, M1, E], DT]:
    return SlateArray(
        diagonal_basis(
            (array.basis.inner[1], array.basis.inner[0]), array.basis.metadata().extra
        ),
        array.raw_data,
    )


def _transpose_from_tuple[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: SlateArray[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> SlateArray[Metadata2D[M0, M1, E], DT]:
    return SlateArray(
        tuple_basis((array.basis[1], array.basis[0]), array.basis.metadata().extra),
        array.raw_data.reshape(array.basis.shape).transpose(),
    )


def transpose[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.generic](
    array: SlateArray[Metadata2D[M1, M2, E], DT],
) -> SlateArray[Metadata2D[M1, M2, E], DT]:
    """Transpose a slate array."""
    basis_as_diagonal = as_diagonal_basis(array.basis)
    if basis_as_diagonal is not None:
        return _transpose_from_diagonal(array.with_basis(basis_as_diagonal))

    basis_as_tuple = as_tuple_basis(array.basis)
    return _transpose_from_tuple(array.with_basis(basis_as_tuple))


def _inv_from_diagonal[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: SlateArray[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> SlateArray[Metadata2D[M0, M1, E], DT]:
    return SlateArray(
        diagonal_basis(
            (array.basis.inner[1].dual_basis(), array.basis.inner[0].dual_basis()),
            array.basis.metadata().extra,
        ),
        np.divide(1.0, array.raw_data),
    )


def _inv_from_tuple[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.generic](
    array: SlateArray[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> SlateArray[Metadata2D[M0, M1, E], DT]:
    raw_data = array.raw_data.reshape(array.basis.shape)
    return SlateArray(
        tuple_basis(
            (array.basis[1].dual_basis(), array.basis[0].dual_basis()),
            array.basis.metadata().extra,
        ),
        np.linalg.inv(raw_data),  # type: ignore unknown
    )


def inv[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.generic](
    array: SlateArray[Metadata2D[M1, M2, E], DT],
) -> SlateArray[Metadata2D[M1, M2, E], DT]:
    """Inverse a slate array."""
    basis_as_diagonal = as_diagonal_basis(array.basis)
    if basis_as_diagonal is not None:
        return _inv_from_diagonal(array.with_basis(basis_as_diagonal))

    basis_as_tuple = as_tuple_basis(array.basis)
    return _inv_from_tuple(array.with_basis(basis_as_tuple))
