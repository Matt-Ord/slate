from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.cropped import CroppedBasis
from slate.basis.stacked_basis import (
    TupleBasis,
    tuple_basis_with_child,
)
from slate.basis.transformed import TransformedBasis
from slate.basis.truncated import TruncatedBasis, Truncation
from slate.linalg._eig import eig

if TYPE_CHECKING:
    from slate.basis._basis import Basis


def _test_einsum_in_basis(
    array: SlateArray[Any, TupleBasis[Any, None, Any]],
    vector: SlateArray[Any, TupleBasis[Any, None, Any]],
    basis: Basis[Any, Any],
) -> None:
    transformed_array = convert_array(
        array,
        tuple_basis_with_child(array.basis, basis.conjugate_basis(), 1),
    )
    transformed_vector = convert_array(
        vector,
        tuple_basis_with_child(vector.basis, basis, 0),
    )

    np.testing.assert_allclose(
        np.einsum(  # type: ignore libary
            "ij,j->i",
            array.raw_data.reshape(array.basis.shape),
            vector.raw_data.reshape(vector.basis.shape),
        ),
        np.einsum(  # type: ignore libary
            "ij,j->i",
            transformed_array.raw_data.reshape(array.basis.shape),
            transformed_vector.raw_data.reshape(vector.basis.shape),
        ),
    )


def test_einsum() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray.from_array(data)

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = SlateArray.from_array(data)

    _test_einsum_in_basis(array, vector, TransformedBasis(vector.basis[0]))
    _test_einsum_in_basis(
        array, vector, CroppedBasis(vector.basis.size, vector.basis[0])
    )
    _test_einsum_in_basis(
        array,
        vector,
        TruncatedBasis(Truncation(vector.basis[0].size, 1, 0), vector.basis[0]),
    )
    _test_einsum_in_basis(
        array,
        vector,
        TruncatedBasis(Truncation(vector.basis[0].size, 1, 10), vector.basis[0]),
    )


def test_einsum_diagonal() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray.from_array(data)

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = SlateArray.from_array(data)

    diagonal_array = eig(array)
    _test_einsum_in_basis(array, vector, diagonal_array.basis.inner[0])
