from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array import SlateArray, convert_array
from slate.basis import (
    CroppedBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    TupleBasis,
    tuple_basis_with_child,
)
from slate.basis._tuple import tuple_basis
from slate.basis.fundamental import FundamentalBasis
from slate.linalg._eig import into_diagonal, into_diagonal_hermitian

if TYPE_CHECKING:
    from slate.basis import Basis


def _test_einsum_in_basis(
    array: SlateArray[Any, Any, TupleBasis[Any, None, Any]],
    vector: SlateArray[Any, Any, Any],
    basis: Basis[Any, Any],
) -> None:
    transformed_array = convert_array(
        array,
        tuple_basis_with_child(array.basis, basis.dual_basis(), 1),
    )
    transformed_vector = convert_array(
        vector,
        basis,
    )

    np.testing.assert_allclose(
        np.einsum(  # type: ignore libary
            "ij,j->i",
            array.raw_data.reshape(array.basis.shape),
            vector.raw_data.reshape(vector.basis.size),
        ),
        np.einsum(  # type: ignore libary
            "ij,j->i",
            transformed_array.raw_data.reshape(array.basis.shape),
            transformed_vector.raw_data.reshape(vector.basis.size),
        ),
        atol=1e-15,
    )


def test_einsum() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray(
        tuple_basis(
            (
                FundamentalBasis.from_size(10),
                FundamentalBasis.from_size(10).dual_basis(),
            )
        ),
        data,
    )

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = SlateArray(array.basis[0], data)

    _test_einsum_in_basis(array, vector, TransformedBasis(vector.basis))
    _test_einsum_in_basis(array, vector, CroppedBasis(vector.basis.size, vector.basis))
    _test_einsum_in_basis(
        array,
        vector,
        TruncatedBasis(Truncation(vector.basis.size, 1, 0), vector.basis),
    )
    _test_einsum_in_basis(
        array,
        vector,
        TruncatedBasis(Truncation(vector.basis.size, 1, 10), vector.basis),
    )


def test_einsum_diagonal() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray(
        tuple_basis(
            (
                FundamentalBasis.from_size(10),
                FundamentalBasis.from_size(10).dual_basis(),
            )
        ),
        data,
    )

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = SlateArray(array.basis[0], data)
    diagonal_array = into_diagonal(array)

    _test_einsum_in_basis(array, vector, diagonal_array.basis.inner[0])

    data = array.raw_data.reshape(array.basis.shape)
    data += np.conj(data.T)
    array = SlateArray(
        tuple_basis(
            (
                FundamentalBasis.from_size(10),
                FundamentalBasis.from_size(10).dual_basis(),
            )
        ),
        data,
    )
    diagonal_array = into_diagonal_hermitian(array)
    _test_einsum_in_basis(array, vector, diagonal_array.basis.inner[0])
