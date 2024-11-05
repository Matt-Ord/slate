from __future__ import annotations

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.stacked._tuple_basis import (
    tuple_basis_with_child,
    tuple_basis_with_modified_child,
)
from slate.basis.transformed import TransformedBasis


def test_einsum_transformed() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray.from_array(data)

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = SlateArray.from_array(data)

    transformed_array = convert_array(
        array,
        tuple_basis_with_modified_child(array.basis, TransformedBasis, 1),
    )
    transformed_vector = convert_array(
        vector,
        tuple_basis_with_child(
            vector.basis, transformed_array.basis[1].conjugate_basis(), 0
        ),
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
