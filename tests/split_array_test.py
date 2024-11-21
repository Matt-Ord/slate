from __future__ import annotations

import numpy as np

from slate.array import SlateArray
from slate.basis import (
    FundamentalBasis,
    TransformedBasis,
    diagonal_basis,
    split_basis,
    tuple_basis,
)


def test_split_array_equals_diagonal() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    array = SlateArray.from_array(data)

    diagonal = array.with_basis(diagonal_basis((array.basis[0], array.basis[1])))
    split = array.with_basis(split_basis((array.basis[0], array.basis[1])))

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        split.raw_data, [1.0, 2.0, 3.0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())


def test_split_array_equals_transformed() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    basis_k = TransformedBasis(FundamentalBasis.from_shape((3,)))
    array = SlateArray(tuple_basis((basis_k, basis_k.conjugate_basis())), data)

    diagonal = array.with_basis(diagonal_basis((array.basis[0], array.basis[1])))
    split = SlateArray(
        split_basis(
            (FundamentalBasis.from_shape((3,)), FundamentalBasis.from_shape((3,)))
        ),
        np.array([0, 0, 0, 1, 1, 1, 1.0, 2.0, 3.0, 1, 1, 1]),
    )

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        split.raw_data, [0, 0, 0, 1, 1, 1, 1.0, 2.0, 3.0, 1, 1, 1]
    )
    np.testing.assert_allclose(
        split.with_basis(diagonal.basis).raw_data,
        diagonal.raw_data,
    )
    np.testing.assert_allclose(
        split.with_basis(diagonal.basis.conjugate_basis()).raw_data,
        diagonal.with_basis(diagonal.basis.conjugate_basis()).raw_data,
    )
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())
