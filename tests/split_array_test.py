from __future__ import annotations

import numpy as np

from slate.array import Array
from slate.basis import (
    DiagonalBasis,
    FundamentalBasis,
    TransformedBasis,
    diagonal_basis,
    tuple_basis,
)
from slate.basis._tuple import from_metadata
from slate.basis.split import SplitBasis


def test_split_array_equals_diagonal() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    array = Array.from_array(data)

    diagonal = array.with_basis(diagonal_basis((array.basis[0], array.basis[1])))
    split = array.with_basis(SplitBasis(diagonal.basis, diagonal.basis))

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(split.raw_data, [1.0, 2.0, 3.0, 0, 0, 0])
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())


def test_split_array_equals_transformed() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    basis_k = TransformedBasis(FundamentalBasis.from_size(3))
    array = Array(tuple_basis((basis_k, basis_k)), data)

    diagonal = array.with_basis(DiagonalBasis(array.basis))
    fundamental = DiagonalBasis(
        from_metadata(array.basis.metadata()),
    )
    split = Array(
        SplitBasis(fundamental, diagonal.basis),
        np.array([0, 0, 0, 1.0, 2.0, 3.0]),
    )

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(split.raw_data, [0, 0, 0, 1.0, 2.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(
        split.with_basis(diagonal.basis).raw_data,
        diagonal.raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        split.with_basis(fundamental.inner.dual_basis()).raw_data,
        split.with_basis(fundamental.inner).raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        diagonal.with_basis(fundamental.inner)
        .with_basis(fundamental.inner.dual_basis())
        .raw_data,
        diagonal.with_basis(fundamental.inner).raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        split.with_basis(diagonal.basis.dual_basis()).raw_data,
        diagonal.with_basis(diagonal.basis.dual_basis()).raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())
