from __future__ import annotations

import numpy as np

from slate_core.array import Array
from slate_core.basis import (
    DiagonalBasis,
    FundamentalBasis,
    SplitBasis,
    TransformedBasis,
    TupleBasis,
    from_metadata,
)


def test_split_array_equals_diagonal() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    array = Array.from_array(data)

    diagonal = array.with_basis(
        DiagonalBasis(
            TupleBasis(
                (array.basis.children[0], array.basis.children[1])
            ).resolve_ctype()
        ).resolve_ctype()
    )
    split = array.with_basis(SplitBasis(diagonal.basis, diagonal.basis).resolve_ctype())

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(split.raw_data, [1.0, 2.0, 3.0, 0, 0, 0])
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())


def test_split_array_equals_transformed() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    basis_k = TransformedBasis(FundamentalBasis.from_size(3)).resolve_ctype()
    array = Array(TupleBasis((basis_k, basis_k)).resolve_ctype(), data)

    diagonal = array.with_basis(DiagonalBasis(array.basis).resolve_ctype())
    fundamental = DiagonalBasis(from_metadata(array.basis.metadata()))
    split = Array(
        SplitBasis(fundamental, diagonal.basis),
        np.array([0, 0, 0, 1.0, 2.0, 3.0]).astype(np.complex128),
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
