from __future__ import annotations

import numpy as np

from slate.array import Array, build
from slate.basis import (
    DiagonalBasis,
    FundamentalBasis,
    TransformedBasis,
)
from slate.basis._split import SplitBasis
from slate.basis._tuple import TupleBasis, from_metadata


def test_split_array_equals_diagonal() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    array = Array.from_array(data)

    diagonal = array.with_basis(
        DiagonalBasis(
            TupleBasis((array.basis.children[0], array.basis.children[1])).upcast()
        )
    ).ok()
    split = array.with_basis(SplitBasis(diagonal.basis, diagonal.basis).upcast()).ok()

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(split.raw_data, [1.0, 2.0, 3.0, 0, 0, 0])
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())


def test_split_array_equals_transformed() -> None:
    data = np.diag(np.arange(1, 4)).astype(np.complex128)
    basis_k = TransformedBasis(FundamentalBasis.from_size(3))
    array = build(TupleBasis((basis_k, basis_k)).upcast(), data).ok()

    diagonal = array.with_basis(DiagonalBasis(array.basis)).ok()
    fundamental = DiagonalBasis(from_metadata(array.basis.metadata()))
    split = build(
        SplitBasis(fundamental, diagonal.basis),
        np.array([0, 0, 0, 1.0, 2.0, 3.0]),
    ).ok()

    np.testing.assert_allclose(diagonal.raw_data, [1.0, 2.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(split.raw_data, [0, 0, 0, 1.0, 2.0, 3.0], atol=1e-15)
    np.testing.assert_allclose(
        split.with_basis(diagonal.basis).ok().raw_data,
        diagonal.raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        split.with_basis(fundamental.inner.dual_basis()).ok().raw_data,
        split.with_basis(fundamental.inner).ok().raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        diagonal.with_basis(fundamental.inner)
        .ok()
        .with_basis(fundamental.inner.dual_basis())
        .ok()
        .raw_data,
        diagonal.with_basis(fundamental.inner).ok().raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        split.with_basis(diagonal.basis.dual_basis()).ok().raw_data,
        diagonal.with_basis(diagonal.basis.dual_basis()).ok().raw_data,
        atol=1e-15,
    )
    np.testing.assert_allclose(diagonal.as_array(), split.as_array())
