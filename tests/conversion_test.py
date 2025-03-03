from __future__ import annotations

from typing import TYPE_CHECKING, Never

import numpy as np
import pytest

from slate.array import Array, build
from slate.basis import (
    BlockDiagonalBasis,
    FundamentalBasis,
    RecastBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    from_shape,
    transformed_from_shape,
)
from slate.basis._diagonal import DiagonalBasis
from slate.basis._tuple import TupleBasis

if TYPE_CHECKING:
    from slate.basis._basis import Basis, ctype
    from slate.metadata import SimpleMetadata
    from slate.metadata._metadata import BasisMetadata


def test_transformed_basis_round_trip(
    slate_array_complex: Array[
        FundamentalBasis[SimpleMetadata], np.dtype[np.complex128]
    ],
) -> None:
    basis = TransformedBasis(slate_array_complex.basis)

    converted_array = slate_array_complex.with_basis(
        TransformedBasis(slate_array_complex.basis).upcast()
    ).ok()
    assert converted_array.basis == basis
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        slate_array_complex.as_array(),
    )

    round_trip_array = converted_array.with_basis(slate_array_complex.basis).ok()
    assert round_trip_array.basis == slate_array_complex.basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        slate_array_complex.raw_data,
    )


def test_transformed_basis() -> None:
    fundamental_basis = FundamentalBasis.from_size(5)
    transformed_basis = TransformedBasis(fundamental_basis).upcast()

    rng = np.random.default_rng()
    data = rng.random(fundamental_basis.size) + 1j * rng.random(fundamental_basis.size)
    array = build(fundamental_basis, data).ok()
    np.testing.assert_array_almost_equal(array.raw_data, data)
    np.testing.assert_array_almost_equal(
        array.with_basis(transformed_basis).ok().raw_data,
        np.fft.fft(data, norm="ortho"),
    )

    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis.dual_basis()).ok().raw_data, np.conj(data)
    )
    array = build(fundamental_basis.dual_basis(), data).ok()
    np.testing.assert_array_almost_equal(
        array.with_basis(transformed_basis.dual_basis()).ok().raw_data,
        np.conj(np.fft.fft(np.conj(data), norm="ortho")),
    )

    array = build(transformed_basis, data).ok()
    np.testing.assert_array_almost_equal(array.raw_data, data)
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis).ok().raw_data,
        np.fft.ifft(data, norm="ortho"),
    )
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis.dual_basis()).ok().raw_data,
        np.conj(np.fft.ifft(data, norm="ortho")),
    )

    array = build(transformed_basis.dual_basis(), data).ok()
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis).ok().raw_data,
        np.fft.ifft(np.conj(data), norm="ortho"),
    )
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis.dual_basis()).ok().raw_data,
        np.conj(np.fft.ifft(np.conj(data), norm="ortho")),
    )


def test_diagonal_basis_round_trip() -> None:
    full_basis = from_shape((10, 10))
    basis_diagonal = DiagonalBasis(full_basis).upcast()

    array = build(full_basis, np.diag(np.ones(10))).ok()

    converted_array = array.with_basis(basis_diagonal).ok()
    assert converted_array.basis == basis_diagonal
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        array.as_array(),
    )

    round_trip_array = converted_array.with_basis(full_basis).ok()
    assert round_trip_array.basis == full_basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        array.raw_data,
    )

    array = build(full_basis, np.ones(full_basis.shape)).ok()

    converted_array = array.with_basis(basis_diagonal).ok()
    assert converted_array.basis == basis_diagonal
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        np.diag(np.diag(array.as_array())),
    )

    round_trip_array = converted_array.with_basis(full_basis).ok()
    assert round_trip_array.basis == full_basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        np.diag(np.diag(array.as_array())).ravel(),
    )


def test_transform_spaced_basis() -> None:
    half_basis = from_shape((105,))
    full_basis = TupleBasis((half_basis, half_basis)).upcast()
    spaced_basis = TruncatedBasis(
        Truncation(3, 5, 0), TransformedBasis(half_basis).upcast()
    ).upcast()

    array = build(
        RecastBasis(
            DiagonalBasis(full_basis).upcast(),
            half_basis,
            spaced_basis,
        ).upcast(),
        np.ones(spaced_basis.size, dtype=np.complex128),
    ).ok()

    converted_array = array.with_basis(full_basis).ok()
    assert converted_array.basis == TupleBasis((half_basis, half_basis))
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        array.as_array(),
    )
    np.testing.assert_array_almost_equal(
        converted_array.raw_data,
        array.as_array().ravel(),
    )
    np.testing.assert_equal(full_basis.size, array.as_array().size)
    np.testing.assert_array_almost_equal(
        array.basis.__convert_vector_into__(array.raw_data, full_basis, axis=-1).ok(),
        array.as_array().ravel(),
    )

    round_trip_array = converted_array.with_basis(array.basis).ok()
    assert round_trip_array.basis == array.basis
    np.testing.assert_array_almost_equal(round_trip_array.raw_data, array.raw_data)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
        transformed_from_shape((10, 10)),
    ],
)
def test_dual_basis_transform(
    basis: TupleBasis[
        tuple[Basis[BasisMetadata, ctype[np.complex128]], ...],
        Never,
        ctype[np.complex128],
    ],
) -> None:
    dual_basis = basis.dual_basis()
    dual_child_basis = TupleBasis(
        tuple(child.dual_basis() for child in basis.children),
        extra=dual_basis.metadata().extra,
    ).upcast()
    array = build(basis, np.ones((10, 10), dtype=np.complex128)).ok()

    np.testing.assert_array_almost_equal(
        array.with_basis(dual_basis).ok().raw_data,
        array.with_basis(dual_child_basis).ok().raw_data,
    )
    np.testing.assert_array_almost_equal(
        array.with_basis(dual_basis).ok().as_array(),
        array.with_basis(dual_child_basis).ok().as_array(),
    )
    np.testing.assert_array_almost_equal(
        array.as_array(),
        array.with_basis(dual_child_basis).ok().as_array(),
    )


def test_block_basis() -> None:
    data = np.arange(4 * 6).reshape(4, 6)
    array = Array.from_array(data)

    block_basis = BlockDiagonalBasis(array.basis, (2, 2)).upcast()

    in_block_basis = array.with_basis(block_basis).ok()

    np.testing.assert_allclose(
        in_block_basis.raw_data,
        [0, 1, 6, 7, 14, 15, 20, 21],
    )
    np.testing.assert_allclose(
        in_block_basis.as_array(),
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 7.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 14.0, 15.0, 0.0, 0.0],
            [0.0, 0.0, 20.0, 21.0, 0.0, 0.0],
        ],
    )
