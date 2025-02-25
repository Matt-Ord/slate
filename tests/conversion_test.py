from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array import Array
from slate.basis import (
    BlockDiagonalBasis,
    FundamentalBasis,
    RecastBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    from_shape,
    fundamental_transformed_tuple_basis_from_shape,
)

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.metadata import SimpleMetadata


def test_transformed_basis_round_trip(
    slate_array_complex: Array[Any, np.complex128, FundamentalBasis[SimpleMetadata]],
) -> None:
    basis = TransformedBasis(slate_array_complex.basis)

    converted_array = with_basis(
        slate_array_complex, TransformedBasis(slate_array_complex.basis)
    )
    assert converted_array.basis == basis
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        slate_array_complex.as_array(),
    )

    round_trip_array = with_basis(converted_array, slate_array_complex.basis)
    assert round_trip_array.basis == slate_array_complex.basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        slate_array_complex.raw_data,
    )


def test_transformed_basis() -> None:
    fundamental_basis = FundamentalBasis.from_size(5)
    transformed_basis = TransformedBasis(fundamental_basis)

    rng = np.random.default_rng()
    data = rng.random(fundamental_basis.size) + 1j * rng.random(fundamental_basis.size)
    array = Array(fundamental_basis, data)
    np.testing.assert_array_almost_equal(array.raw_data, data)
    np.testing.assert_array_almost_equal(
        array.with_basis(transformed_basis).raw_data, np.fft.fft(data, norm="ortho")
    )

    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis.dual_basis()).raw_data, np.conj(data)
    )
    array = Array(fundamental_basis.dual_basis(), data)
    np.testing.assert_array_almost_equal(
        array.with_basis(transformed_basis.dual_basis()).raw_data,
        np.conj(np.fft.fft(np.conj(data), norm="ortho")),
    )

    array = Array(transformed_basis, data)
    np.testing.assert_array_almost_equal(array.raw_data, data)
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis).raw_data, np.fft.ifft(data, norm="ortho")
    )
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis.dual_basis()).raw_data,
        np.conj(np.fft.ifft(data, norm="ortho")),
    )

    array = Array(transformed_basis.dual_basis(), data)
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis).raw_data,
        np.fft.ifft(np.conj(data), norm="ortho"),
    )
    np.testing.assert_array_almost_equal(
        array.with_basis(fundamental_basis.dual_basis()).raw_data,
        np.conj(np.fft.ifft(np.conj(data), norm="ortho")),
    )


def test_diagonal_basis_round_trip() -> None:
    full_basis = from_shape((10, 10))
    basis_diagonal = diagonal_basis(full_basis.children)

    array = Array(full_basis, np.diag(np.ones(10)))

    converted_array = with_basis(array, basis_diagonal)
    assert converted_array.basis == basis_diagonal
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        array.as_array(),
    )

    round_trip_array = with_basis(converted_array, full_basis)
    assert round_trip_array.basis == full_basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        array.raw_data,
    )

    array = Array(full_basis, np.ones(full_basis.shape))

    converted_array = with_basis(array, basis_diagonal)
    assert converted_array.basis == basis_diagonal
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        np.diag(np.diag(array.as_array())),
    )

    round_trip_array = with_basis(converted_array, full_basis)
    assert round_trip_array.basis == full_basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        np.diag(np.diag(array.as_array())).ravel(),
    )


def test_transform_spaced_basis() -> None:
    half_basis = from_shape((105,))
    full_basis = tuple_basis((half_basis, half_basis))
    spaced_basis = TruncatedBasis(Truncation(3, 5, 0), TransformedBasis(half_basis))

    array = Array(
        RecastBasis(
            diagonal_basis((half_basis, half_basis)),
            half_basis,
            spaced_basis,
        ),
        np.ones(spaced_basis.size),
    )

    converted_array = with_basis(array, full_basis)
    assert converted_array.basis == tuple_basis((half_basis, half_basis))
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

    round_trip_array = with_basis(converted_array, array.basis)
    assert round_trip_array.basis == array.basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        array.raw_data,
    )


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
        fundamental_transformed_tuple_basis_from_shape((10, 10)),
    ],
)
def test_dual_basis_transform(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    basis = from_shape((10, 10))

    dual_basis = basis.dual_basis()
    dual_child_basis = tuple_basis(
        tuple(child.dual_basis() for child in basis.children),
        extra_metadata=dual_basis.metadata().extra,
    )

    array = Array(basis, np.ones(basis.size))

    np.testing.assert_array_almost_equal(
        array.with_basis(dual_basis).raw_data,
        array.with_basis(dual_child_basis).raw_data,
    )
    np.testing.assert_array_almost_equal(
        array.with_basis(dual_basis).as_array(),
        array.with_basis(dual_child_basis).as_array(),
    )
    np.testing.assert_array_almost_equal(
        array.as_array(),
        array.with_basis(dual_child_basis).as_array(),
    )


def test_block_basis() -> None:
    data = np.arange(4 * 6).reshape(4, 6)
    array = Array.from_array(data)

    block_basis = BlockDiagonalBasis(array.basis, (2, 2))

    in_block_basis = array.with_basis(block_basis)

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
