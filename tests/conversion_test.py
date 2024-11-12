from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.recast import RecastBasis
from slate.basis.stacked import diagonal_basis
from slate.basis.stacked._tuple_basis import (
    fundamental_tuple_basis_from_shape,
    tuple_basis,
)
from slate.basis.transformed import TransformedBasis
from slate.basis.truncated import TruncatedBasis, Truncation

if TYPE_CHECKING:
    from slate.basis import FundamentalBasis
    from slate.metadata import SimpleMetadata


def test_transformed_basis_round_trip(
    slate_array_complex: SlateArray[np.complex128, FundamentalBasis[SimpleMetadata]],
) -> None:
    basis = TransformedBasis(slate_array_complex.basis)

    converted_array = convert_array(
        slate_array_complex, TransformedBasis(slate_array_complex.basis)
    )
    assert converted_array.basis == basis
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        slate_array_complex.as_array(),
    )

    round_trip_array = convert_array(converted_array, slate_array_complex.basis)
    assert round_trip_array.basis == slate_array_complex.basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        slate_array_complex.raw_data,
    )


def test_diagonal_basis_round_trip() -> None:
    full_basis = fundamental_tuple_basis_from_shape((10, 10))
    basis_diagonal = diagonal_basis(full_basis.children)

    array = SlateArray(full_basis, np.diag(np.ones(10)))

    converted_array = convert_array(array, basis_diagonal)
    assert converted_array.basis == basis_diagonal
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        array.as_array(),
    )

    round_trip_array = convert_array(converted_array, full_basis)
    assert round_trip_array.basis == full_basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        array.raw_data,
    )

    array = SlateArray(full_basis, np.ones(full_basis.shape))

    converted_array = convert_array(array, basis_diagonal)
    assert converted_array.basis == basis_diagonal
    np.testing.assert_array_almost_equal(
        converted_array.as_array(),
        np.diag(np.diag(array.as_array())),
    )

    round_trip_array = convert_array(converted_array, full_basis)
    assert round_trip_array.basis == full_basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        np.diag(np.diag(array.as_array())).ravel(),
    )


def test_transform_spaced_basis() -> None:
    half_basis = fundamental_tuple_basis_from_shape((105,))
    full_basis = tuple_basis((half_basis, half_basis.conjugate_basis()))
    spaced_basis = TruncatedBasis(Truncation(3, 5, 0), TransformedBasis(half_basis))

    array = SlateArray(
        RecastBasis(
            diagonal_basis((half_basis, half_basis.conjugate_basis())),
            half_basis,
            spaced_basis,
        ),
        np.ones(spaced_basis.size),
    )

    converted_array = convert_array(array, full_basis)
    assert converted_array.basis == tuple_basis(
        (half_basis, half_basis.conjugate_basis())
    )
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
        array.basis.__convert_vector_into__(array.raw_data, full_basis, axis=-1),
        array.as_array().ravel(),
    )

    round_trip_array = convert_array(converted_array, array.basis)
    assert round_trip_array.basis == array.basis
    np.testing.assert_array_almost_equal(
        round_trip_array.raw_data,
        array.raw_data,
    )
