from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis import DiagonalBasis, FundamentalBasis, TupleBasis
from slate.basis.transformed import TransformedBasis

if TYPE_CHECKING:
    from slate.basis.metadata import FundamentalBasisMetadata


def test_transformed_basis_round_trip(
    slate_array_complex: SlateArray[
        FundamentalBasis[FundamentalBasisMetadata, np.complex128]
    ],
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
    full_basis = TupleBasis(
        (FundamentalBasis.from_shape((10,)), FundamentalBasis.from_shape((10,))), None
    )
    diagonal_basis = DiagonalBasis(full_basis)

    array = SlateArray(full_basis, np.diag(np.ones(10)))

    converted_array = convert_array(array, diagonal_basis)
    assert converted_array.basis == diagonal_basis
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

    converted_array = convert_array(array, diagonal_basis)
    assert converted_array.basis == diagonal_basis
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
