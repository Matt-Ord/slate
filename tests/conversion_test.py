from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.transformed import TransformedBasis

if TYPE_CHECKING:
    from slate.array.array import SlateArray
    from slate.basis.basis import FundamentalBasis
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
