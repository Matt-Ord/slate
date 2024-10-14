from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from slate.array.array import SlateArray
from slate.basis import FundamentalBasis
from slate.metadata import SimpleMetadata


def test_slate_array_as_array(
    slate_array_integer: SlateArray[np.int64, FundamentalBasis[SimpleMetadata]],
) -> None:
    np.testing.assert_array_equal(
        slate_array_integer.raw_data, slate_array_integer.as_array().ravel()
    )


def test_slate_array_dtype(
    slate_array_integer: SlateArray[np.int64, FundamentalBasis[SimpleMetadata]],
) -> None:
    assert slate_array_integer.dtype == np.int64


def test_slate_array_basis(
    slate_array_integer: SlateArray[np.int64, FundamentalBasis[SimpleMetadata]],
) -> None:
    assert slate_array_integer.basis == FundamentalBasis(
        SimpleMetadata(slate_array_integer.fundamental_shape)
    )


def test_create_array_with_wrong_size() -> None:
    with pytest.raises(AssertionError):
        SlateArray(FundamentalBasis.from_shape((2, 3)), np.array([1, 2, 3, 4]))


def test_create_array_shape(sample_data: np.ndarray[Any, np.dtype[np.int64]]) -> None:
    slate_array = SlateArray.from_array(sample_data)
    np.testing.assert_array_equal(slate_array.as_array(), sample_data)
