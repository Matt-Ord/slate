from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array import SlateArray
from slate.array._transpose import transpose
from slate.array._util import conjugate
from slate.basis import (
    from_shape,
)
from slate.basis.transformed import fundamental_transformed_tuple_basis_from_shape

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.basis._tuple import TupleBasis2D
    from slate.metadata import SimpleMetadata


def test_slate_array_as_array(
    slate_array_integer: SlateArray[SimpleMetadata, np.int64],
) -> None:
    np.testing.assert_array_equal(
        slate_array_integer.raw_data, slate_array_integer.as_array().ravel()
    )


def test_slate_array_dtype(
    slate_array_integer: SlateArray[SimpleMetadata, np.int64],
) -> None:
    assert slate_array_integer.dtype == np.int64


def test_slate_array_basis(
    slate_array_integer: SlateArray[SimpleMetadata, np.int64],
) -> None:
    assert slate_array_integer.basis == from_shape(
        slate_array_integer.fundamental_shape
    )


def test_create_array_with_wrong_size() -> None:
    with pytest.raises(AssertionError):
        SlateArray(from_shape((2, 3)), np.array([1, 2, 3, 4]))


def test_create_array_shape(sample_data: np.ndarray[Any, np.dtype[np.int64]]) -> None:
    slate_array = SlateArray.from_array(sample_data)
    np.testing.assert_array_equal(slate_array.as_array(), sample_data)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((2, 3)),
        from_shape((2, 3), is_dual=(False, True)),
        fundamental_transformed_tuple_basis_from_shape((2, 3)),
    ],
)
def test_transpose_array(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    data = np.array([1, 2, 3, 4, 4, 6])
    array = SlateArray(basis, data)

    np.testing.assert_allclose(
        array.as_array().transpose(), transpose(array).as_array()
    )
    np.testing.assert_allclose(array.as_array(), transpose(transpose(array)).as_array())
    assert transpose(transpose(array)).basis == array.basis


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((2, 3)),
        from_shape((2, 3), is_dual=(False, True)),
        fundamental_transformed_tuple_basis_from_shape((2, 3)),
    ],
)
def test_conjugate_array(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    data = np.array([1, 2, 3, 4, 4, 6])
    array = SlateArray(basis, data)

    np.testing.assert_allclose(
        array.as_array().conjugate(), conjugate(array).as_array()
    )
    np.testing.assert_allclose(array.as_array(), conjugate(conjugate(array)).as_array())
    assert conjugate(array).basis == array.basis


def test_add_mul_array() -> None:
    data = SlateArray.from_array(np.array([1, 2, 3, 4, 4, 6]).reshape(2, 3))

    np.testing.assert_array_equal(
        data.as_array() + data.as_array(), (data + data).as_array()
    )
    np.testing.assert_array_equal(
        data.as_array() - data.as_array(), (data - data).as_array()
    )
    np.testing.assert_array_equal(2 * data.as_array(), (data * 2).as_array())
