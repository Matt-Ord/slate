from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate import array
from slate.array import Array, conjugate, transpose
from slate.basis import (
    Basis,
    BlockDiagonalBasis,
    DiagonalBasis,
    TupleBasis2D,
    from_shape,
    fundamental_transformed_tuple_basis_from_shape,
)

if TYPE_CHECKING:
    from slate.metadata import SimpleMetadata, StackedMetadata


def test_slate_array_as_array(
    slate_array_integer: Array[SimpleMetadata, np.int64],
) -> None:
    np.testing.assert_array_equal(
        slate_array_integer.raw_data, slate_array_integer.as_array().ravel()
    )


def test_slate_array_dtype(
    slate_array_integer: Array[SimpleMetadata, np.int64],
) -> None:
    assert slate_array_integer.dtype == np.int64


def test_slate_array_basis(
    slate_array_integer: Array[SimpleMetadata, np.int64],
) -> None:
    assert slate_array_integer.basis == from_shape(
        slate_array_integer.fundamental_shape
    )


def test_create_array_with_wrong_size() -> None:
    with pytest.raises(AssertionError):
        Array(from_shape((2, 3)), np.array([1, 2, 3, 4]))


def test_create_array_shape(sample_data: np.ndarray[Any, np.dtype[np.int64]]) -> None:
    slate_array = Array.from_array(sample_data)
    np.testing.assert_array_equal(slate_array.as_array(), sample_data)


@pytest.mark.parametrize(
    "basis",
    [
        (from_shape((10, 5))),
        (from_shape((4, 10), is_dual=(False, True))),
        (from_shape((10, 7), is_dual=(True, False))),
        (from_shape((8, 10), is_dual=(True, True))),
        (DiagonalBasis(from_shape((3, 3)))),
        (BlockDiagonalBasis(from_shape((10, 10)), (2, 2))),
        (fundamental_transformed_tuple_basis_from_shape((2, 3))),
    ],
)
def test_transpose_array(basis: Basis[StackedMetadata[Any, Any], Any]) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.size).astype(np.complex128)
    arr = Array(basis, data)

    transposed = array.transpose(arr)
    np.testing.assert_allclose(transposed.as_array(), arr.as_array().transpose())
    np.testing.assert_allclose(arr.as_array(), transpose(transpose(arr)).as_array())


@pytest.mark.parametrize(
    ("basis", "axes"),
    [
        (from_shape((10, 5)), None),
        (from_shape((1, 10, 2, 7, 9)), None),
        (from_shape((1, 10, 2, 7, 9)), tuple(range(5))),
        (from_shape((1, 10, 2, 7, 9)), (1, 3, 4, 2, 0)),
        (from_shape((1, 10, 2, 7, 9)), (3, 4, 0, 2, 1)),
    ],
)
def test_transpose_with_axes(
    basis: Basis[StackedMetadata[Any, Any], Any], axes: tuple[int, ...] | None
) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.size).astype(np.complex128)
    arr = Array(basis, data)

    transposed = array.transpose(arr, axes=axes)
    np.testing.assert_allclose(
        transposed.as_array(), arr.as_array().transpose(axes), rtol=1e-15
    )


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
    array = Array(basis, data)

    np.testing.assert_allclose(
        array.as_array().conjugate(), conjugate(array).as_array()
    )
    np.testing.assert_allclose(array.as_array(), conjugate(conjugate(array)).as_array())
    assert conjugate(array).basis == array.basis


def test_add_mul_array() -> None:
    data = Array.from_array(np.array([1, 2, 3, 4, 4, 6]).reshape(2, 3))

    np.testing.assert_array_equal(
        data.as_array() + data.as_array(), (data + data).as_array()
    )
    np.testing.assert_array_equal(
        data.as_array() - data.as_array(), (data - data).as_array()
    )
    np.testing.assert_array_equal(2 * data.as_array(), (data * 2).as_array())
