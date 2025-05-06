from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate_core import array, basis
from slate_core.array import Array, build, conjugate, transpose
from slate_core.basis import (
    Basis,
    BlockDiagonalBasis,
    DiagonalBasis,
    from_shape,
    transformed_from_shape,
)

if TYPE_CHECKING:
    from slate_core.basis import Ctype, TupleBasis
    from slate_core.metadata import BasisMetadata, SimpleMetadata, TupleMetadata


def test_slate_array_as_array(
    slate_array_integer: Array[Basis, np.dtype[np.int64]],
) -> None:
    np.testing.assert_array_equal(
        slate_array_integer.raw_data, slate_array_integer.as_array().ravel()
    )


def test_slate_array_dtype(
    slate_array_integer: Array[Basis, np.dtype[np.int64]],
) -> None:
    assert slate_array_integer.dtype == np.int64


def test_slate_array_basis(
    slate_array_integer: Array[Basis, np.dtype[np.int64]],
) -> None:
    assert slate_array_integer.basis == from_shape(
        slate_array_integer.fundamental_shape
    )


def test_create_array_with_wrong_size() -> None:
    with pytest.raises(AssertionError):
        build(from_shape((2, 3)), np.array([1, 2, 3, 4])).ok()


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
        (transformed_from_shape((2, 3))),
    ],
)
def test_transpose_array(basis: Basis[TupleMetadata[Any, Any], Any]) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.size).astype(np.complex128)
    arr = build(basis, data).ok()

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
    basis: Basis[
        TupleMetadata[tuple[BasisMetadata, BasisMetadata], Any], Ctype[np.generic]
    ],
    axes: tuple[int, ...] | None,
) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.size).astype(np.complex128)
    arr = build(basis, data).ok()

    transposed = array.transpose(arr, axes=axes)
    np.testing.assert_allclose(
        transposed.as_array(), arr.as_array().transpose(axes), rtol=1e-15
    )


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((2, 3)),
        from_shape((2, 3), is_dual=(False, True)),
        transformed_from_shape((2, 3)),
    ],
)
def test_conjugate_array(
    basis: TupleBasis[
        tuple[
            Basis[SimpleMetadata, Ctype[np.generic]],
            Basis[SimpleMetadata, Ctype[np.generic]],
        ],
        None,
        Ctype[np.generic],
    ],
) -> None:
    data = np.array([1, 2, 3, 4, 4, 6])
    array = build(basis, data).ok()

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


def test_extract_diagonal() -> None:
    data = np.array([1, 2, 3, 4, 4, 6, 7, 8, 9]).reshape(3, 3)
    slate_array = Array.build(basis.from_shape((3, 3)).upcast(), data).assert_ok()
    diagonal = array.extract_diagonal(slate_array)
    np.testing.assert_array_equal(diagonal.as_array(), data.diagonal())
