from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate_core import array
from slate_core.array import (
    Array,
    average,
    max,  # noqa: A004
    min,  # noqa: A004
    standard_deviation,
    transpose,
)
from slate_core.basis import (
    Basis,
    BlockDiagonalBasis,
    DiagonalBasis,
    from_shape,
    transformed_from_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_core.metadata import TupleMetadata


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
    arr = Array(basis, data)

    transposed = array.transpose(arr)
    np.testing.assert_allclose(transposed.as_array(), arr.as_array().transpose())
    np.testing.assert_allclose(arr.as_array(), transpose(transpose(arr)).as_array())


def test_standard_deviation() -> None:
    # Create a 2D array with known values
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float64)
    arr = Array.from_array(data)

    # Get standard deviation of all elements
    std_all = standard_deviation(arr, axis=None)
    assert np.isclose(std_all, 3.452052529534663)
    # Verify it matches NumPy's std
    np.testing.assert_allclose(std_all, np.std(data))

    # Get standard deviation along rows (axis 0)
    std_axis0 = standard_deviation(arr, axis=0)
    expected_axis0 = np.array([3.26598632, 3.26598632, 3.26598632, 3.26598632])
    np.testing.assert_allclose(std_axis0.as_array(), expected_axis0)
    # Verify it matches NumPy's std along axis 0
    np.testing.assert_allclose(std_axis0.as_array(), np.std(data, axis=0))

    # Get standard deviation along columns (axis 1)
    std_axis1 = standard_deviation(arr, axis=1)
    expected_axis1 = np.array([1.11803399, 1.11803399, 1.11803399])
    np.testing.assert_allclose(std_axis1.as_array(), expected_axis1)
    # Verify it matches NumPy's std along axis 1
    np.testing.assert_allclose(std_axis1.as_array(), np.std(data, axis=1))


def test_average() -> None:
    # Create a 2D array with known values
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float64)
    arr = Array.from_array(data)

    # Get average of all elements
    avg_all = average(arr, axis=None)
    assert np.isclose(avg_all, 6.5)
    # Verify it matches NumPy's mean
    np.testing.assert_allclose(avg_all, np.mean(data))

    # Get average along rows (axis 0)
    avg_axis0 = average(arr, axis=0)
    np.testing.assert_allclose(avg_axis0.as_array(), [5.0, 6.0, 7.0, 8.0])
    # Verify it matches NumPy's mean along axis 0
    np.testing.assert_allclose(avg_axis0.as_array(), np.mean(data, axis=0))

    # Get average along columns (axis 1)
    avg_axis1 = average(arr, axis=1)
    np.testing.assert_allclose(avg_axis1.as_array(), [2.5, 6.5, 10.5])
    # Verify it matches NumPy's mean along axis 1
    np.testing.assert_allclose(avg_axis1.as_array(), np.mean(data, axis=1))


def test_min() -> None:
    # Create a 2D array with known values
    data = np.array([[4, 1, 8, 2], [3, 9, 1, 7], [6, 2, 5, 3]], dtype=np.float64)
    arr = Array.from_array(data)

    # Get minimum of all elements
    min_all = min(arr, axis=None)
    assert np.isclose(min_all, 1.0)
    # Verify it matches NumPy's min
    np.testing.assert_allclose(min_all, np.min(data))

    # Get minimum along rows (axis 0)
    min_axis0 = min(arr, axis=0)
    np.testing.assert_allclose(min_axis0.as_array(), [3.0, 1.0, 1.0, 2.0])
    # Verify it matches NumPy's min along axis 0
    np.testing.assert_allclose(min_axis0.as_array(), np.min(data, axis=0))

    # Get minimum along columns (axis 1)
    min_axis1 = min(arr, axis=1)
    np.testing.assert_allclose(min_axis1.as_array(), [1.0, 1.0, 2.0])
    # Verify it matches NumPy's min along axis 1
    np.testing.assert_allclose(min_axis1.as_array(), np.min(data, axis=1))


def test_max() -> None:
    # Create a 2D array with known values
    data = np.array([[4, 1, 8, 2], [3, 9, 1, 7], [6, 2, 5, 3]], dtype=np.float64)
    arr = Array.from_array(data)

    # Get maximum of all elements
    max_all = max(arr, axis=None)
    assert np.isclose(max_all, 9.0)
    # Verify it matches NumPy's max
    np.testing.assert_allclose(max_all, np.max(data))

    # Get maximum along rows (axis 0)
    max_axis0 = max(arr, axis=0)
    np.testing.assert_allclose(max_axis0.as_array(), [6.0, 9.0, 8.0, 7.0])
    # Verify it matches NumPy's max along axis 0
    np.testing.assert_allclose(max_axis0.as_array(), np.max(data, axis=0))

    # Get maximum along columns (axis 1)
    max_axis1 = max(arr, axis=1)
    np.testing.assert_allclose(max_axis1.as_array(), [8.0, 9.0, 6.0])
    # Verify it matches NumPy's max along axis 1
    np.testing.assert_allclose(max_axis1.as_array(), np.max(data, axis=1))


@pytest.mark.parametrize(
    "func",
    [standard_deviation, average, min, max],
)
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.complex64, np.complex128],
)
@pytest.mark.parametrize(
    "axis",
    [None, 0, 1],
)
def test_preserve_dtype(
    func: Callable[..., Array[Basis, np.dtype[np.generic]]],
    dtype: np.dtype[np.generic],
    axis: int | None,
) -> None:
    rng = np.random.default_rng(42)
    data = rng.random((5, 4)).astype(dtype)
    arr = Array.from_array(data)

    # Apply statistical function with the given axis
    result = func(arr, axis=axis)

    # Check if result is a scalar or an array
    if axis is None:
        assert result.dtype == dtype
    else:
        assert result.as_array().dtype == dtype


@pytest.mark.parametrize(
    ("func", "neg_axis", "pos_axis"),
    [
        (standard_deviation, -1, 2),
        (average, -2, 1),
        (min, -3, 0),
        (max, -1, 2),
    ],
)
def test_stats_negative_axis(
    func: Callable[..., Array[Basis, np.dtype[np.number]]],
    neg_axis: int,
    pos_axis: int,
) -> None:
    rng = np.random.default_rng(42)
    data = rng.random((5, 4, 3)).astype(np.float64)
    arr = Array.from_array(data)

    # Test that negative and positive axis values produce the same result
    result_neg = func(arr, axis=neg_axis)
    result_pos = func(arr, axis=pos_axis)

    # Compare arrays
    np.testing.assert_allclose(
        result_neg.as_array(),
        result_pos.as_array(),
        err_msg=f"{func.__name__} results differ for axis {neg_axis} vs {pos_axis}",
    )


@pytest.mark.parametrize(
    "func",
    [
        standard_deviation,
        average,
        min,
        max,
    ],
)
@pytest.mark.parametrize(
    ("axis", "expected_shape"),
    [
        (0, (3, 4)),
        (1, (2, 4)),
        (2, (2, 3)),
    ],
)
def test_shape(
    func: Callable[..., Array[Basis, np.dtype[np.number]]],
    axis: int,
    expected_shape: tuple[int, int],
) -> None:
    data = np.arange(24).reshape(2, 3, 4)
    arr = Array.from_array(data)

    # Apply statistical function with the given axis
    result = func(arr, axis=axis)

    # Verify the function produces the correct output shape
    assert result.as_array().shape == expected_shape, (
        f"{func.__name__} with axis={axis} produced incorrect shape"
    )


@pytest.mark.parametrize(
    ("func", "expected_scalar"),
    [
        (standard_deviation, 3.452052529534663),
        (average, 6.5),
        (min, 1.0),
        (max, 12.0),
    ],
)
def test_stats_axis_none(func: Callable[..., float], expected_scalar: float) -> None:
    # Create a 2D array with known values
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float64)
    arr = Array.from_array(data)

    # Apply statistical function with axis=None
    result = func(arr, axis=None)

    # Verify the result is a scalar with the expected value
    assert np.isclose(result, expected_scalar), (
        f"{func.__name__} with axis=None returned {result} instead of {expected_scalar}"
    )
