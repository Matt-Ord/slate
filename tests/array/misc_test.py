from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from slate_core.array import (
    Array,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    cos,
    cosh,
    sin,
    sinh,
    tan,
    tanh,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_core.basis import Basis


def generate_ranged_data[T: np.generic](
    input_range: tuple[float, float], shape: tuple[int, ...], dtype: type[T]
) -> Array[Basis, np.dtype[T]]:
    rng = np.random.default_rng()
    start, end = input_range
    width = end - start
    data = (rng.random(shape) * width + start).astype(dtype)
    return Array.from_array(data)


def test_trig_identities() -> None:
    """Test fundamental trigonometric identities on slate arrays."""
    arr = generate_ranged_data((-np.pi, np.pi), (3, 4), np.float64)

    # Test sin²(x) + cos²(x) = 1
    sin_squared = sin(arr).as_array() ** 2
    cos_squared = cos(arr).as_array() ** 2
    np.testing.assert_allclose(
        sin_squared + cos_squared, np.ones_like(arr.as_array()), rtol=1e-10
    )

    # Test cosh²(x) - sinh²(x) = 1
    sinh_squared = sinh(arr).as_array() ** 2
    cosh_squared = cosh(arr).as_array() ** 2
    np.testing.assert_allclose(
        cosh_squared - sinh_squared, np.ones_like(arr.as_array()), rtol=1e-10
    )


@pytest.mark.parametrize(
    ("func", "input_range"),
    [
        (sin, (-np.pi, np.pi)),
        (cos, (-np.pi, np.pi)),
        (tan, (-np.pi / 2 + 0.1, np.pi / 2 - 0.1)),
        (arcsin, (-1, 1)),
        (arccos, (0, 1)),
        (arctan, (-np.pi, np.pi)),
        (sinh, (-1, 1)),
        (cosh, (0, np.pi)),
        (tanh, (-2, 2)),
        (arcsinh, (-np.pi, np.pi)),
        (arccosh, (1, np.pi)),
        (arctanh, (-1, 1)),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.float64, np.complex64, np.complex128],
)
def test_trig_preserve_dtype(
    func: Callable[..., Array[Basis, np.dtype[np.generic]]],
    input_range: tuple[float, float],
    dtype: type[np.generic],
) -> None:
    """Test that trigonometric functions preserve the input data type."""
    arr = generate_ranged_data(input_range, (5, 4), dtype)

    result = func(arr)
    assert result.as_array().dtype == dtype


@pytest.mark.parametrize(
    ("func", "input_range"),
    [
        (sin, (-np.pi, np.pi)),
        (cos, (-np.pi, np.pi)),
        (tan, (-np.pi / 2 + 0.1, np.pi / 2 - 0.1)),
        (arcsin, (-1, 1)),
        (arccos, (0, 1)),
        (arctan, (-np.pi, np.pi)),
        (sinh, (-1, 1)),
        (cosh, (0, np.pi)),
        (tanh, (-2, 2)),
        (arcsinh, (-np.pi, np.pi)),
        (arccosh, (1, np.pi)),
        (arctanh, (-1, 1)),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [(3, 4), (2, 5, 3), (4, 1, 6, 2)],
)
def test_trig_preserve_shape(
    func: Callable[..., Array[Basis, np.dtype[np.generic]]],
    input_range: tuple[float, float],
    shape: tuple[int, ...],
) -> None:
    """Test that trigonometric functions preserve the shape of input arrays."""
    arr = generate_ranged_data(input_range, shape, np.float64)

    result = func(arr)
    assert result.as_array().shape == shape


def test_sin() -> None:
    arr = generate_ranged_data((-np.pi, np.pi), (3, 4), np.float64)

    # The sin function should match numpy's sin function
    result = sin(arr)
    expected = np.sin(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Arcsin acts as the inverse of sin in the range [-np.pi / 2, np.pi / 2]
    arr = generate_ranged_data((-np.pi / 2, np.pi / 2), (3, 4), np.float64)
    inverse = arcsin(sin(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_cos() -> None:
    """Test and demonstrate the cosine function on slate arrays."""
    arr = generate_ranged_data((-np.pi, np.pi), (3, 4), np.float64)

    # The cos function should match numpy's cos function
    result = cos(arr)
    expected = np.cos(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Arccos acts as the inverse of cos in the range [0, pi]
    arr = generate_ranged_data((0, np.pi), (3, 4), np.float64)
    inverse = arccos(cos(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_arcsin() -> None:
    """Test and demonstrate the arcsine function on slate arrays."""
    # Create an array with values in the domain of arcsin [-1, 1]
    arr = generate_ranged_data((-1, 1), (3, 4), np.float64)

    # The arcsin function should match numpy's arcsin function
    result = arcsin(arr)
    expected = np.arcsin(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Sin acts as the inverse of arcsin
    inverse = sin(arcsin(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_arccos() -> None:
    """Test and demonstrate the arccosine function on slate arrays."""
    # Create an array with values in the domain of arccos [-1, 1]
    arr = generate_ranged_data((-1, 1), (3, 4), np.float64)

    # The arccos function should match numpy's arccos function
    result = arccos(arr)
    expected = np.arccos(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Cos acts as the inverse of arccos
    inverse = cos(arccos(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_arctan() -> None:
    """Test and demonstrate the arctangent function on slate arrays."""
    # Create an array with a range of values for arctan
    arr = generate_ranged_data((-10, 10), (3, 4), np.float64)

    # The arctan function should match numpy's arctan function
    result = arctan(arr)
    expected = np.arctan(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Tan acts as the inverse of arctan
    arr = generate_ranged_data((-np.pi / 2, np.pi / 2), (3, 4), np.float64)
    inverse = tan(arctan(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_sinh() -> None:
    """Test and demonstrate the hyperbolic sine function on slate arrays."""
    arr = generate_ranged_data((-2, 2), (3, 4), np.float64)

    # The sinh function should match numpy's sinh function
    result = sinh(arr)
    expected = np.sinh(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Arcsinh acts as the inverse of sinh
    inverse = arcsinh(sinh(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_cosh() -> None:
    """Test and demonstrate the hyperbolic cosine function on slate arrays."""
    arr = generate_ranged_data((-2, 2), (3, 4), np.float64)

    # The cosh function should match numpy's cosh function
    result = cosh(arr)
    expected = np.cosh(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Arccosh acts as the inverse of cosh for values ≥ 1
    # cosh's range is [1, inf), so we need values ≥ 1
    cosh_result = cosh(arr)
    inverse = arccosh(cosh_result)
    # Since cosh is even, arccosh gives the absolute value of the input
    np.testing.assert_allclose(inverse.as_array(), np.abs(arr.as_array()), rtol=1e-10)


def test_arcsinh() -> None:
    """Test and demonstrate the inverse hyperbolic sine function on slate arrays."""
    arr = generate_ranged_data((-10, 10), (3, 4), np.float64)

    # The arcsinh function should match numpy's arcsinh function
    result = arcsinh(arr)
    expected = np.arcsinh(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Sinh acts as the inverse of arcsinh
    inverse = sinh(arcsinh(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_arccosh() -> None:
    """Test and demonstrate the inverse hyperbolic cosine function on slate arrays."""
    # Domain of arccosh is [1, inf)
    arr = generate_ranged_data((1, 10), (3, 4), np.float64)

    # The arccosh function should match numpy's arccosh function
    result = arccosh(arr)
    expected = np.arccosh(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Cosh acts as the inverse of arccosh
    inverse = cosh(arccosh(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)


def test_arctanh() -> None:
    """Test and demonstrate the inverse hyperbolic tangent function on slate arrays."""
    # Domain of arctanh is (-1, 1)
    arr = generate_ranged_data((-0.99, 0.99), (3, 4), np.float64)

    # The arctanh function should match numpy's arctanh function
    result = arctanh(arr)
    expected = np.arctanh(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # tanh(arctanh(x)) = x
    result = tanh(arctanh(arr))
    np.testing.assert_allclose(result.as_array(), arr.as_array(), rtol=1e-10)


def test_tan() -> None:
    """Test and demonstrate the tangent function on slate arrays."""
    # Avoid values close to π/2 + nπ where tan has singularities
    arr = generate_ranged_data((-np.pi / 2 + 0.1, np.pi / 2 - 0.1), (3, 4), np.float64)

    # The tan function should match numpy's tan function
    result = tan(arr)
    expected = np.tan(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Arctan acts as the inverse of tan
    inverse = arctan(tan(arr))
    np.testing.assert_allclose(inverse.as_array(), arr.as_array(), rtol=1e-10)

    # Test the relation tan(x) = sin(x) / cos(x)
    sin_result = sin(arr).as_array()
    cos_result = cos(arr).as_array()
    expected_tan = sin_result / cos_result
    np.testing.assert_allclose(result.as_array(), expected_tan, rtol=1e-10)


def test_tanh() -> None:
    """Test and demonstrate the hyperbolic tangent function on slate arrays."""
    arr = generate_ranged_data((-2, 2), (3, 4), np.float64)

    # The tanh function should match numpy's tanh function
    result = tanh(arr)
    expected = np.tanh(arr.as_array())
    np.testing.assert_allclose(result.as_array(), expected, rtol=1e-10)

    # Arctanh acts as the inverse of tanh
    # Domain of tanh is (-1, 1), so we need to ensure values are in that range
    tanh_arr = generate_ranged_data((-0.99, 0.99), (3, 4), np.float64)
    inverse = arctanh(tanh(tanh_arr))
    np.testing.assert_allclose(inverse.as_array(), tanh_arr.as_array(), rtol=1e-10)

    # Test the relation tanh(x) = sinh(x) / cosh(x)
    sinh_result = sinh(arr).as_array()
    cosh_result = cosh(arr).as_array()
    expected_tanh = sinh_result / cosh_result
    np.testing.assert_allclose(result.as_array(), expected_tanh, rtol=1e-10)
