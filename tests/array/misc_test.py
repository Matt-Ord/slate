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
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_core.basis import Basis


@pytest.mark.parametrize(
    ("func", "input_range"),
    [
        (sin, (-np.pi, np.pi)),
        (cos, (-np.pi, np.pi)),
        (arcsin, (-1, 1)),
        (arccos, (0, 1)),
        (arctan, (-np.pi, np.pi)),
        (sinh, (-1, 1)),
        (cosh, (0, np.pi)),
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
    dtype: np.dtype[np.generic],
) -> None:
    """Test that trigonometric functions preserve the input data type."""
    rng = np.random.default_rng()
    start, end = input_range
    width = end - start
    data = (rng.random((5, 4)) * width + start).astype(dtype)

    arr = Array.from_array(data)

    # Apply trigonometric function
    result = func(arr)

    # Verify the result has the same dtype as the input
    assert result.as_array().dtype == dtype


def test_trig_identities() -> None:
    """Test fundamental trigonometric identities on slate arrays."""
    rng = np.random.default_rng(42)
    data = rng.random((3, 4)) * 2 * np.pi - np.pi  # Values between -π and π
    arr = Array.from_array(data)

    # Test sin²(x) + cos²(x) = 1
    sin_squared = sin(arr).as_array() ** 2
    cos_squared = cos(arr).as_array() ** 2
    np.testing.assert_allclose(
        sin_squared + cos_squared, np.ones_like(data), rtol=1e-10
    )

    # Test cosh²(x) - sinh²(x) = 1
    sinh_squared = sinh(arr).as_array() ** 2
    cosh_squared = cosh(arr).as_array() ** 2
    np.testing.assert_allclose(
        cosh_squared - sinh_squared, np.ones_like(data), rtol=1e-10
    )
