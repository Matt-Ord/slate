from __future__ import annotations

import numpy as np

from slate.basis import (
    Padding,
    Truncation,
)
from slate.util import pad_along_axis, truncate_along_axis


def test_pad_cropped() -> None:
    data = np.array([1, 2, 3, 4, 5])
    rng = np.random.default_rng()
    n = rng.integers(data.size, 2 * data.size)
    padded = pad_along_axis(data, Padding(n, 1, 0))
    expected = np.zeros(n)
    expected[: data.size] = data
    np.testing.assert_array_equal(padded, expected)


def test_pad_with_step() -> None:
    data = np.array([1, 2, 3, 4, 5])

    padded = pad_along_axis(data, Padding(10, 2, 0))
    np.testing.assert_array_equal(padded, [1, 0, 2, 0, 3, 0, 4, 0, 5, 0])
    truncated = truncate_along_axis(padded, Truncation(5, 2, 0))
    np.testing.assert_array_equal(data, truncated)

    padded = pad_along_axis(data, Padding(10, 2, 1))
    np.testing.assert_array_equal(padded, [0, 1, 0, 2, 0, 3, 0, 4, 0, 5])
    truncated = truncate_along_axis(padded, Truncation(5, 2, 1))
    np.testing.assert_array_equal(data, truncated)

    padded = pad_along_axis(data, Padding(10, 3, 1))
    np.testing.assert_array_equal(padded, [4, 1, 0, 5, 2, 0, 0, 3, 0, 0])
    truncated = truncate_along_axis(padded, Truncation(5, 3, 1))
    np.testing.assert_array_equal(data, truncated)

    padded = pad_along_axis(data, Padding(15, 3, 1))
    np.testing.assert_array_equal(padded, [0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0])
    truncated = truncate_along_axis(padded, Truncation(5, 3, 1))
    np.testing.assert_array_equal(data, truncated)

    padded = pad_along_axis(data, Padding(10, 1, 1))
    np.testing.assert_array_equal(padded, [0, 1, 2, 3, 4, 5, 0, 0, 0, 0])
    truncated = truncate_along_axis(padded, Truncation(5, 1, 1))
    np.testing.assert_array_equal(data, truncated)


def test_truncate_cropped() -> None:
    data = np.array([1, 2, 3, 4, 5])
    rng = np.random.default_rng()
    n = rng.integers(0, data.size)
    padded = truncate_along_axis(data, Truncation(n, 1, 0))
    expected = data[:n]
    np.testing.assert_array_equal(padded, expected)


def test_truncate_with_step() -> None:
    data = np.array([1, 2, 3, 4, 5])

    padded = truncate_along_axis(data, Truncation(5, 4, 0))
    np.testing.assert_array_equal(padded, [1, 5, 4, 3, 2])

    padded = truncate_along_axis(data, Truncation(5, 4, 1))
    np.testing.assert_array_equal(padded, [2, 1, 5, 4, 3])

    padded = truncate_along_axis(data, Truncation(5, 4, 2))
    np.testing.assert_array_equal(padded, [3, 2, 1, 5, 4])

    padded = truncate_along_axis(data, Truncation(5, 4, -1))
    np.testing.assert_array_equal(padded, [5, 4, 3, 2, 1])

    padded = truncate_along_axis(data, Truncation(5, 4, -2))
    np.testing.assert_array_equal(padded, [4, 3, 2, 1, 5])
