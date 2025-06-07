from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from slate_core.metadata import LabelSpacing, LobattoSpacedMetadata
from slate_core.metadata.volume._volume import (
    _project_directions,  # noqa: PLC2701 # type: ignore[import-untyped]
)


def test_lobatto_points_known_results() -> None:
    known_results = {
        2: (np.array([-1.0, 1.0]), np.array([1.0, 1.0])),
        3: (np.array([-1.0, 0.0, 1.0]), np.array([1 / 3, 4 / 3, 1 / 3])),
        4: (
            np.array([-1.0, -np.sqrt(1 / 5), np.sqrt(1 / 5), 1.0]),
            np.array([1 / 6, 5 / 6, 5 / 6, 1 / 6]),
        ),
    }

    for n, (expected_points, expected_weights) in known_results.items():
        result = LobattoSpacedMetadata(n, spacing=LabelSpacing(delta=2.0))
        np.testing.assert_allclose(result.values - 1, expected_points, rtol=1e-5)
        np.testing.assert_allclose(
            result.quadrature_weights,
            expected_weights,
            rtol=1e-5,
        )


def test_lobatto_points_symmetry() -> None:
    rng = np.random.default_rng()
    random_n = rng.integers(2, 300).item()

    result = LobattoSpacedMetadata(random_n, spacing=LabelSpacing(delta=2.0))
    np.testing.assert_allclose(
        result.values - 1.0,
        -(result.values[::-1] - 1.0),
        err_msg=f"Points not symmetric for n={random_n}",
        atol=2e-7,
    )
    np.testing.assert_allclose(
        result.quadrature_weights,
        result.quadrature_weights[::-1],
        err_msg=f"Weights not symmetric for n={random_n}",
        atol=1e-10,
    )


def _lobatto_from_fortran(
    a: float,
    b: float,
    n: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """
    Calculate lobatto weights based on the fortran approach.

    Coppied directly from the code provided by <https://doi.org/10.1007/978-94-015-8240-7_4>

    Parameters
    ----------
    a : float
    b : float
    n : int

    Returns
    -------
    tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]

    """
    n_unique = (n + 1) // 2
    pi = np.arccos(-1.0)
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)
    weight = (b - a) / (n * (n - 1))
    w = np.zeros(n)
    x = np.zeros(n)

    x[0] = a
    w[0] = weight

    for k in range(2, n_unique + 1):
        z = np.cos(pi * (4 * k - 3) / (4 * n - 2))
        p2 = 0.0
        p1 = 1.0
        for _ in range(7):
            p2 = 0.0
            p1 = 1.0

            for j in range(1, n):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j

            p2 = (n - 1) * (p2 - z * p1) / (1.0 - z * z)
            p3 = (2 * z * p2 - n * (n - 1) * p1) / (1.0 - z * z)
            z -= p2 / p3

        x[k - 1] = shift - scale * z
        x[n - k] = shift + scale * z
        w[k - 1] = weight / (p1 * p1)
        w[n - k] = w[k - 1]

    x[n - 1] = b
    w[n - 1] = weight

    return x, w


def test_lobatto_points_against_fortran() -> None:
    rng = np.random.default_rng()
    random_n = rng.integers(2, 300).item()

    result = LobattoSpacedMetadata(random_n, spacing=LabelSpacing(delta=1.0))
    fortran_result = _lobatto_from_fortran(0, 1, random_n)
    np.testing.assert_allclose(result.values, fortran_result[0], atol=1e-8)
    np.testing.assert_allclose(result.quadrature_weights, fortran_result[1], atol=1e-8)


def _normalize_directions(
    directions: tuple[np.ndarray[Any, Any], ...],
) -> tuple[np.ndarray[Any, Any], ...]:
    """Normalize the directions to unit length."""
    return tuple(d / np.linalg.norm(d) for d in directions)


@pytest.mark.parametrize(
    ("directions", "expected"),
    [
        (
            (np.array([1, 0]), np.array([0, 1])),
            (np.array([1, 0]), np.array([0, 1])),
        ),
        # 45 degree angle
        ((np.array([1, 0]), np.array([1, 1])), (np.array([1, 0]), np.array([1, 1]))),
        # Arbitrary non-orthogonal 3D
        (
            (np.array([1, 2, 0]), np.array([2, 1, 1])),
            (np.array([1, 0]), np.array([0.73029674, 0.68313005])),
        ),
        # Linearly dependent (should still preserve angle = 1)
        (
            (np.array([1, 1, 5]), np.array([2, 2, 10])),
            (np.array([1, 0]), np.array([1, 0])),
        ),
        # Slightly rotated vectors
        (
            (np.array([1, 0]), np.array([0.99, 0.1])),
            (np.array([1, 0]), np.array([0.99, 0.1])),
        ),
    ],
)
def test_projected(
    directions: tuple[np.ndarray[Any, Any], ...],
    expected: tuple[np.ndarray[Any, Any], ...],
) -> None:
    directions = _normalize_directions(directions)
    result = _project_directions(directions)

    # Check that each direction is normalized
    for res in result:
        assert np.allclose(np.linalg.norm(res), 1.0, atol=1e-10)

    # It should preserve the angles between the directions
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            assert np.isclose(
                np.dot(result[i], result[j]),
                np.dot(directions[i], directions[j]),
                atol=1e-10,
            )

    expected = _normalize_directions(expected)
    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), atol=1e-3)
