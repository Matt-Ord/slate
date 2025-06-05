from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from slate_core.metadata.volume._volume import (
    _project_directions,  # noqa: PLC2701 # type: ignore[import-untyped]
)


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
