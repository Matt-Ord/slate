from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, override

import numpy as np
from scipy.special import eval_legendre  # type: ignore[import]

from slate_core.metadata._metadata import SpacedMetadata

if TYPE_CHECKING:
    from slate_core.metadata._metadata import LabelSpacing


def _eval_legendre_derivatives(n: int, x: float) -> tuple[float, float, float]:
    """
    Evaluate the first and second derivative of the Legendre polynomial.

    This uses the formula
    .. math::
        P_n'(x) = n * (P_{n-1}(x) - x P_n(x)) / (1 - x^2)

    which when applied recursively gives the second derivative as well:
    .. math::
        P_n''(x) = n * (P_{n-1}'(x) - x P_n'(x)) / (1 - x^2)

    where :math:`P_n(x)` is the Legendre polynomial of degree n.
    """
    p_n, p_n_minus_1 = eval_legendre([n, n - 1], x)

    derivative_1 = n * (p_n_minus_1 - x * p_n) / (1 - x**2)
    derivative_2 = (2 * x * derivative_1 - n * (n + 1) * p_n) / (1.0 - x**2)

    return p_n, derivative_1, derivative_2


def _fundamental_lobatto_data(
    n: int,
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    """
    Calculate lobatto weights based on the fortran approach.

    Translated into python from the code provided by <https://doi.org/10.1007/978-94-015-8240-7_4>

    The n-2 free points are the roots of the n-1 Legendre polynomial P(x_i)
    The n-2 free weights are 2 / (n(n-1) * P(x_i)^2)

    The remaining two points are at the end of the interval (-1, 1)
    with weight 2 / (n(n-1))

    For more details see:
    https://mathworld.wolfram.com/LobattoQuadrature.html.

    """
    weights = np.zeros(n)
    points = np.zeros(n)

    # Setup the endpoints
    points[0], points[-1] = (-1, 1)
    weights[0] = weights[-1] = 2 / (n * (n - 1))
    # For each interior point, upto the midpoint
    for k in range(1, (n + 1) // 2):
        # Initial guess for the root of the Legendre polynomial
        z = np.cos(np.pi * (4 * k + 1) / (4 * n - 2))

        p1 = 1.0
        # Iteratively find the root of the Legendre polynomial
        for _ in range(7):
            p1, p2, p3 = _eval_legendre_derivatives(n - 1, z)
            z -= p2 / p3

        points[k] = -z
        weights[k] = weights[0] / p1**2
        # Using symmetry of the lobatto points we fill the second half
        points[n - (k + 1)] = z
        weights[n - (k + 1)] = weights[k]

    return points, weights


def _get_lobatto_data(
    n: int,
    limits: tuple[float, float] = (-1, 1),
) -> tuple[
    np.ndarray[Any, np.dtype[np.float64]],
    np.ndarray[Any, np.dtype[np.float64]],
]:
    points, weights = _fundamental_lobatto_data(n)

    (a, b) = limits
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)

    return (
        shift + scale * points,
        weights * scale,
    )


@dataclass(frozen=True, kw_only=True)
class LobattoSpacedMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """Metadata defining a basis who's points are lobatto spaced.

    The lobatto points are defined in the range [-1, 1] as the endpoints,
    plus the n-2 roots of the Legendre polynomial of degree n-1.

    This metadata defines a basis with points that are a rescaled version of the lobatto points,
    which is configured using the `spacing` attribute.

    This is particularly useful for systems in which precision is required at the endpoints,
    as the lobatto points have a higher density of points near the endpoints compared to the midpoint.
    """

    spacing: LabelSpacing

    @cached_property
    @override
    def _lobatto_data(
        self,
    ) -> tuple[
        np.ndarray[Any, np.dtype[np.float64]],
        np.ndarray[Any, np.dtype[np.float64]],
    ]:
        """Get the fundamental lobatto points and weights."""
        return _get_lobatto_data(
            self.fundamental_size,
            (self.spacing.start, self.delta + self.spacing.start),
        )

    @property
    @override
    def quadrature_weights(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the weights of the lobatto points."""
        return self._lobatto_data[1]

    @property
    @override
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the points of the lobatto points."""
        return self._lobatto_data[0]

    @property
    @override
    def delta(self) -> float:
        return self.spacing.delta
