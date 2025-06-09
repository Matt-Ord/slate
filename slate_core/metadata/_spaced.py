from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, override

import numpy as np
from scipy.special import eval_legendre  # type: ignore[import-untyped]

from slate_core.metadata._metadata import (
    BARICENTRIC_FEATURE,
    PERIODIC_FEATUIRE,
    SIMPLE_FEATURE,
    LabeledMetadata,
)


@dataclass(frozen=True, kw_only=True)
class Domain:
    start: float = 0
    delta: float

    @property
    def end(self) -> float:
        """End of the domain."""
        return self.start + self.delta


@dataclass(frozen=True, kw_only=True)
class SpacedMetadata[DT: np.dtype[np.generic]](LabeledMetadata[DT], ABC):
    domain: Domain

    @property
    def delta(self) -> float:
        """Shape of the full data."""
        return self.domain.delta

    @property
    @override
    @abstractmethod
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]: ...

    @property
    @override
    @abstractmethod
    def features(self) -> set[str]: ...


@dataclass(frozen=True, kw_only=True)
class EvenlySpacedMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """A metadata with some data associated to each location."""

    interpolation: Literal["Fourier", "DST"] = "Fourier"

    @property
    @override
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        """Shape of the full data."""
        return np.linspace(
            self.domain.start,
            self.domain.end,
            self.fundamental_size,
            endpoint=self.interpolation != "Fourier",
        ).reshape(-1)

    @property
    @override
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.floating]]:
        return np.ones(self.fundamental_size)

    @property
    @override
    def features(self) -> set[str]:
        """Shape of the full data."""
        return (
            {SIMPLE_FEATURE, PERIODIC_FEATUIRE}
            if self.interpolation == "Fourier"
            else {SIMPLE_FEATURE}
        )


def _get_barycentric_basis_weights(
    domain: Domain,
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    r"""Calculate stable barycentric weights for a set of interpolation points."""
    scaled_values = values - domain.start / domain.delta
    difference = scaled_values[:, None] - scaled_values[None, :]
    np.fill_diagonal(difference, 1)

    # the log of the barycentric weights
    # λ_i = 1 / Π_{k ≠ i} (x_i - x_k)
    # For numerical stability, we subtract the maximum log weight
    # to avoid overflow when exponentiating. This scales
    # the weights but does not change the output.
    log_abs_diff = np.log(np.abs(difference))
    log_weights = -np.sum(log_abs_diff, axis=1)
    log_weights -= np.max(log_weights)
    # Exponentiate to get the barycentric weights
    # multiplied by a prefactor
    barycentric_weights = np.exp(log_weights)

    return np.sqrt(np.sum(barycentric_weights**2) / domain.delta) / barycentric_weights


class BarycentricMetadata(SpacedMetadata[np.dtype[np.floating]]):
    def __init__(
        self,
        fundamental_size: int,
        domain: Domain,
        values: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Initialize the lobatto spaced metadata."""
        self._values: np.ndarray[tuple[int], np.dtype[np.float64]] = values
        super().__init__(fundamental_size=fundamental_size, domain=domain)

    @property
    @override
    def features(self) -> set[str]:
        """Get the features of the barycentric points."""
        return {BARICENTRIC_FEATURE}

    _basis_weights: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None

    @property
    @override
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        if self._basis_weights is None:
            self._basis_weights = _get_barycentric_basis_weights(
                self.domain, self._values
            )
        return self._basis_weights

    @property
    @override
    def values(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Get the points of the lobatto points."""
        return self._values


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
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
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
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int], np.dtype[np.float64]],
]:
    points, weights = _fundamental_lobatto_data(n)

    (a, b) = limits
    shift = 0.5 * (b + a)
    scale = 0.5 * (b - a)

    return (
        (shift + scale * points).ravel(),
        (weights * scale).ravel(),
    )


class LobattoSpacedMetadata(BarycentricMetadata):
    """Metadata defining a basis who's points are lobatto spaced.

    The lobatto points are defined in the range [-1, 1] as the endpoints,
    plus the n-2 roots of the Legendre polynomial of degree n-1.

    This metadata defines a basis with points that are a rescaled version of the lobatto points,
    which is configured using the `spacing` attribute.

    This is particularly useful for systems in which precision is required at the endpoints,
    as the lobatto points have a higher density of points near the endpoints compared to the midpoint.
    """

    _basis_weights: np.ndarray[tuple[int], np.dtype[np.float64]]

    def __init__(self, fundamental_size: int, *, domain: Domain) -> None:
        """Initialize the lobatto spaced metadata."""
        values, weights = _get_lobatto_data(
            fundamental_size,
            (domain.start, domain.end),
        )
        self._basis_weights = (1 / np.sqrt(weights)).ravel()  # type: ignore[assignment]
        super().__init__(
            fundamental_size=fundamental_size, domain=domain, values=values
        )

    @property
    @override
    def basis_weights(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._basis_weights
