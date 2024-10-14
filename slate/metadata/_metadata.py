from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np


class BasisMetadata(Protocol):
    """Protocol that all Metadata should implement."""

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Shape of the full data."""
        ...


@dataclass(frozen=True)
class SimpleMetadata:
    """Basic Metadata for a Fundamental Basis."""

    fundamental_shape: tuple[int, ...]


class LabeledMetadata[DT: np.generic](SimpleMetadata, ABC):
    """A metadata with some data associated to each location."""

    fundamental_shape: tuple[int, ...]

    @property
    @abstractmethod
    def values(self: Self) -> np.ndarray[Any, np.dtype[DT]]:
        """Shape of the full data."""

    @property
    def delta(self: Self) -> DT:
        """Shape of the full data."""
        return self.values[-1] - self.values[0]


@dataclass(frozen=True)
class ExplicitLabeledMetadata[DT: np.generic](LabeledMetadata[DT]):
    """A metadata with some data associated to each location."""

    _values: np.ndarray[Any, np.dtype[DT]]

    @property
    def values(self: Self) -> np.ndarray[Any, np.dtype[DT]]:
        """Shape of the full data."""
        return self._values


@dataclass(frozen=True, kw_only=True)
class LabelSpacing:
    start: float = 0
    delta: float


@dataclass(frozen=True, kw_only=True)
class SpacedLabeledMetadata(LabeledMetadata[np.float64]):
    """A metadata with some data associated to each location."""

    spacing: LabelSpacing

    @property
    def values(self: Self) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Shape of the full data."""
        return np.linspace(
            self.spacing.start,
            self.spacing.delta + self.spacing.start,
            np.prod(self.fundamental_shape),
        )

    @property
    def delta(self: Self) -> np.float64:
        """Shape of the full data."""
        return np.floating(self.spacing.delta)
