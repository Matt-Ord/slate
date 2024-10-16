from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Protocol, Self, SupportsIndex

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


class LabelCollection[DT](Protocol):
    def __getitem__(self: Self, i: SupportsIndex, /) -> DT: ...

    def __iter__(self: Self) -> Iterator[DT]: ...


class LabeledMetadata[DT](SimpleMetadata, ABC):
    """A metadata with some data associated to each location."""

    fundamental_shape: tuple[int, ...]

    @property
    @abstractmethod
    def values(self: Self) -> LabelCollection[DT]:
        """Shape of the full data."""


class DeltaMetadata[DT](LabeledMetadata[DT], ABC):
    @property
    @abstractmethod
    def delta(self: Self) -> DT:
        """Shape of the full data."""


@dataclass(frozen=True)
class ExplicitLabeledMetadata[DT: np.generic](LabeledMetadata[DT]):
    """A metadata with some data associated to each location."""

    _values: np.ndarray[Any, np.dtype[DT]]

    @property
    def values(self: Self) -> LabelCollection[DT]:
        """Shape of the full data."""
        return self._values


@dataclass(frozen=True, kw_only=True)
class LabelSpacing:
    start: float = 0
    delta: float


@dataclass(frozen=True, kw_only=True)
class SpacedLabeledMetadata(DeltaMetadata[float]):
    """A metadata with some data associated to each location."""

    spacing: LabelSpacing

    @property
    def values(self: Self) -> LabelCollection[float]:
        """Shape of the full data."""
        return np.linspace(
            self.spacing.start,
            self.spacing.delta + self.spacing.start,
            np.prod(self.fundamental_shape),
        )

    @property
    def delta(self: Self) -> float:
        """Shape of the full data."""
        return self.spacing.delta
