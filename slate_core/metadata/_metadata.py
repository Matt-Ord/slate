from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, override

import numpy as np

if TYPE_CHECKING:
    from slate_core.metadata._shape import NestedLength


class BasisMetadata(Protocol):
    """Protocol that all Metadata should implement."""

    @property
    def fundamental_shape(self) -> NestedLength:
        """Shape of the full data."""
        ...


@dataclass(frozen=True)
class SimpleMetadata(BasisMetadata):
    """Basic Metadata for a Fundamental Basis."""

    fundamental_size: int

    @property
    @override
    def fundamental_shape(self) -> int:
        """Shape of the full data."""
        return self.fundamental_size


def is_simple_metadata(
    metadata: BasisMetadata,
) -> TypeGuard[SimpleMetadata]:
    return isinstance(metadata, SimpleMetadata)


class LabeledMetadata[DT: np.dtype[np.generic]](SimpleMetadata, ABC):
    """A metadata with some data associated to each location."""

    @property
    @abstractmethod
    def values(self) -> np.ndarray[Any, DT]:
        """Shape of the full data."""


class DeltaMetadata[DT: np.dtype[np.generic]](LabeledMetadata[DT], ABC):
    @property
    @abstractmethod
    def delta(self) -> float:
        """Shape of the full data."""


class ExplicitLabeledMetadata[DT: np.dtype[np.generic]](LabeledMetadata[DT]):
    """A metadata with some data associated to each location."""

    def __init__(self, values: np.ndarray[Any, DT]) -> None:
        self._values = values
        super().__init__(values.shape)

    @property
    @override
    def values(self) -> np.ndarray[Any, DT]:
        """Shape of the full data."""
        return self._values


@dataclass(frozen=True, kw_only=True)
class LabelSpacing:
    start: float = 0
    delta: float


@dataclass(frozen=True, kw_only=True)
class SpacedLabeledMetadata(DeltaMetadata[np.dtype[np.floating]]):
    """A metadata with some data associated to each location."""

    spacing: LabelSpacing

    @property
    @override
    def values(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Shape of the full data."""
        return np.linspace(
            self.spacing.start,
            self.spacing.delta + self.spacing.start,
            np.prod(self.fundamental_shape),
        )

    @property
    @override
    def delta(self) -> float:
        """Shape of the full data."""
        return self.spacing.delta
