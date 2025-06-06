from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, TypeGuard, override

import numpy as np

if TYPE_CHECKING:
    from slate_core.metadata._shape import NestedLength


class BasisMetadata(Protocol):
    """Protocol that all Metadata should implement."""

    @property
    def is_periodic(self) -> bool:
        """Whether the metadata is periodic."""
        ...

    @property
    def fundamental_shape(self) -> NestedLength:
        """Shape of the full data."""
        ...


@dataclass(frozen=True)
class SimpleMetadata(BasisMetadata):
    """Basic Metadata for a Fundamental Basis."""

    fundamental_size: int
    is_periodic: bool = field(default=False, kw_only=True)  # type: ignore[assignment]

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

    @property
    def unit(self) -> str:
        """Unit value for the metadata."""
        return "a.u."


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


class SpacedMetadata[DT: np.dtype[np.generic]](LabeledMetadata[DT], ABC):
    @property
    @abstractmethod
    def delta(self) -> float:
        """Shape of the full data."""

    @property
    @abstractmethod
    def quadrature_weights(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Quadrature weights for the metadata."""


@dataclass(frozen=True, kw_only=True)
class LabelSpacing:
    start: float = 0
    delta: float


@dataclass(frozen=True, kw_only=True)
class EvenlySpacedMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """A metadata with some data associated to each location."""

    spacing: LabelSpacing

    @property
    @override
    def values(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Shape of the full data."""
        return np.linspace(
            self.spacing.start,
            self.spacing.delta + self.spacing.start,
            self.fundamental_size,
            endpoint=False,
        )

    @property
    @override
    def quadrature_weights(self) -> np.ndarray[Any, np.dtype[np.floating]]:
        """Quadrature weights for the metadata.

        An evenly spaced metadata has a unit weight for each point.
        """
        return np.ones(self.fundamental_size)

    @property
    @override
    def delta(self) -> float:
        """Shape of the full data."""
        return self.spacing.delta
