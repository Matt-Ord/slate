from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self


class BasisMetadata(Protocol):
    """Protocol that all Metadata should implement."""

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Shape of the full data."""
        ...


@dataclass(frozen=True)
class FundamentalBasisMetadata:
    """Basic Metadata for a Fundamental Basis."""

    fundamental_shape: tuple[int, ...]


@dataclass(frozen=True)
class BasisMetadataWithLength(BasisMetadata):
    """Metadata with the addition of length."""

    length: float
