from __future__ import annotations

from typing import Any, override

import numpy as np

from slate_core.metadata._lobatto import LobattoSpacedMetadata
from slate_core.metadata._metadata import (
    EvenlySpacedMetadata,
    SpacedMetadata,
)
from slate_core.metadata.util import (
    fundamental_nk_points,
    fundamental_nx_points,
    fundamental_size,
)


class LengthMetadata(SpacedMetadata[np.dtype[np.floating]]):
    """Metadata with the addition of length."""

    @property
    @override
    def unit(self) -> str:
        return "m"


class EvenlySpacedLengthMetadata(EvenlySpacedMetadata, LengthMetadata):
    """Metadata with the addition of length."""


class LobattoSpacedLengthMetadata(LobattoSpacedMetadata, LengthMetadata):
    """Metadata with the addition of length."""


def fundamental_delta_x(metadata: LengthMetadata) -> float:
    """Get the fundamental delta x."""
    return metadata.delta


def fundamental_dx(metadata: EvenlySpacedLengthMetadata) -> float:
    """Get the fundamental dx."""
    return fundamental_delta_x(metadata) / fundamental_size(metadata)


def fundamental_dk(metadata: EvenlySpacedLengthMetadata) -> float:
    """Get the fundamental dk."""
    return 2 * np.pi / fundamental_delta_x(metadata)


def fundamental_delta_k(metadata: EvenlySpacedLengthMetadata) -> float:
    """Get the fundamental delta k."""
    return fundamental_size(metadata) * fundamental_dk(metadata)


def fundamental_x_points(
    metadata: EvenlySpacedLengthMetadata,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the coordinates, using the x convention (0...N)."""
    return fundamental_delta_x(metadata) * fundamental_nx_points(metadata)


def fundamental_k_points(
    metadata: EvenlySpacedLengthMetadata,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Get the coordinates, using the kx convention (0...N/2-N/2...)."""
    return fundamental_dk(metadata) * fundamental_nk_points(metadata)
