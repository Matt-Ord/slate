"""Metadata classes for stacked and volume data."""

from __future__ import annotations

from .stacked import Metadata1D, Metadata2D, Metadata3D, MetadataND, StackedMetadata
from .volume import (
    SpacedVolumeMetadata,
    VolumeMetadata,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)

__all__ = [
    "Metadata1D",
    "Metadata2D",
    "Metadata3D",
    "MetadataND",
    "SpacedVolumeMetadata",
    "StackedMetadata",
    "VolumeMetadata",
    "fundamental_stacked_nk_points",
    "fundamental_stacked_nx_points",
]
