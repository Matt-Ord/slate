"""Metadata classes for stacked and volume data."""

from __future__ import annotations

from .stacked import StackedMetadata
from .volume import SpacedVolumeMetadata, VolumeMetadata

__all__ = ["SpacedVolumeMetadata", "StackedMetadata", "VolumeMetadata"]
