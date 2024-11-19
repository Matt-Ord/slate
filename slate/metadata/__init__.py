"""Metadata is used to store information about a basis."""

from __future__ import annotations

from ._metadata import (
    BasisMetadata,
    DeltaMetadata,
    ExplicitLabeledMetadata,
    LabeledMetadata,
    LabelSpacing,
    SimpleMetadata,
    SpacedLabeledMetadata,
)
from .length import LengthMetadata
from .stacked import (
    Metadata1D,
    Metadata2D,
    Metadata3D,
    MetadataND,
    SpacedVolumeMetadata,
    StackedMetadata,
    VolumeMetadata,
)

__all__ = [
    "BasisMetadata",
    "DeltaMetadata",
    "ExplicitLabeledMetadata",
    "LabelSpacing",
    "LabeledMetadata",
    "LengthMetadata",
    "LengthMetadata",
    "Metadata1D",
    "Metadata2D",
    "Metadata3D",
    "MetadataND",
    "SimpleMetadata",
    "SpacedLabeledMetadata",
    "SpacedVolumeMetadata",
    "StackedMetadata",
    "VolumeMetadata",
]
