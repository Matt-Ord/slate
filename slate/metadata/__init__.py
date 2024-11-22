"""Metadata is used to store information about a basis."""

from __future__ import annotations

from slate.metadata._metadata import (
    BasisMetadata,
    DeltaMetadata,
    ExplicitLabeledMetadata,
    LabeledMetadata,
    LabelSpacing,
    SimpleMetadata,
    SpacedLabeledMetadata,
)
from slate.metadata.length import LengthMetadata, SpacedLengthMetadata
from slate.metadata.stacked import (
    Metadata1D,
    Metadata2D,
    Metadata3D,
    MetadataND,
    StackedMetadata,
)
from slate.metadata.util import (
    fundamental_ndim,
    fundamental_nk_points,
    fundamental_nx_points,
    fundamental_size,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)
from slate.metadata.volume import (
    AxisDirections,
    SpacedVolumeMetadata,
    VolumeMetadata,
)

__all__ = [
    "AxisDirections",
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
    "SpacedLengthMetadata",
    "SpacedVolumeMetadata",
    "StackedMetadata",
    "VolumeMetadata",
    "fundamental_ndim",
    "fundamental_nk_points",
    "fundamental_nx_points",
    "fundamental_size",
    "fundamental_stacked_nk_points",
    "fundamental_stacked_nx_points",
]
