"""Metadata is used to store information about a basis."""

from __future__ import annotations

from slate.metadata import length, volume
from slate.metadata._metadata import (
    BasisMetadata,
    DeltaMetadata,
    ExplicitLabeledMetadata,
    LabeledMetadata,
    LabelSpacing,
    SimpleMetadata,
    SpacedLabeledMetadata,
)
from slate.metadata._shape import (
    NestedLength,
    shallow_shape_from_nested,
    size_from_nested_shape,
)
from slate.metadata.length import LengthMetadata, SpacedLengthMetadata
from slate.metadata.stacked import (
    AnyMetadata,
    Metadata1D,
    Metadata2D,
    Metadata3D,
    MetadataND,
    StackedMetadata,
)
from slate.metadata.util import (
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
    spaced_volume_metadata_from_stacked_delta_x,
)

__all__ = [
    "AnyMetadata",
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
    "NestedLength",
    "SimpleMetadata",
    "SpacedLabeledMetadata",
    "SpacedLengthMetadata",
    "SpacedVolumeMetadata",
    "StackedMetadata",
    "VolumeMetadata",
    "fundamental_nk_points",
    "fundamental_nx_points",
    "fundamental_size",
    "fundamental_stacked_nk_points",
    "fundamental_stacked_nx_points",
    "length",
    "shallow_shape_from_nested",
    "size_from_nested_shape",
    "spaced_volume_metadata_from_stacked_delta_x",
    "volume",
]
