"""Metadata is used to store information about a basis."""

from __future__ import annotations

from slate_core.metadata import length, volume
from slate_core.metadata._metadata import (
    BasisMetadata,
    DeltaMetadata,
    ExplicitLabeledMetadata,
    LabeledMetadata,
    LabelSpacing,
    SimpleMetadata,
    SpacedLabeledMetadata,
    is_simple_metadata,
)
from slate_core.metadata._shape import (
    NestedLength,
    shallow_shape_from_nested,
    size_from_nested_shape,
)
from slate_core.metadata._stacked import AnyMetadata, TupleMetadata, is_tuple_metadata
from slate_core.metadata.length import LengthMetadata, SpacedLengthMetadata
from slate_core.metadata.util import (
    fundamental_nk_points,
    fundamental_nx_points,
    fundamental_size,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)
from slate_core.metadata.volume import (
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
    "NestedLength",
    "SimpleMetadata",
    "SpacedLabeledMetadata",
    "SpacedLengthMetadata",
    "SpacedVolumeMetadata",
    "TupleMetadata",
    "VolumeMetadata",
    "fundamental_nk_points",
    "fundamental_nx_points",
    "fundamental_size",
    "fundamental_stacked_nk_points",
    "fundamental_stacked_nx_points",
    "is_simple_metadata",
    "is_tuple_metadata",
    "length",
    "shallow_shape_from_nested",
    "size_from_nested_shape",
    "spaced_volume_metadata_from_stacked_delta_x",
    "volume",
]
