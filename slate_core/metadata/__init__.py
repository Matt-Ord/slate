"""Metadata is used to store information about a basis."""

from __future__ import annotations

from slate_core.metadata import length, volume
from slate_core.metadata._lobatto import LobattoSpacedMetadata
from slate_core.metadata._metadata import (
    BasisMetadata,
    EvenlySpacedMetadata,
    ExplicitLabeledMetadata,
    LabeledMetadata,
    LabelSpacing,
    SimpleMetadata,
    SpacedMetadata,
    is_simple_metadata,
)
from slate_core.metadata._shape import (
    NestedLength,
    shallow_shape_from_nested,
    size_from_nested_shape,
)
from slate_core.metadata._tuple import AnyMetadata, TupleMetadata, is_tuple_metadata
from slate_core.metadata.length import (
    EvenlySpacedLengthMetadata,
    LengthMetadata,
    LobattoSpacedLengthMetadata,
)
from slate_core.metadata.util import (
    fundamental_nk_points,
    fundamental_nx_points,
    fundamental_size,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
    fundamental_stacked_weights,
)
from slate_core.metadata.volume import (
    AxisDirections,
    EvenlySpacedVolumeMetadata,
    VolumeMetadata,
    spaced_volume_metadata_from_stacked_delta_x,
)

__all__ = [
    "AnyMetadata",
    "AxisDirections",
    "BasisMetadata",
    "EvenlySpacedLengthMetadata",
    "EvenlySpacedMetadata",
    "EvenlySpacedVolumeMetadata",
    "ExplicitLabeledMetadata",
    "LabelSpacing",
    "LabeledMetadata",
    "LengthMetadata",
    "LengthMetadata",
    "LobattoSpacedLengthMetadata",
    "LobattoSpacedMetadata",
    "NestedLength",
    "SimpleMetadata",
    "SpacedMetadata",
    "TupleMetadata",
    "VolumeMetadata",
    "fundamental_nk_points",
    "fundamental_nx_points",
    "fundamental_size",
    "fundamental_stacked_nk_points",
    "fundamental_stacked_nx_points",
    "fundamental_stacked_weights",
    "is_simple_metadata",
    "is_tuple_metadata",
    "length",
    "shallow_shape_from_nested",
    "size_from_nested_shape",
    "spaced_volume_metadata_from_stacked_delta_x",
    "volume",
]
