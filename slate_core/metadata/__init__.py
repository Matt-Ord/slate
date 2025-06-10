"""Metadata is used to store information about a basis."""

from __future__ import annotations

from slate_core.metadata import length, volume
from slate_core.metadata._metadata import (
    PERIODIC_FEATUIRE,
    SIMPLE_FEATURE,
    BasisMetadata,
    ExplicitLabeledMetadata,
    LabeledMetadata,
    SimpleMetadata,
    is_simple_metadata,
)
from slate_core.metadata._shape import (
    NestedLength,
    shallow_shape_from_nested,
    size_from_nested_shape,
)
from slate_core.metadata._spaced import (
    BarycentricMetadata,
    Domain,
    EvenlySpacedMetadata,
    LobattoSpacedMetadata,
    SpacedMetadata,
)
from slate_core.metadata._tuple import AnyMetadata, TupleMetadata, is_tuple_metadata
from slate_core.metadata.length import (
    EvenlySpacedLengthMetadata,
    LengthMetadata,
    LobattoSpacedLengthMetadata,
    fundamental_delta_k,
    fundamental_delta_x,
    fundamental_dk,
    fundamental_dx,
    fundamental_k_points,
)
from slate_core.metadata.util import (
    array_points_from_basis_components,
    basis_components_from_array_points,
    fundamental_nk_points,
    fundamental_nx_points,
    fundamental_size,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)
from slate_core.metadata.volume import (
    AxisDirections,
    EvenlySpacedVolumeMetadata,
    VolumeMetadata,
    spaced_volume_metadata_from_stacked_delta_x,
)

__all__ = [
    "PERIODIC_FEATUIRE",
    "SIMPLE_FEATURE",
    "AnyMetadata",
    "AxisDirections",
    "BarycentricMetadata",
    "BasisMetadata",
    "Domain",
    "EvenlySpacedLengthMetadata",
    "EvenlySpacedMetadata",
    "EvenlySpacedVolumeMetadata",
    "ExplicitLabeledMetadata",
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
    "array_points_from_basis_components",
    "basis_components_from_array_points",
    "fundamental_delta_k",
    "fundamental_delta_x",
    "fundamental_dk",
    "fundamental_dx",
    "fundamental_k_points",
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
