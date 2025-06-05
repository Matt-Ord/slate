"""Module for stacked volume metadata."""

from __future__ import annotations

from slate_core.metadata.volume._util import (
    get_fundamental_stacked_k_points_projected_along_axes,
    get_fundamental_stacked_x_points_projected_along_axes,
    get_k_coordinates_in_axes,
    get_x_coordinates_in_axes,
    project_points_along_axes,
    spaced_volume_metadata_from_stacked_delta_x,
)
from slate_core.metadata.volume._volume import (
    AxisDirections,
    SpacedVolumeMetadata,
    VolumeMetadata,
    fundamental_reciprocal_volume,
    fundamental_stacked_delta_k,
    fundamental_stacked_delta_x,
    fundamental_stacked_dk,
    fundamental_stacked_dx,
    fundamental_stacked_k_points,
    fundamental_stacked_x_points,
    fundamental_volume,
)

__all__ = [
    "AxisDirections",
    "SpacedVolumeMetadata",
    "VolumeMetadata",
    "fundamental_reciprocal_volume",
    "fundamental_stacked_delta_k",
    "fundamental_stacked_delta_x",
    "fundamental_stacked_dk",
    "fundamental_stacked_dx",
    "fundamental_stacked_k_points",
    "fundamental_stacked_x_points",
    "fundamental_volume",
    "get_fundamental_stacked_k_points_projected_along_axes",
    "get_fundamental_stacked_x_points_projected_along_axes",
    "get_k_coordinates_in_axes",
    "get_x_coordinates_in_axes",
    "project_points_along_axes",
    "spaced_volume_metadata_from_stacked_delta_x",
]
