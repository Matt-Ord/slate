"""Module for stacked volume metadata."""

from __future__ import annotations

from ._volume import (
    AxisDirections,
    SpacedVolumeMetadata,
    VolumeMetadata,
    fundamental_reciprocal_volume,
    fundamental_stacked_delta_k,
    fundamental_stacked_delta_x,
    fundamental_stacked_dk,
    fundamental_stacked_dx,
    fundamental_stacked_k_points,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
    fundamental_stacked_x_points,
    fundamental_volume,
)
from .util import (
    get_fundamental_stacked_k_points_projected_along_axes,
    get_fundamental_stacked_x_points_projected_along_axes,
    get_k_coordinates_in_axes,
    get_x_coordinates_in_axes,
    spaced_volume_metadata_from_stacked_delta_x,
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
    "fundamental_stacked_nk_points",
    "fundamental_stacked_nx_points",
    "fundamental_stacked_x_points",
    "fundamental_volume",
    "get_fundamental_stacked_k_points_projected_along_axes",
    "get_fundamental_stacked_x_points_projected_along_axes",
    "get_k_coordinates_in_axes",
    "get_x_coordinates_in_axes",
    "spaced_volume_metadata_from_stacked_delta_x",
]
