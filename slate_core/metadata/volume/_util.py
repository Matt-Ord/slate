from __future__ import annotations

from typing import (
    Any,
    cast,
)

import numpy as np

from slate_core.metadata._metadata import LabelSpacing
from slate_core.metadata.length import SpacedLengthMetadata
from slate_core.metadata.volume._volume import (
    AxisDirections,
    SpacedVolumeMetadata,
    TupleMetadata,
    fundamental_stacked_delta_k,
    fundamental_stacked_k_points,
    fundamental_stacked_x_points,
)
from slate_core.util import (
    get_position_in_sorted,
    slice_ignoring_axes,
)


def _project_directions(
    directions: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    # Perform QR decomposition to get an orthonormal basis of the column space
    _q, r = np.linalg.qr(np.column_stack(directions))
    # We want the first vector to be (1,0,...)
    if r[0, 0] < 0:
        r[0, :] *= -1
    return tuple(r.T)


def _project_points_along_directions(
    points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    directions: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    return tuple(np.tensordot(_project_directions(directions), points, axes=(0, 0)))


def _project_k_points_along_axes(
    points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    metadata: SpacedVolumeMetadata,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the list of k points projected onto the plane including both axes."""
    directions = fundamental_stacked_delta_k(metadata)
    return _project_points_along_directions(
        points, tuple(directions[ax] for ax in axes)
    )


def get_fundamental_stacked_k_points_projected_along_axes(
    metadata: SpacedVolumeMetadata,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental_k_points projected onto the plane including both axes."""
    points = fundamental_stacked_k_points(metadata)
    return _project_k_points_along_axes(points, metadata, axes=axes)


def get_k_coordinates_in_axes(
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
    idx: tuple[int, ...] | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the fundamental_k_points projected onto the plane including both axes."""
    idx = (
        tuple(0 for _ in range(len(metadata.children) - len(axes)))
        if idx is None
        else idx
    )
    points = get_fundamental_stacked_k_points_projected_along_axes(metadata, axes=axes)
    slice_ = slice_ignoring_axes(idx, axes)
    return cast(
        "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
        np.transpose(
            np.asarray(points).reshape(-1, *metadata.fundamental_shape)[:, *slice_],
            (0, *(1 + np.array(get_position_in_sorted(axes)))),
        ),
    )


def project_points_along_axes(
    points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    directions: AxisDirections,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the list of points projected onto the plane including both axes."""
    return _project_points_along_directions(
        points, tuple(directions.vectors[ax] for ax in axes)
    )


def get_fundamental_stacked_x_points_projected_along_axes(
    metadata: SpacedVolumeMetadata,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental_x_points projected onto the plane including both axes."""
    points = fundamental_stacked_x_points(metadata)
    return project_points_along_axes(points, metadata.extra, axes=axes)


def get_x_coordinates_in_axes(
    metadata: SpacedVolumeMetadata,
    axes: tuple[int, ...],
    idx: tuple[int, ...] | None,
) -> np.ndarray[tuple[int, int], np.dtype[np.floating]]:
    """Get the fundamental_x_points projected onto the plane including both axes."""
    idx = (
        tuple(0 for _ in range(len(metadata.children) - len(axes)))
        if idx is None
        else idx
    )
    points = get_fundamental_stacked_x_points_projected_along_axes(metadata, axes=axes)
    slice_ = slice_ignoring_axes(idx, axes)
    return cast(
        "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
        np.transpose(
            np.asarray(points).reshape(-1, *metadata.fundamental_shape)[:, *slice_],
            (0, *(1 + np.array(get_position_in_sorted(axes)))),
        ),
    )


def spaced_volume_metadata_from_stacked_delta_x(
    vectors: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    shape: tuple[int, ...],
) -> SpacedVolumeMetadata:
    """Get the metadata for a spaced volume from the vectors and spacing."""
    delta_v = tuple(np.linalg.norm(v).item() for v in vectors)
    normalized_vectors = tuple(v / dv for v, dv in zip(vectors, delta_v, strict=False))
    return TupleMetadata(
        tuple(
            SpacedLengthMetadata(s, spacing=LabelSpacing(delta=delta))
            for (s, delta) in zip(shape, delta_v, strict=True)
        ),
        AxisDirections(vectors=normalized_vectors),
    )
