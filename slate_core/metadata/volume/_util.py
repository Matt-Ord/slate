from __future__ import annotations

from typing import (
    Any,
    Literal,
    cast,
)

import numpy as np

from slate_core.metadata._spaced import Domain
from slate_core.metadata.length import EvenlySpacedLengthMetadata
from slate_core.metadata.volume._volume import (
    AxisDirections,
    EvenlySpacedVolumeMetadata,
    TupleMetadata,
    fundamental_stacked_delta_k,
    fundamental_stacked_k_points,
    fundamental_stacked_x_points,
    project_points_along_directions,
)
from slate_core.util import (
    get_position_in_sorted,
    slice_ignoring_axes,
)


def _project_k_points_along_axes(
    points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    metadata: EvenlySpacedVolumeMetadata,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the list of k points projected onto the plane including both axes."""
    directions = fundamental_stacked_delta_k(metadata)
    return project_points_along_directions(points, tuple(directions[ax] for ax in axes))


def get_fundamental_stacked_k_points_projected_along_axes(
    metadata: EvenlySpacedVolumeMetadata,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental_k_points projected onto the plane including both axes."""
    points = fundamental_stacked_k_points(metadata)
    return _project_k_points_along_axes(points, metadata, axes=axes)


def get_k_coordinates_in_axes(
    metadata: EvenlySpacedVolumeMetadata,
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
    return project_points_along_directions(
        points, tuple(directions.vectors[ax] for ax in axes)
    )


def get_fundamental_stacked_x_points_projected_along_axes(
    metadata: EvenlySpacedVolumeMetadata,
    *,
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental_x_points projected onto the plane including both axes."""
    points = fundamental_stacked_x_points(metadata)
    return project_points_along_axes(points, metadata.extra, axes=axes)


def get_x_coordinates_in_axes(
    metadata: EvenlySpacedVolumeMetadata,
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
    *,
    interpolation: tuple[Literal["Fourier", "DST"]] | None = None,
) -> EvenlySpacedVolumeMetadata:
    """Get the metadata for a spaced volume from the vectors and spacing."""
    delta_v = tuple(np.linalg.norm(v).item() for v in vectors)
    normalized_vectors = tuple(v / dv for v, dv in zip(vectors, delta_v, strict=False))
    interpolation = (
        tuple[Literal["Fourier"]]("Fourier" for _ in shape)
        if interpolation is None
        else interpolation
    )
    return TupleMetadata(
        tuple(
            EvenlySpacedLengthMetadata(s, domain=Domain(delta=delta), interpolation=i)
            for (s, delta, i) in zip(shape, delta_v, interpolation, strict=True)
        ),
        AxisDirections(vectors=normalized_vectors),
    )
