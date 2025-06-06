from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import Any, cast, override

import numpy as np

from slate_core.metadata._shape import shallow_shape_from_nested
from slate_core.metadata._tuple import TupleMetadata
from slate_core.metadata.length import EvenlySpacedLengthMetadata, LengthMetadata
from slate_core.metadata.util import (
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)


def _assert_normal(
    vectors: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
) -> None:
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1),
        1,
        err_msg="The provided axis vectors are not normal.",
        atol=1e-10,
    )


@dataclass(frozen=True, kw_only=True)
class AxisDirections:
    """Data to store the axis vectors of an array."""

    vectors: tuple[np.ndarray[Any, np.dtype[np.floating]], ...]

    def __post_init__(self) -> None:
        _assert_normal(self.vectors)

    @override
    def __eq__(self, value: object) -> bool:
        if isinstance(value, AxisDirections):
            return all(
                starmap(np.allclose, zip(self.vectors, value.vectors, strict=False))
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash(tuple(tuple(x) for x in self.vectors))


type VolumeMetadata = TupleMetadata[tuple[LengthMetadata, ...], AxisDirections]
type EvenlySpacedVolumeMetadata = TupleMetadata[
    tuple[EvenlySpacedLengthMetadata, ...], AxisDirections
]


def fundamental_stacked_delta_x(
    metadata: VolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental stacked delta x."""
    scaled = cast(
        "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
        np.einsum(  # type: ignore unknown
            "ij,i->ij", metadata.extra.vectors, [c.delta for c in metadata.children]
        ),
    )
    return tuple(scaled)


def fundamental_stacked_dx(
    metadata: EvenlySpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental stacked dx."""
    scaled = cast(
        "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
        np.einsum(  # type: ignore unknown
            "ij,i->ij",
            fundamental_stacked_delta_x(metadata),
            [1 / s for s in shallow_shape_from_nested(metadata.fundamental_shape)],
        ),
    )
    return tuple(scaled)


def fundamental_stacked_dk(
    metadata: EvenlySpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental stacked dk."""
    return tuple(2 * np.pi * np.linalg.inv(fundamental_stacked_delta_x(metadata)).T)


def fundamental_stacked_delta_k(
    metadata: EvenlySpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental stacked delta k."""
    scaled = cast(
        "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
        np.einsum(  # type: ignore unknown
            "ij,i->ij", fundamental_stacked_dk(metadata), metadata.fundamental_shape
        ),
    )
    return tuple(scaled)


def fundamental_volume(metadata: VolumeMetadata) -> float:
    """Get the fundamental volume."""
    return np.linalg.det(fundamental_stacked_delta_x(metadata))


def fundamental_reciprocal_volume(
    metadata: EvenlySpacedVolumeMetadata,
) -> float:
    """Get the fundamental reciprocal volume."""
    return np.linalg.det(fundamental_stacked_delta_k(metadata))


def _project_directions(
    directions: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    # Perform QR decomposition to get an orthonormal basis of the column space
    _q, r = np.linalg.qr(np.column_stack(directions))
    # We want the first vector to be (1,0,...)
    if r[0, 0] < 0:
        r[0, :] *= -1
    return tuple(r.T)


def project_directions_onto_axes(
    directions: AxisDirections, axes: tuple[int, ...]
) -> AxisDirections:
    """Project the AxisDirections onto the specified axes."""
    projected = _project_directions(tuple(directions.vectors[ax] for ax in axes))
    return AxisDirections(vectors=projected)


def project_points_along_directions(
    points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    directions: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    return tuple(np.tensordot(_project_directions(directions), points, axes=(0, 0)))


def _wrap_and_offset(
    points: tuple[np.ndarray[Any, np.dtype[np.floating]], ...],
    delta: tuple[float, ...],
    *,
    offset: tuple[float, ...] | None = None,
    wrapped: bool = False,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    if offset is not None:
        points = tuple(p + o for p, o in zip(points, offset, strict=False))

    if wrapped:
        points = tuple(
            (p + (d / 2)) % d - (d / 2) for p, d in zip(points, delta, strict=False)
        )
    return points


def _fundamental_stacked_x_points_evenly_spaced(
    metadata: EvenlySpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    return tuple(
        cast(
            "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
            np.einsum(  # type: ignore unknown
                "ij,ik->jk",
                fundamental_stacked_dx(metadata),
                fundamental_stacked_nx_points(metadata),
            ),
        )
    )


def _fundamental_stacked_x_points_generic(
    metadata: VolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    return project_points_along_directions(
        np.meshgrid(*(c.values for c in metadata.children)), metadata.extra.vectors
    )


def fundamental_stacked_x_points(
    metadata: VolumeMetadata,
    *,
    offset: tuple[float, ...] | None = None,
    wrapped: bool = False,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the stacked coordinates, using the x convention (0...N)."""
    if all(isinstance(c, EvenlySpacedLengthMetadata) for c in metadata.children):
        points = _fundamental_stacked_x_points_evenly_spaced(
            cast("EvenlySpacedVolumeMetadata", metadata)
        )
    else:
        points = _fundamental_stacked_x_points_generic(metadata)
    delta = tuple(np.linalg.norm(fundamental_stacked_delta_x(metadata), axis=1))
    return _wrap_and_offset(points, delta, offset=offset, wrapped=wrapped)


def fundamental_stacked_k_points(
    metadata: EvenlySpacedVolumeMetadata,
    *,
    offset: tuple[float, ...] | None = None,
    wrapped: bool = False,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the stacked coordinates, using the kx convention (0...N/2-N/2...)."""
    points = tuple(
        cast(
            "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
            np.einsum(  # type: ignore unknown
                "ij,ik->jk",
                fundamental_stacked_dk(metadata),
                fundamental_stacked_nk_points(metadata),
            ),
        )
    )
    delta = tuple(np.linalg.norm(fundamental_stacked_delta_k(metadata), axis=1))
    return _wrap_and_offset(points, delta, offset=offset, wrapped=wrapped)
