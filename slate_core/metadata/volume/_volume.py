from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import Any, cast, override

import numpy as np

from slate_core.metadata._shape import shallow_shape_from_nested
from slate_core.metadata._tuple import TupleMetadata
from slate_core.metadata.length import LengthMetadata, SpacedLengthMetadata
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
type SpacedVolumeMetadata = TupleMetadata[
    tuple[SpacedLengthMetadata, ...], AxisDirections
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
    metadata: SpacedVolumeMetadata,
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
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the fundamental stacked dk."""
    return tuple(2 * np.pi * np.linalg.inv(fundamental_stacked_delta_x(metadata)).T)


def fundamental_stacked_delta_k(
    metadata: SpacedVolumeMetadata,
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
    metadata: SpacedVolumeMetadata,
) -> float:
    """Get the fundamental reciprocal volume."""
    return np.linalg.det(fundamental_stacked_delta_k(metadata))


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


def fundamental_stacked_x_points(
    metadata: SpacedVolumeMetadata,
    *,
    offset: tuple[float, ...] | None = None,
    wrapped: bool = False,
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the stacked coordinates, using the x convention (0...N)."""
    points = tuple(
        cast(
            "np.ndarray[tuple[int, int], np.dtype[np.floating]]",
            np.einsum(  # type: ignore unknown
                "ij,ik->jk",
                fundamental_stacked_dx(metadata),
                fundamental_stacked_nx_points(metadata),
            ),
        )
    )
    delta = tuple(np.linalg.norm(fundamental_stacked_delta_x(metadata), axis=1))
    return _wrap_and_offset(points, delta, offset=offset, wrapped=wrapped)


def fundamental_stacked_k_points(
    metadata: SpacedVolumeMetadata,
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
