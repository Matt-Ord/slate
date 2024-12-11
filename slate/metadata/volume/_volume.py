from __future__ import annotations

from dataclasses import dataclass
from itertools import starmap
from typing import Any, cast, override

import numpy as np

from slate.metadata._shape import shallow_shape_from_nested
from slate.metadata.length import LengthMetadata, SpacedLengthMetadata
from slate.metadata.stacked import StackedMetadata
from slate.metadata.util import (
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)


def _assert_orthonormal(
    vectors: tuple[np.ndarray[Any, np.dtype[np.float64]], ...],
) -> None:
    identity_matrix = np.eye(len(vectors))
    result = np.dot(vectors, np.transpose(vectors))

    np.testing.assert_allclose(
        result,
        identity_matrix,
        err_msg="The provided axis vectors are not orthonormal.",
    )


@dataclass(frozen=True, kw_only=True)
class AxisDirections:
    """Data to store the axis vectors of an array."""

    vectors: tuple[np.ndarray[Any, np.dtype[np.float64]], ...]

    def __post_init__(self) -> None:
        _assert_orthonormal(self.vectors)

    @override
    def __eq__(self, value: object) -> bool:
        if isinstance(value, AxisDirections):
            return all(starmap(np.allclose, zip(self.vectors, value.vectors)))
        return False

    @override
    def __hash__(self) -> int:
        return hash(tuple(tuple(x) for x in self.vectors))


type VolumeMetadata = StackedMetadata[LengthMetadata, AxisDirections]
type SpacedVolumeMetadata = StackedMetadata[SpacedLengthMetadata, AxisDirections]


def fundamental_stacked_delta_x(
    metadata: VolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked delta x."""
    scaled = cast(
        "np.ndarray[Any, np.dtype[np.float64]]",
        np.einsum(  # type: ignore unknown
            "ij,i->ij", metadata.extra.vectors, [c.delta for c in metadata.children]
        ),
    )
    return tuple(scaled)


def fundamental_stacked_dx(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked dx."""
    scaled = cast(
        "np.ndarray[Any, np.dtype[np.float64]]",
        np.einsum(  # type: ignore unknown
            "ij,i->ij",
            fundamental_stacked_delta_x(metadata),
            [1 / s for s in shallow_shape_from_nested(metadata.fundamental_shape)],
        ),
    )
    return tuple(scaled)


def fundamental_stacked_dk(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked dk."""
    return tuple(2 * np.pi * np.linalg.inv(fundamental_stacked_delta_x(metadata)).T)


def fundamental_stacked_delta_k(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked delta k."""
    scaled = cast(
        "np.ndarray[Any, np.dtype[np.float64]]",
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


def fundamental_stacked_x_points(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the stacked coordinates, using the x convention (0...N)."""
    scaled = cast(
        "np.ndarray[Any, np.dtype[np.float64]]",
        np.einsum(  # type: ignore unknown
            "ij,ik->jk",
            fundamental_stacked_dx(metadata),
            fundamental_stacked_nx_points(metadata),
        ),
    )
    return tuple(scaled)


def fundamental_stacked_k_points(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the stacked coordinates, using the kx convention (0...N/2-N/2...)."""
    scaled = cast(
        "np.ndarray[Any, np.dtype[np.float64]]",
        np.einsum(  # type: ignore unknown
            "ij,ik->jk",
            fundamental_stacked_dk(metadata),
            fundamental_stacked_nk_points(metadata),
        ),
    )
    return tuple(scaled)
