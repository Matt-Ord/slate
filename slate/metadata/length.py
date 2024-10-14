from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from slate.metadata._metadata import LabeledMetadata, SpacedLabeledMetadata
from slate.metadata.stacked import StackedMetadata
from slate.metadata.util import (
    fundamental_nk_points,
    fundamental_size,
    fundamental_stacked_nk_points,
    fundamental_stacked_nx_points,
)


class LengthMetadata(LabeledMetadata[np.float64]):
    """Metadata with the addition of length."""


class SpacedLengthMetadata(SpacedLabeledMetadata, LengthMetadata):
    """Metadata with the addition of length."""


def fundamental_delta_x(metadata: LengthMetadata) -> float:
    """Get the fundamental delta x.

    Parameters
    ----------
    metadata : LengthMetadata

    Returns
    -------
    float
    """
    return metadata.delta.item()


def fundamental_dx(metadata: SpacedLengthMetadata) -> float:
    """Get the fundamental dx.

    Parameters
    ----------
    metadata : SpacedLengthMetadata

    Returns
    -------
    float
    """
    return fundamental_delta_x(metadata) / fundamental_size(metadata)


def fundamental_dk(metadata: SpacedLengthMetadata) -> float:
    """Get the fundamental dk.

    Parameters
    ----------
    metadata : SpacedLengthMetadata

    Returns
    -------
    float
    """
    return 2 * np.pi / fundamental_delta_x(metadata)


def fundamental_delta_k(metadata: SpacedLengthMetadata) -> float:
    """Get the fundamental delta k.

    Parameters
    ----------
    metadata : SpacedLengthMetadata

    Returns
    -------
    float
    """
    return fundamental_size(metadata) * fundamental_dk(metadata)


def fundamental_x_points(
    metadata: SpacedLengthMetadata,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the coordinates, using the x convention (0...N).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    return fundamental_delta_k(metadata) * fundamental_nk_points(metadata)


def fundamental_k_points(
    metadata: SpacedLengthMetadata,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Get the coordinates, using the kx convention (0...N/2-N/2...).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    return fundamental_dk(metadata) * fundamental_nk_points(metadata)


def _assert_orthonormal(vectors: tuple[np.ndarray[Any, np.dtype[np.float64]]]) -> None:
    identity_matrix = np.eye(len(vectors))
    result = np.dot(vectors, np.transpose(vectors))

    assert np.testing.assert_allclose(
        result,
        identity_matrix,
        err_msg="The provided axis vectors are not orthonormal.",
    )


@dataclass(frozen=True, kw_only=True)
class AxisDirections:
    """Data to store the axis vectors of an array."""

    vectors: tuple[np.ndarray[Any, np.dtype[np.float64]]]

    def __post_init__(self) -> None:
        _assert_orthonormal(self.vectors)


type VolumeMetadata = StackedMetadata[LengthMetadata, AxisDirections]
type SpacedVolumeMetadata = StackedMetadata[SpacedLengthMetadata, AxisDirections]


def fundamental_stacked_delta_x(
    metadata: VolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked delta x.

    Parameters
    ----------
    metadata : VolumeMetadata

    Returns
    -------
    tuple[np.ndarray[Any, np.dtype[np.float64]], ...]
    """
    scaled = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.einsum("ij,i->ij", metadata.extra, [c.delta for c in metadata.children]),  # type: ignore unknown
    )
    return tuple(scaled)


def fundamental_stacked_dx(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked dx.

    Parameters
    ----------
    metadata : VolumeMetadata

    Returns
    -------
    tuple[np.ndarray[Any, np.dtype[np.float64]], ...]
    """
    scaled = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.einsum(  # type: ignore unknown
            "ij,i->ij",
            fundamental_stacked_delta_x(metadata),
            [1 / s for s in metadata.fundamental_shape],
        ),
    )
    return tuple(scaled)


def fundamental_stacked_dk(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked dk.

    Parameters
    ----------
    metadata : VolumeMetadata

    Returns
    -------
    tuple[np.ndarray[Any, np.dtype[np.float64]], ...]
    """
    return tuple(2 * np.pi * np.linalg.inv(fundamental_stacked_delta_x(metadata)).T)


def fundamental_stacked_delta_k(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the fundamental stacked delta k.

    Parameters
    ----------
    metadata : VolumeMetadata

    Returns
    -------
    tuple[np.ndarray[Any, np.dtype[np.float64]], ...]
    """
    scaled = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.einsum(  # type: ignore unknown
            "ij,i->ij", fundamental_stacked_dk(metadata), metadata.fundamental_shape
        ),
    )
    return tuple(scaled)


def fundamental_volume(metadata: VolumeMetadata) -> float:
    """Get the fundamental volume.

    Parameters
    ----------
    metadata : VolumeMetadata

    Returns
    -------
    float
    """
    return np.linalg.det(fundamental_stacked_delta_x(metadata))


def fundamental_reciprocal_volume(
    metadata: SpacedVolumeMetadata,
) -> float:
    """Get the fundamental reciprocal volume.

    Parameters
    ----------
    metadata : VolumeMetadata

    Returns
    -------
    float
    """
    return np.linalg.det(fundamental_stacked_delta_k(metadata))


def fundamental_stacked_x_points(
    metadata: SpacedVolumeMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.float64]], ...]:
    """Get the stacked coordinates, using the x convention (0...N).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    scaled = cast(
        np.ndarray[Any, np.dtype[np.float64]],
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
    """Get the stacked coordinates, using the kx convention (0...N/2-N/2...).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    scaled = cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.einsum(  # type: ignore unknown
            "ij,ik->jk",
            fundamental_stacked_dk(metadata),
            fundamental_stacked_nk_points(metadata),
        ),
    )
    return tuple(scaled)
