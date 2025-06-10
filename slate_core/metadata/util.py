from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate_core.metadata._metadata import SIMPLE_FEATURE
from slate_core.metadata._shape import size_from_nested_shape

if TYPE_CHECKING:
    from slate_core.metadata._metadata import (
        BasisMetadata,
    )

    from ._tuple import TupleMetadata


def fundamental_size(
    metadata: BasisMetadata,
) -> int:
    """Get the size."""
    return size_from_nested_shape(metadata.fundamental_shape)


def nx_points(size: int) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the index, using the x convention (0...N)."""
    return np.arange(0, size, dtype=int)


def nk_points(size: int) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the index, using the kx convention (0...N/2-N/2...)."""
    n = size
    return np.fft.ifftshift(np.arange((-n + 1) // 2, (n + 1) // 2))


def fundamental_nx_points(
    metadata: BasisMetadata,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the index, using the x convention (0...N)."""
    return np.arange(0, fundamental_size(metadata), dtype=int)


def fundamental_nk_points(
    metadata: BasisMetadata,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the index, using the kx convention (0...N/2-N/2...)."""
    n = fundamental_size(metadata)
    return np.fft.ifftshift(np.arange((-n + 1) // 2, (n + 1) // 2))


def fundamental_stacked_nx_points(
    metadata: TupleMetadata[tuple[BasisMetadata, ...], Any],
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the x convention (0...N)."""
    mesh = np.meshgrid(
        *(fundamental_nx_points(m) for m in metadata.children),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)


def fundamental_stacked_nk_points(
    metadata: TupleMetadata[tuple[BasisMetadata, ...], Any],
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the kx convention (0...N/2-N/2...)."""
    mesh = np.meshgrid(
        *(fundamental_nk_points(m) for m in metadata.children),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)


def basis_components_from_array_points[DT: np.dtype[np.generic]](
    metadata: BasisMetadata,
    points: np.ndarray[Any, DT],
) -> np.ndarray[Any, DT]:
    r"""Get the components f_j from the points f(R_j).

    Data is not always stored as the points in the array, but rather in a
    form that is weighted by the basis functions. This property provides the weights
    that are applied to the data when it is transformed to the basis.

    The fundamental basis represents data discretised into a series
    of orthogonal basis functions U_i at discrete points R_j
    where j = 0, ..., N-1 such that:

    .. math::
        U_i(R_j) = w_k \delta_{ij}

    where :math:`w_k` are the weights. Data represented in this basis
    must therefore be scaled by these weights

    .. math::
        f(R) = f_j U_j(R)
    where

    .. math::
        f_j = f(R_j) / w_j
    """
    if SIMPLE_FEATURE in metadata.features:
        return points

    assert np.issubdtype(points.dtype, np.floating) or np.issubdtype(
        points.dtype, np.complexfloating
    ), f"Expected float or complex dtype, got {points.dtype}"
    return points.ravel() / metadata.basis_weights  # type: ignore[return-value]


def array_points_from_basis_components[DT: np.dtype[np.generic]](
    metadata: BasisMetadata,
    components: np.ndarray[Any, DT],
) -> np.ndarray[Any, DT]:
    r"""Get the points f(R_j) from the components f_j.

    Data is not always stored as the points in the array, but rather in a
    form that is weighted by the basis functions. This property provides the weights
    that are applied to the data when it is transformed to the basis.

    The fundamental basis represents data discretised into a series
    of orthogonal basis functions U_i at discrete points R_j
    where j = 0, ..., N-1 such that:

    .. math::
        U_i(R_j) = w_k \delta_{ij}

    where :math:`w_k` are the weights. Data represented in this basis
    must therefore be scaled by these weights

    .. math::
        f(R) = f_j U_j(R)
    where

    .. math::
        f_j = f(R_j) / w_j
    """
    if SIMPLE_FEATURE in metadata.features:
        return components
    assert np.issubdtype(components.dtype, np.floating) or np.issubdtype(
        components.dtype, np.complexfloating
    ), f"Expected float or complex dtype, got {components.dtype}"
    return components.ravel() * metadata.basis_weights  # type: ignore[return-value]
