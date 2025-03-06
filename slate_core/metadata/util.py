from __future__ import annotations

from typing import Any

import numpy as np

from slate_core.metadata._metadata import BasisMetadata, SimpleMetadata
from slate_core.metadata._shape import shallow_shape_from_nested, size_from_nested_shape


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
    metadata: BasisMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the x convention (0...N)."""
    shallow_shape = shallow_shape_from_nested(metadata.fundamental_shape)
    mesh = np.meshgrid(
        *(fundamental_nx_points(SimpleMetadata(n)) for n in shallow_shape),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)


def fundamental_stacked_nk_points(
    metadata: BasisMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the kx convention (0...N/2-N/2...)."""
    shallow_shape = shallow_shape_from_nested(metadata.fundamental_shape)
    mesh = np.meshgrid(
        *(fundamental_nk_points(SimpleMetadata(n)) for n in shallow_shape),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)
