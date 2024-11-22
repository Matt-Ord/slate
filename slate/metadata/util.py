from __future__ import annotations

from typing import Any

import numpy as np

from slate.metadata import BasisMetadata, SimpleMetadata


def fundamental_ndim(
    metadata: BasisMetadata,
) -> int:
    """Get the number of dimensions."""
    return len(metadata.fundamental_shape)


def fundamental_size(
    metadata: BasisMetadata,
) -> int:
    """Get the size."""
    return np.prod(metadata.fundamental_shape).item()


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
    mesh = np.meshgrid(
        *(fundamental_nx_points(SimpleMetadata(n)) for n in metadata.fundamental_shape),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)


def fundamental_stacked_nk_points(
    metadata: BasisMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the kx convention (0...N/2-N/2...)."""
    mesh = np.meshgrid(
        *(fundamental_nk_points(SimpleMetadata(n)) for n in metadata.fundamental_shape),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)
