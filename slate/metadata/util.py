from __future__ import annotations

from typing import Any

import numpy as np

from slate.metadata._metadata import BasisMetadata, SimpleMetadata


def fundamental_ndim(
    metadata: BasisMetadata,
) -> int:
    """Get the number of dimensions.

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    int
    """
    return len(metadata.fundamental_shape)


def fundamental_size(
    metadata: BasisMetadata,
) -> int:
    """Get the size.

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    int
    """
    return np.prod(metadata.fundamental_shape).item()


def fundamental_nx_points(
    metadata: BasisMetadata,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the index, using the x convention (0...N).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    return np.arange(0, fundamental_size(metadata), dtype=int)


def fundamental_nk_points(
    metadata: BasisMetadata,
) -> np.ndarray[Any, np.dtype[np.int_]]:
    """Get the index, using the kx convention (0...N/2-N/2...).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    n = fundamental_size(metadata)
    return np.fft.ifftshift(np.arange((-n + 1) // 2, (n + 1) // 2))


def fundamental_stacked_nx_points(
    metadata: BasisMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the x convention (0...N).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    mesh = np.meshgrid(
        *(
            fundamental_nx_points(SimpleMetadata((n,)))
            for n in metadata.fundamental_shape
        ),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)


def fundamental_stacked_nk_points(
    metadata: BasisMetadata,
) -> tuple[np.ndarray[Any, np.dtype[np.int_]], ...]:
    """Get the stacked index, using the kx convention (0...N/2-N/2...).

    Parameters
    ----------
    metadata : BasisMetadata

    Returns
    -------
    np.ndarray[Any, np.dtype[np.int_]]
    """
    mesh = np.meshgrid(
        *(
            fundamental_nk_points(SimpleMetadata((n,)))
            for n in metadata.fundamental_shape
        ),
        indexing="ij",
    )
    return tuple(nki.ravel() for nki in mesh)
