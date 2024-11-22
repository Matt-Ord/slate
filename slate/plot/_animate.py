from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.animation import ArtistAnimation

from slate.plot._plot import (
    plot_data_1d_k,
    plot_data_1d_n,
    plot_data_1d_x,
    plot_data_2d_k,
    plot_data_2d_x,
)
from slate.plot._util import (
    Axes,
    Figure,
    Measure,
    Scale,
    get_figure,
)

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.lines import Line2D

    from slate.array import SlateArray
    from slate.metadata import SpacedVolumeMetadata


def _get_slice_idx(
    axes: tuple[int, ...],
    x_0_idx: int,
    idx: tuple[int, ...],
) -> tuple[int, ...]:
    insert_pos = axes[0] - len([i for i in axes if i < axes[0]])
    return idx[:insert_pos] + (x_0_idx,) + idx[insert_pos:]


def animate_data_1d_n[DT: np.number[Any]](  # noqa: PLR0913
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
    periodic: bool = False,
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)
    idx = tuple(np.zeros(data.basis.n_dim - 2)) if idx is None else idx

    frames: list[list[Line2D]] = []

    for idx_x0 in range(data.basis.fundamental_shape[axes[0]]):
        _, _, line = plot_data_1d_n(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
            periodic=periodic,
        )

        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)
    return fig, ax, ani


def animate_data_1d_x[DT: np.number[Any]](  # noqa: PLR0913
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)
    idx = tuple(np.zeros(data.basis.n_dim - 2)) if idx is None else idx

    frames: list[list[Line2D]] = []

    for idx_x0 in range(data.basis.fundamental_shape[axes[0]]):
        _, _, line = plot_data_1d_x(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)
    return fig, ax, ani


def animate_data_1d_k[DT: np.number[Any]](  # noqa: PLR0913
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)
    idx = tuple(np.zeros(data.basis.n_dim - 2)) if idx is None else idx

    frames: list[list[Line2D]] = []

    for idx_x0 in range(data.basis.fundamental_shape[axes[0]]):
        _, _, line = plot_data_1d_k(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append([line])
        line.set_color(frames[0][0].get_color())

    ani = ArtistAnimation(fig, frames)

    return fig, ax, ani


def animate_data_2d_x[DT: np.number[Any]](  # noqa: PLR0913
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)
    idx = tuple(np.zeros(data.basis.n_dim - 3)) if idx is None else idx

    frames: list[list[QuadMesh]] = []

    for idx_x0 in range(data.basis.fundamental_shape[axes[0]]):
        _, _, mesh = plot_data_2d_x(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append([mesh])

    ani = ArtistAnimation(fig, frames)
    return fig, ax, ani


def animate_data_2d_k[DT: np.number[Any]](  # noqa: PLR0913
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data, animate along the given direction.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int, int], optional
        plot axes (z, y, z), by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        idx in remaining dimensions, by default None
    ax : Axes | None, optional
        plot ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)
    idx = tuple(np.zeros(data.basis.n_dim - 3)) if idx is None else idx

    frames: list[list[QuadMesh]] = []
    for idx_x0 in range(data.basis.fundamental_shape[axes[0]]):
        _, _, mesh = plot_data_2d_k(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append([mesh])

    ani = ArtistAnimation(fig, frames)
    return fig, ax, ani
