from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.animation import ArtistAnimation

from slate.basis.stacked import (
    fundamental_tuple_basis_from_metadata,
)
from slate.metadata.stacked.volume.util import get_x_coordinates_in_axes
from slate.plot.plot import (
    plot_data_1d_k,
    plot_data_1d_n,
    plot_data_1d_x,
    plot_data_2d_k,
    plot_data_2d_x,
)
from slate.util import get_data_in_axes

from ._util import (
    Axes,
    Figure,
    Measure,
    Scale,
    get_figure,
    get_lim,
    get_measured_data,
    get_norm_with_lim,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.collections import QuadMesh
    from matplotlib.image import AxesImage
    from matplotlib.lines import Line2D

    from slate.array.array import SlateArray
    from slate.metadata.stacked.volume import SpacedVolumeMetadata


def build_animation(  # noqa: PLR0913
    build_frame: Callable[[int, Axes], QuadMesh | AxesImage],
    n: int,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
    clim: tuple[float | None, float | None] = (None, None),
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Build an animation from the data, set the scale and clim to the correct values.

    Parameters
    ----------
    build_frame : Callable[[int, Axes], QuadMesh | AxesImage]
        function to generate each frame
    n : int
        number of frames to generate
    ax : Axes | None, optional
        plot axis, by default None
    scale : Literal[&quot;symlog&quot;, &quot;linear&quot;], optional
        scale, by default "linear"
    clim : tuple[float  |  None, float  |  None], optional
        plot clim, by default (None, None)

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    fig, ax = get_figure(ax)

    mesh0 = build_frame(0, ax)
    frames = [[build_frame(d, ax)] for d in range(n)]

    clim = get_lim(clim, measure, np.array([i[0].get_clim() for i in frames]))
    norm = get_norm_with_lim(scale, clim)
    for (mesh,) in frames:
        mesh.set_norm(norm)
        mesh.set_clim(*clim)
    mesh0.set_norm(norm)
    mesh0.set_clim(*clim)

    return (fig, ax, ArtistAnimation(fig, frames))


def animate_data_through_surface_x[DT: np.number[Any]](  # noqa: PLR0913
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    clim: tuple[float | None, float | None] = (None, None),
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, ArtistAnimation]:
    """
    Given data on a given coordinate grid in 3D, animate through the surface.

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
    clim : tuple[float  |  None, float  |  None], optional
        clim, by default (None, None)
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, ArtistAnimation]
    """
    idx = tuple(0 for _ in range(data.basis.n_dim - 3)) if idx is None else idx
    clim = (0.0, clim[1]) if clim[0] is None and measure == "abs" else clim

    basis_x = fundamental_tuple_basis_from_metadata(data.basis.metadata)
    converted_data = data.with_basis(basis_x).raw_data.reshape(basis_x.shape)

    coordinates = get_x_coordinates_in_axes(data.basis.metadata, axes, idx)
    data_in_axis = get_data_in_axes(converted_data, axes, idx)
    measured_data = get_measured_data(data_in_axis, measure)

    fig, ax, ani = build_animation(
        lambda i, ax: ax.pcolormesh(
            *coordinates[:2, :, :, i],
            measured_data[:, :, i],
            shading="nearest",
        ),
        converted_data.shape[2],
        clim=clim,
        ax=ax,
        scale=scale,
        measure=measure,
    )
    ax.set_aspect("equal", adjustable="box")
    fig.colorbar(ax.collections[0], ax=ax, format="%4.1e")

    ax.set_xlabel(f"x{axes[0]} axis")
    ax.set_ylabel(f"x{axes[1]} axis")
    return fig, ax, ani


def _get_slice_idx(
    axes: tuple[int, ...],
    x_0_idx: int,
    idx: tuple[int, ...],
) -> tuple[int, ...]:
    insert_pos = axes[0] - len([i for i in axes if i < axes[0]])
    return idx[:insert_pos] + (x_0_idx,) + idx[insert_pos:]


def animate_data_through_list_1d_n[DT: np.number[Any]](  # noqa: PLR0913
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


def animate_data_through_list_1d_x[DT: np.number[Any]](  # noqa: PLR0913
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


def animate_data_through_list_1d_k[DT: np.number[Any]](  # noqa: PLR0913
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


def animate_data_through_list_2d_k[DT: np.number[Any]](  # noqa: PLR0913
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


def animate_data_through_list_2d_x[DT: np.number[Any]](  # noqa: PLR0913
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
