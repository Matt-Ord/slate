from __future__ import annotations

from typing import TYPE_CHECKING, Never

import numpy as np

from slate_core.metadata import shallow_shape_from_nested
from slate_core.plot._plot import (
    array_against_axes_1d,
    array_against_axes_1d_k,
    array_against_axes_2d,
    array_against_axes_2d_k,
    array_against_basis,
    index_text,
)
from slate_core.plot._util import (
    Axes,
    Figure,
    Measure,
    Scale,
    TupleAnimation,
    get_figure,
)

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from slate_core.array import Array
    from slate_core.basis import TupleBasisLike
    from slate_core.basis._basis import Basis, Ctype
    from slate_core.metadata import (
        BasisMetadata,
        EvenlySpacedVolumeMetadata,
        SimpleMetadata,
    )
    from slate_core.metadata._tuple import TupleMetadata


def _get_slice_idx(
    axes: tuple[int, ...],
    x_0_idx: int,
    idx: tuple[int, ...],
) -> tuple[int, ...]:
    insert_pos = axes[2] - len([i for i in axes if i < axes[2]])
    return (*idx[:insert_pos], x_0_idx, *idx[insert_pos:])


def animate_array_over_list[DT: np.dtype[np.number]](
    data: Array[
        TupleBasisLike[tuple[SimpleMetadata, BasisMetadata], Never, Ctype[np.floating]],
        DT,
    ],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)

    frames: list[tuple[Line2D]] = []

    for idx_x0 in range(shape[0]):
        _, _, line = array_against_basis(
            data[idx_x0, :],
            ax=ax,
            scale=scale,
            measure=measure,
        )

        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_over_list_1d_x[DT: np.dtype[np.number]](  # noqa: PLR0913
    data: Array[TupleBasisLike[tuple[SimpleMetadata, EvenlySpacedVolumeMetadata]], DT],
    axes: tuple[int] = (0,),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    frames: list[tuple[Line2D]] = []

    for d in data:
        _, _, line = array_against_axes_1d(
            d,
            axes,
            idx,
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_1d_x[DT: np.dtype[np.number]](  # noqa: PLR0913
    data: Array[Basis[EvenlySpacedVolumeMetadata, Ctype[np.floating]], DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 1)) if idx is None else idx

    frames: list[tuple[Line2D]] = []

    for idx_x0 in range(shape[axes[1]]):
        _, _, line = array_against_axes_1d(
            data,
            axes[:1],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_over_list_1d_k[DT: np.dtype[np.complexfloating]](  # noqa: PLR0913
    data: Array[TupleBasisLike[tuple[SimpleMetadata, EvenlySpacedVolumeMetadata]], DT],
    axes: tuple[int] = (0,),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)

    frames: list[tuple[Line2D]] = []

    for d in data:
        _, _, line = array_against_axes_1d_k(
            d,
            axes,
            idx,
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_1d_k[DT: np.dtype[np.complexfloating]](  # noqa: PLR0913
    data: Array[Basis[EvenlySpacedVolumeMetadata], DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 2)) if idx is None else idx

    frames: list[tuple[Line2D]] = []

    for idx_x0 in range(shape[axes[1]]):
        _, _, line = array_against_axes_1d_k(
            data,
            axes[:1],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)

    return fig, ax, ani


def animate_data_2d[DT: np.dtype[np.number]](  # noqa: PLR0913
    data: Array[Basis[TupleMetadata], DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[QuadMesh, Text]]:
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
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 3)) if idx is None else idx

    frames: list[tuple[QuadMesh, Text]] = []

    for idx_x0 in range(shape[axes[2]]):
        slice_idx = _get_slice_idx(axes, idx_x0, idx)
        _, _, mesh = array_against_axes_2d(
            data,
            axes[:2],
            slice_idx,
            ax=ax,
            scale=scale,
            measure=measure,
            plot_index_text=False,
        )
        text = index_text(ax, slice_idx)
        frames.append((mesh, text))

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_2d_k[DT: np.dtype[np.complexfloating]](  # noqa: PLR0913
    data: Array[Basis[EvenlySpacedVolumeMetadata], DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, TupleAnimation[QuadMesh]]:
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
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 2)) if idx is None else idx

    frames: list[tuple[QuadMesh]] = []
    for idx_x0 in range(shape[axes[2]]):
        _, _, mesh = array_against_axes_2d_k(
            data,
            axes[:2],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((mesh,))

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani
