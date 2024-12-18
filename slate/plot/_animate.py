from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate import basis
from slate.array import Array
from slate.basis import (
    tuple_basis,
)
from slate.metadata._metadata import BasisMetadata
from slate.metadata._shape import shallow_shape_from_nested
from slate.plot._plot import (
    basis_against_array,
    basis_against_array_1d_k,
    basis_against_array_1d_x,
    basis_against_array_2d_k,
    basis_against_array_2d_x,
)
from slate.plot._util import (
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

    from slate.metadata import SpacedVolumeMetadata
    from slate.metadata._metadata import SimpleMetadata
    from slate.metadata.stacked import Metadata2D


def _get_slice_idx(
    axes: tuple[int, ...],
    x_0_idx: int,
    idx: tuple[int, ...],
) -> tuple[int, ...]:
    insert_pos = axes[0] - len([i for i in axes if i < axes[0]])
    return idx[:insert_pos] + (x_0_idx,) + idx[insert_pos:]


def _index_array[M: BasisMetadata, DT: np.generic](
    array: Array[Metadata2D[BasisMetadata, M, Any], DT], idx: int
) -> Array[M, DT]:
    as_tuple = basis.as_tuple_basis(array.basis)
    converted = array.with_basis(
        tuple_basis((basis.as_fundamental(as_tuple[0]), as_tuple[1]))
    )
    return Array(as_tuple[1], converted.raw_data.reshape(converted.basis.shape)[idx])


def animate_array_over_list[DT: np.number[Any]](
    data: Array[Metadata2D[SimpleMetadata, BasisMetadata, Any], DT],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
    periodic: bool = False,
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)

    frames: list[tuple[Line2D]] = []

    for idx_x0 in range(shape[0]):
        _, _, line = basis_against_array(
            _index_array(data, idx_x0),
            ax=ax,
            scale=scale,
            measure=measure,
            periodic=periodic,
        )

        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_over_list_1d_x[DT: np.number[Any]](  # noqa: PLR0913
    data: Array[Metadata2D[SimpleMetadata, SpacedVolumeMetadata, Any], DT],
    axes: tuple[int] = (0,),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)

    basis_as_tuple = basis.as_tuple_basis(data.basis)
    basis_x = basis.from_metadata(data.basis.metadata()[1], is_dual=data.basis.is_dual)
    final_basis = tuple_basis((basis.as_index_basis(basis_as_tuple[0]), basis_x))

    data = data.with_basis(final_basis)
    shape = shallow_shape_from_nested(basis_x.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 1)) if idx is None else idx

    frames: list[tuple[Line2D]] = []

    for raw_data in data.raw_data.reshape(data.basis.shape):
        _, _, line = basis_against_array_1d_x(
            Array(basis_x, raw_data),
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


def animate_data_1d_x[DT: np.number[Any]](  # noqa: PLR0913
    data: Array[SpacedVolumeMetadata, DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 1)) if idx is None else idx

    frames: list[tuple[Line2D]] = []

    for idx_x0 in range(shape[axes[0]]):
        _, _, line = basis_against_array_1d_x(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_over_list_1d_k[DT: np.number[Any]](  # noqa: PLR0913
    data: Array[Metadata2D[SimpleMetadata, SpacedVolumeMetadata, Any], DT],
    axes: tuple[int] = (0,),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)

    basis_as_tuple = basis.as_tuple_basis(data.basis)
    basis_k = basis.fundamental_transformed_tuple_basis_from_metadata(
        data.basis.metadata()[1], is_dual=data.basis.is_dual
    )
    final_basis = tuple_basis((basis.as_index_basis(basis_as_tuple[0]), basis_k))

    data = data.with_basis(final_basis)
    shape = shallow_shape_from_nested(basis_k.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 1)) if idx is None else idx

    frames: list[tuple[Line2D]] = []

    for raw_data in data.raw_data.reshape(data.basis.shape):
        _, _, line = basis_against_array_1d_k(
            Array(final_basis[1], raw_data),
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


def animate_data_1d_k[DT: np.number[Any]](  # noqa: PLR0913
    data: Array[SpacedVolumeMetadata, DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, TupleAnimation[Line2D]]:
    """Given data, animate along the given direction."""
    fig, ax = get_figure(ax)
    shape = shallow_shape_from_nested(data.basis.fundamental_shape)
    idx = tuple(0 for _ in range(len(shape) - 2)) if idx is None else idx

    frames: list[tuple[Line2D]] = []

    for idx_x0 in range(shape[axes[0]]):
        _, _, line = basis_against_array_1d_k(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((line,))
        line.set_color(frames[0][0].get_color())

    ani = TupleAnimation(fig, frames)

    return fig, ax, ani


def animate_data_2d_x[DT: np.number[Any]](  # noqa: PLR0913
    data: Array[SpacedVolumeMetadata, DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
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

    for idx_x0 in range(shape[axes[0]]):
        _, _, mesh = basis_against_array_2d_x(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((mesh,))

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani


def animate_data_2d_k[DT: np.number[Any]](  # noqa: PLR0913
    data: Array[SpacedVolumeMetadata, DT],
    axes: tuple[int, int, int] = (0, 1, 2),
    idx: tuple[int, ...] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
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
    for idx_x0 in range(shape[axes[0]]):
        _, _, mesh = basis_against_array_2d_k(
            data,
            axes[1:],
            _get_slice_idx(axes, idx_x0, idx),
            ax=ax,
            scale=scale,
            measure=measure,
        )
        frames.append((mesh,))

    ani = TupleAnimation(fig, frames)
    return fig, ax, ani
