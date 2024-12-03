from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import numpy as np

from slate import array, basis
from slate.metadata._metadata import LabeledMetadata
from slate.metadata.length import fundamental_k_points, fundamental_x_points
from slate.metadata.volume import (
    AxisDirections,
    get_k_coordinates_in_axes,
    get_x_coordinates_in_axes,
)
from slate.plot._util import (
    Axes,
    Figure,
    Measure,
    get_axis_colorbar,
    get_figure,
    get_lim,
    get_measured_data,
    get_norm_with_lim,
    get_scale_with_lim,
)
from slate.util import (
    get_data_in_axes,
    get_max_idx,
)

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.lines import Line2D

    from slate.array import SlateArray
    from slate.metadata import BasisMetadata, StackedMetadata
    from slate.metadata.length import SpacedLengthMetadata
from slate.metadata import SpacedVolumeMetadata
from slate.plot._util import Scale, set_ymargin


class PlotKwargs(TypedDict, total=False):
    """Extra arguments to plot functions."""

    ax: Axes | None
    scale: Scale
    measure: Measure


def _plot_raw_data_1d[DT: np.number[Any]](  # noqa: PLR0913
    data: np.ndarray[tuple[int], np.dtype[DT]],
    coordinates: np.ndarray[tuple[int], np.dtype[np.float64]],
    y_errors: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
    periodic: bool = False,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data in 1d.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.complex128]]
    coordinates : np.ndarray[tuple[int], np.dtype[np.float64]]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(data, measure)
    # The data is periodic and evenly spaced
    # so we repeat the first point at the end
    if periodic:
        # Seperation between the coordinates (assume evenly spaced and repeating)
        d_coord = coordinates[1] - coordinates[0]
        coordinates = np.append(coordinates, coordinates[-1] + d_coord)
        measured_data = np.append(measured_data, measured_data[0])

    container = ax.errorbar(coordinates, measured_data, yerr=y_errors)
    line = container.lines[0]
    ax.set_xmargin(0)
    set_ymargin(ax, 0, 0.05)
    if measure == "abs":
        ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_yscale(get_scale_with_lim(scale, ax.get_ylim()))
    return fig, ax, line


def plot_array[DT: np.number[Any]](
    data: SlateArray[BasisMetadata, DT],
    *,
    periodic: bool = False,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data in 1d.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.complex128]]
    coordinates : np.ndarray[tuple[int], np.dtype[np.float64]]
    ax : Axes | None, optional
        ax, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    converted = array.as_index_basis(data)

    metadata = converted.basis.metadata()
    if isinstance(metadata, LabeledMetadata):
        values = metadata.values.astype(np.float64)  # type: ignore unknown
        coordinates = values[converted.basis.points]
    else:
        coordinates = converted.basis.points.astype(np.float64)

    return _plot_raw_data_1d(
        converted.raw_data, coordinates, periodic=periodic, **kwargs
    )


def plot_data_1d_k[DT: np.number[Any]](
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int,] = (0,),
    idx: tuple[int, ...] | None = None,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data along axes in the k basis.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int], optional
        axes to plot in, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis_k = basis.fundamental_transformed_tuple_basis_from_metadata(
        data.basis.metadata(), is_dual=data.basis.is_dual
    )
    converted_data = data.with_basis(basis_k).raw_data.reshape(basis_k.shape)

    idx = get_max_idx(converted_data, axes) if idx is None else idx

    coordinates = get_k_coordinates_in_axes(basis_k.metadata(), axes, idx)
    data_in_axis = get_data_in_axes(converted_data, axes, idx)

    shifted_data = np.fft.fftshift(data_in_axis)
    shifted_coordinates = np.fft.fftshift(coordinates[0])

    fig, ax, line = _plot_raw_data_1d(
        shifted_data, shifted_coordinates, periodic=True, **kwargs
    )

    ax.set_xlabel(f"k{(axes[0] % 3)} axis")
    return fig, ax, line


def plot_data_1d_x[DT: np.number[Any]](
    data: SlateArray[SpacedVolumeMetadata, DT],
    axes: tuple[int,] = (0,),
    idx: tuple[int, ...] | None = None,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data along axes in the x basis.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int], optional
        axes to plot in, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, Line2D]
    """
    basis_x = basis.from_metadata(data.basis.metadata(), is_dual=data.basis.is_dual)
    converted_data = data.with_basis(basis_x).raw_data.reshape(basis_x.shape)

    idx = get_max_idx(converted_data, axes) if idx is None else idx

    coordinates = get_x_coordinates_in_axes(basis_x.metadata(), axes, idx)
    data_in_axis = get_data_in_axes(converted_data, axes, idx)

    fig, ax, line = _plot_raw_data_1d(
        data_in_axis, coordinates[0], periodic=True, **kwargs
    )

    ax.set_xlabel(f"x{(axes[0] % 3)} axis")

    return fig, ax, line


def _has_colorbar(axis: Axes) -> bool:
    colourbar = get_axis_colorbar(axis)
    return colourbar is not None


def plot_data_2d[DT: np.number[Any]](
    data: np.ndarray[Any, np.dtype[DT]],
    coordinates: np.ndarray[tuple[int, int], np.dtype[np.float64]] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "abs",
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot data in 2d."""
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(data, measure)

    mesh = (
        ax.pcolormesh(measured_data)
        if coordinates is None
        else ax.pcolormesh(*coordinates, measured_data, shading="nearest")
    )
    clim = get_lim((None, None), measure, measured_data)
    norm = get_norm_with_lim(scale, clim)
    mesh.set_norm(norm)
    mesh.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    if not _has_colorbar(ax):
        fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def _get_lengths_in_axes(
    metadata: StackedMetadata[SpacedLengthMetadata, Any],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the lengths from each axis in a grid."""
    points = tuple(fundamental_x_points(metadata.children[ax]) for ax in axes)
    aa = np.meshgrid(*points, indexing="ij")
    return np.asarray(aa)


def plot_data_2d_x[DT: np.number[Any], E](
    data: SlateArray[StackedMetadata[SpacedLengthMetadata, E], DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the data in 2d along the x axis in the given basis.

    Parameters
    ----------
    basis : TupleBasisLike
        basis to interpret the data in
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
        plot data
    axes : tuple[int, int, int], optional
        axes to plot in, by default (0, 1, 2)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    metadata = data.basis.metadata()
    basis_x = basis.from_metadata(metadata, is_dual=data.basis.is_dual)
    converted_data = data.with_basis(basis_x).raw_data.reshape(basis_x.shape)

    idx = get_max_idx(converted_data, axes) if idx is None else idx

    if isinstance(metadata.extra, AxisDirections):
        metadata = cast(SpacedVolumeMetadata, metadata)
        coordinates = get_x_coordinates_in_axes(metadata, axes, idx)
    else:
        coordinates = _get_lengths_in_axes(metadata, axes)

    data_in_axis = get_data_in_axes(converted_data, axes, idx)

    fig, ax, mesh = plot_data_2d(
        data_in_axis.reshape(coordinates.shape[1:]), coordinates, **kwargs
    )

    ax.set_xlabel(f"x{axes[0]} axis / m")
    ax.set_ylabel(f"x{axes[1]} axis / m")
    if len(idx) > 0:
        ax.text(
            0.05,
            0.95,
            f"x = {idx}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
    return fig, ax, mesh


def _get_frequencies_in_axes(
    metadata: StackedMetadata[SpacedLengthMetadata, Any],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Get the lengths from each axis in a grid."""
    points = tuple(fundamental_k_points(metadata.children[ax]) for ax in axes)
    aa = np.meshgrid(*points, indexing="ij")
    return np.asarray(aa)


def plot_data_2d_k[DT: np.number[Any], E](
    data: SlateArray[StackedMetadata[SpacedLengthMetadata, E], DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, QuadMesh]:
    """
    Plot the data in a 2d slice in k along the given axis.

    Parameters
    ----------
    basis : TupleBasisLike
    data : np.ndarray[tuple[_L0Inv], np.dtype[np.complex_]]
    axes : tuple[int, int], optional
        axes to plot in, by default (0, 1)
    idx : SingleStackedIndexLike | None, optional
        index to plot, by default None
    ax : Axes | None, optional
        plot axis, by default None
    scale : Scale, optional
        scale, by default "linear"
    measure : Measure, optional
        measure, by default "abs"

    Returns
    -------
    tuple[Figure, Axes, QuadMesh]
    """
    metadata = data.basis.metadata()
    basis_k = basis.fundamental_transformed_tuple_basis_from_metadata(
        metadata, is_dual=data.basis.is_dual
    )
    converted_data = data.with_basis(basis_k).raw_data.reshape(basis_k.shape)

    idx = get_max_idx(converted_data, axes) if idx is None else idx

    if isinstance(metadata.extra, AxisDirections):
        metadata = cast(SpacedVolumeMetadata, metadata)
        coordinates = get_k_coordinates_in_axes(metadata, axes, idx)
    else:
        coordinates = _get_frequencies_in_axes(metadata, axes)
    data_in_axis = get_data_in_axes(converted_data, axes, idx)

    shifted_data = np.fft.fftshift(data_in_axis)
    shifted_coordinates = np.fft.fftshift(coordinates, axes=(1, 2))

    fig, ax, mesh = plot_data_2d(shifted_data, shifted_coordinates, **kwargs)

    ax.set_xlabel(f"k{axes[0]} axis / $m^-1$")
    ax.set_ylabel(f"k{axes[1]} axis / $m^-1$")
    if len(idx) > 0:
        ax.text(
            0.05,
            0.95,
            f"k = {idx}",
            transform=ax.transAxes,
            verticalalignment="top",
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )
    return fig, ax, mesh
