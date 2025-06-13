from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict, Unpack, cast

import numpy as np
from matplotlib.collections import QuadMesh

from slate_core import array, basis
from slate_core.array import Array, get_data_in_axes
from slate_core.array._conversion import as_fundamental_basis
from slate_core.metadata import AxisDirections, LabeledMetadata
from slate_core.metadata._metadata import (
    PERIODIC_FEATUIRE,
    SIMPLE_FEATURE,
)
from slate_core.metadata.length import (
    fundamental_k_points,
)
from slate_core.metadata.volume import (
    get_k_coordinates_in_axes,
)
from slate_core.metadata.volume._util import project_points_along_axes
from slate_core.plot._util import (
    Axes,
    Figure,
    Measure,
    Scale,
    get_axis_colorbar,
    get_figure,
    get_lim,
    get_measured_data,
    get_norm_with_lim,
    get_scale_with_lim,
    set_ymargin,
)
from slate_core.util import (
    get_max_idx,
)

if TYPE_CHECKING:
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from slate_core.basis._basis import Basis, Ctype
    from slate_core.basis._tuple import TupleBasis, TupleBasisLike
    from slate_core.metadata import (
        BasisMetadata,
        EvenlySpacedVolumeMetadata,
        TupleMetadata,
    )
    from slate_core.metadata.length import EvenlySpacedLengthMetadata


class PlotKwargs(TypedDict, total=False):
    """Extra arguments to plot functions."""

    ax: Axes | None
    scale: Scale
    measure: Measure


def _plot_raw_data_1d[DT: np.dtype[np.number]](  # noqa: PLR0913
    data: np.ndarray[Any, DT],
    coordinates: np.ndarray[Any, np.dtype[np.floating]],
    y_errors: np.ndarray[Any, np.dtype[np.floating]] | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
    periodic: bool = False,
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data in 1d.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.complex128]]
    coordinates : np.ndarray[tuple[int], np.dtype[np.floating]]
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


def array_against_array[M: BasisMetadata, DT: np.dtype[np.number]](
    x_data: Array[Basis[M], np.dtype[np.floating]],
    y_data: Array[Basis[M], DT],
    *,
    y_error: Array[Basis[M], np.dtype[np.floating]] | None = None,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, Line2D]:
    """Plot two arrays against each other."""
    common_basis = cast(
        "Basis[Any, Ctype[np.generic]]",
        basis.get_common(basis.as_index(x_data.basis), basis.as_index(y_data.basis)),
    )

    y_errors = None if y_error is None else y_error.with_basis(common_basis).raw_data
    return _plot_raw_data_1d(
        y_data.with_basis(common_basis).raw_data,
        x_data.with_basis(common_basis).raw_data,
        y_errors,
        periodic=PERIODIC_FEATUIRE in x_data.basis.metadata().features,
        **kwargs,
    )


def _get_basis_coordinates(basis: Basis) -> np.ndarray[Any, np.dtype[np.floating]]:
    metadata = basis.metadata()
    if isinstance(metadata, LabeledMetadata):
        values = metadata.values.astype(np.float64)  # type: ignore unknown
        coordinates = values[basis.points]
    else:
        coordinates = basis.points.astype(np.float64)
    return coordinates


def _get_basis_units(basis: Basis) -> str:
    metadata = basis.metadata()
    if isinstance(metadata, LabeledMetadata):
        return metadata.unit
    # Otherwise we use the index of the basis
    return "idx"


def array_against_basis[M: BasisMetadata, DT: np.dtype[np.number]](
    data: Array[Basis[M], DT],
    *,
    y_error: Array[Basis[M], np.dtype[np.floating]] | None = None,
    **kwargs: Unpack[PlotKwargs],
) -> tuple[Figure, Axes, Line2D]:
    """
    Plot data in 1d.

    Parameters
    ----------
    data : np.ndarray[tuple[int], np.dtype[np.complex128]]
    coordinates : np.ndarray[tuple[int], np.dtype[np.floating]]
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
    converted = array.as_index_basis(array.as_supports_type_basis(data, np.floating))
    coordinates = _get_basis_coordinates(converted.basis)

    raw_data = converted.raw_data
    if SIMPLE_FEATURE not in converted.basis.metadata().features:
        raw_data = cast(
            "np.ndarray[Any, DT]",
            raw_data * converted.basis.metadata().basis_weights,
        )
    y_errors = None if y_error is None else y_error.with_basis(converted.basis).raw_data
    if (
        y_errors is not None
        and SIMPLE_FEATURE not in converted.basis.metadata().features
    ):
        y_errors = cast(
            "np.ndarray[Any, np.dtype[np.floating]]",
            y_errors * converted.basis.metadata().basis_weights,
        )

    return _plot_raw_data_1d(
        raw_data,
        coordinates,
        y_errors,
        periodic=PERIODIC_FEATUIRE in converted.basis.metadata().features,
        **kwargs,
    )


def array_against_axes_1d[DT: np.dtype[np.number]](
    data: Array[TupleBasisLike[tuple[BasisMetadata, ...]], DT],
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
    metadata = data.basis.metadata()
    idx = tuple(0 for _ in range(metadata.n_dim - 1)) if idx is None else idx

    data_in_axis = array.flatten(get_data_in_axes(data, axes, idx))
    fig, ax, line = array_against_basis(data_in_axis, **kwargs)

    return fig, ax, line


def array_against_axes_1d_k[DT: np.dtype[np.complexfloating]](
    data: Array[Basis[EvenlySpacedVolumeMetadata], DT],
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
    metadata = data.basis.metadata()
    converted_data = array.as_raw_tuple(array.as_transformed_basis(data))

    idx = tuple(0 for _ in range(metadata.n_dim - 1)) if idx is None else idx

    coordinates = get_k_coordinates_in_axes(metadata, axes, idx)
    data_in_axis = get_data_in_axes(converted_data, axes, idx)

    shifted_data = np.fft.fftshift(data_in_axis.as_array())
    shifted_coordinates = np.fft.fftshift(coordinates[0])

    fig, ax, line = _plot_raw_data_1d(
        shifted_data, shifted_coordinates, periodic=True, **kwargs
    )

    ax.set_xlabel(f"k{(axes[0] % 3)} axis")
    return fig, ax, line


def _has_colorbar(axis: Axes) -> bool:
    colourbar = get_axis_colorbar(axis)
    return colourbar is not None


def _get_max_clim(
    clim: list[tuple[float, float]],
) -> tuple[float, float]:
    return min(c[0] for c in clim), max(c[1] for c in clim)


def _plot_raw_data_2d[DT: np.dtype[np.number]](
    data: np.ndarray[Any, DT],
    coordinates: tuple[np.ndarray[tuple[int], np.dtype[np.floating]], ...]
    | None = None,
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
) -> tuple[Figure, Axes, QuadMesh]:
    """Plot data in 2d."""
    fig, ax = get_figure(ax)

    measured_data = get_measured_data(data, measure)

    mesh = (
        ax.pcolormesh(measured_data)
        if coordinates is None
        else ax.pcolormesh(*coordinates, measured_data, shading="nearest")
    )

    meshes = [
        mesh,
        *(child for child in ax.get_children() if isinstance(child, QuadMesh)),
    ]
    clim = _get_max_clim(
        [
            get_lim((None, None), measure, measured_data),
            *(x.get_clim() for x in meshes[1:]),
        ]
    )
    norm = get_norm_with_lim(scale, clim)

    for m in meshes:
        m.set_norm(norm)
        m.set_clim(*clim)
    ax.set_aspect("equal", adjustable="box")
    if not _has_colorbar(ax):
        fig.colorbar(mesh, ax=ax, format="%4.1e")
    return fig, ax, mesh


def _get_tuple_basis_coordinates(
    basis: TupleBasis[tuple[Basis[BasisMetadata, Any], ...], Any],
    axes: tuple[int, ...],
) -> tuple[np.ndarray[Any, np.dtype[np.floating]], ...]:
    """Get the lengths from each axis in a grid."""
    points = tuple(_get_basis_coordinates(basis.children[ax]) for ax in axes)
    mesh = np.meshgrid(*points, indexing="ij")
    if isinstance(basis.metadata().extra, AxisDirections):
        return project_points_along_axes(mesh, basis.metadata().extra, axes=axes)

    return mesh


def _get_tuple_basis_units(
    basis: TupleBasis[tuple[Basis[BasisMetadata, Any], ...], Any],
    axes: tuple[int, ...],
) -> tuple[str, ...]:
    """Get the lengths from each axis in a grid."""
    return tuple(_get_basis_units(basis.children[ax]) for ax in axes)


def index_text(ax: Axes, idx: tuple[int, ...]) -> Text:
    return ax.text(
        0.05,
        0.95,
        f"x = {idx}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )


class PlotKwargs2D(PlotKwargs, total=False):
    """Extra arguments to plot functions."""

    plot_index_text: bool


def array_against_axes_2d[DT: np.dtype[np.number], E](
    data: Array[Basis[TupleMetadata[tuple[BasisMetadata, ...], E]], DT],
    axes: tuple[int, int] = (0, 1),
    idx: tuple[int, ...] | None = None,
    **kwargs: Unpack[PlotKwargs2D],
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
    converted_data = as_fundamental_basis(data).raw_data.reshape(basis_x.shape)

    idx = get_max_idx(converted_data, axes=axes) if idx is None else idx

    coordinates = _get_tuple_basis_coordinates(basis_x, axes)
    data_in_axis = get_data_in_axes(data, axes, idx)

    raw_data = data_in_axis.as_array()

    fig, ax, mesh = _plot_raw_data_2d(
        raw_data,
        coordinates,
        ax=kwargs.get("ax"),
        scale=kwargs.get("scale", "linear"),
        measure=kwargs.get("measure", "real"),
    )

    unit_0, unit_1 = _get_tuple_basis_units(basis_x, axes)
    ax.set_xlabel(f"x{axes[0]} axis ({unit_0})")
    ax.set_ylabel(f"x{axes[1]} axis ({unit_1})")
    if len(idx) > 0 and kwargs.get("plot_index_text", True):
        index_text(ax, idx)
    return fig, ax, mesh


def _get_frequencies_in_axes[E](
    metadata: TupleMetadata[tuple[EvenlySpacedLengthMetadata, ...], E],
    axes: tuple[int, ...],
) -> np.ndarray[tuple[int, ...], np.dtype[np.floating]]:
    """Get the lengths from each axis in a grid."""
    points = tuple(fundamental_k_points(metadata.children[ax]) for ax in axes)
    aa = np.meshgrid(*points, indexing="ij")
    return np.asarray(aa)


def array_against_axes_2d_k[DT: np.dtype[np.complexfloating], E](
    data: Array[TupleBasisLike[tuple[EvenlySpacedLengthMetadata, ...], E], DT],
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
    converted_data = array.as_raw_tuple(array.as_transformed_basis(data))

    idx = (
        get_max_idx(
            converted_data.raw_data.reshape(converted_data.basis.shape), axes=axes
        )
        if idx is None
        else idx
    )

    if isinstance(metadata.extra, AxisDirections):
        metadata = cast("EvenlySpacedVolumeMetadata", metadata)
        coordinates = get_k_coordinates_in_axes(metadata, axes, idx)
    else:
        coordinates = _get_frequencies_in_axes(metadata, axes)
    data_in_axis = get_data_in_axes(converted_data, axes, idx)

    shifted_data = np.fft.fftshift(data_in_axis.as_array())
    shifted_coordinates = np.fft.fftshift(coordinates, axes=(1, 2))

    fig, ax, mesh = _plot_raw_data_2d(
        shifted_data, tuple(shifted_coordinates[i] for i in range(2)), **kwargs
    )

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
