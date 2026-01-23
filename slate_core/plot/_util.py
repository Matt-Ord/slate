from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
)

import numpy as np
from matplotlib.animation import ArtistAnimation

try:
    from matplotlib import pyplot as plt
    from matplotlib.animation import ArtistAnimation
    from matplotlib.colors import LogNorm, SymLogNorm
    from matplotlib.colors import Normalize as BaseNorm
    from matplotlib.scale import LinearScale, LogScale, SymmetricalLogScale

    from slate_core.plot._squared_scale import SquaredScale
except ImportError:
    plt = None
    LogNorm, BaseNorm, SymLogNorm = (None, None, None)
    LinearScale, LogScale, SymmetricalLogScale = (None, None, None)
    SquaredScale = None
    ArtistAnimation = None


if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes as MPLAxesBase
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Normalize
    from matplotlib.scale import ScaleBase

    from slate_core.plot._annotations import Axes, Figure, TupleAnimation


Scale = Literal["symlog", "linear", "squared", "log"]


def get_figure(ax: MPLAxesBase | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """  # noqa: DOC501
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)  # noqa: RUF100

    if ax is None:
        return cast("tuple[Figure, Axes]", plt.subplots())  # type: ignore plt.subplots Unknown type

    ax = cast("Axes", ax)
    fig = ax.get_figure()
    if fig is None:
        fig = cast("Figure", plt.figure())  # type: ignore plt.figure Unknown type
        ax.set_figure(fig)
    return fig, ax


def get_axis_colorbar(axis: Axes) -> Colorbar | None:
    """Get a colorbar attached to the axis."""  # noqa: DOC501
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)
    for artist in axis.get_children():
        if isinstance(artist, plt.cm.ScalarMappable) and artist.colorbar is not None:
            return artist.colorbar
    return None


Measure = Literal["real", "imag", "abs", "angle"]


def _measure_data[DT: np.dtype[np.number]](
    data: np.ndarray[Any, DT],
    measure: Measure,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    match measure:
        case "real":
            return np.real(data)  # type: ignore[no-any-return]
        case "imag":
            return np.imag(data)  # type: ignore[no-any-return]
        case "abs":
            return np.abs(data)  # type: ignore[no-any-return]
        case "angle":
            return np.unwrap(np.angle(data))  # type: ignore[no-any-return]


def get_measured_data[DT: np.dtype[np.number[Any]]](
    data: np.ndarray[Any, DT],
    measure: Measure,
) -> np.ndarray[Any, np.dtype[np.floating]]:
    """Transform data with the given measure.

    Raises
    ------
        ValueError: If the data contains NaN values.
    """  # noqa: DOC501
    measured = _measure_data(data, measure)
    if np.any(np.isnan(measured)):
        msg = "The data contains NaN values."
        raise ValueError(msg)
    return measured


def _get_default_lim(
    measure: Measure, data: np.ndarray[Any, np.dtype[np.floating]]
) -> tuple[float, float]:
    if measure == "abs":
        return (0, float(np.max(data)))
    return (float(np.min(data)), float(np.max(data)))


def get_lim(
    lim: tuple[float | None, float | None],
    measure: Measure,
    data: np.ndarray[Any, np.dtype[np.floating]],
) -> tuple[float, float]:
    (default_min, default_max) = _get_default_lim(measure, data)
    l_max = default_max if lim[1] is None else lim[1]
    l_min = default_min if lim[0] is None else lim[0]
    return (l_min, l_max)


def get_norm_with_lim(
    scale: Scale,
    lim: tuple[float, float],
) -> Normalize:
    if BaseNorm is None or LogNorm is None or SymLogNorm is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)

    match scale:
        case "linear":
            return BaseNorm(vmin=lim[0], vmax=lim[1])
        case "log":
            return LogNorm(vmin=lim[0], vmax=lim[1])
        case "symlog":
            max_abs = max([np.abs(lim[0]), np.abs(lim[1])])
            return SymLogNorm(
                vmin=lim[0],
                vmax=lim[1],
                linthresh=1 if max_abs <= 0 else 1e-3 * max_abs,  # type: ignore No parameter named "linthresh"
            )
        case "squared":
            return BaseNorm(vmin=lim[0], vmax=lim[1])


def get_scale_with_lim(scale: Scale, lim: tuple[float, float]) -> ScaleBase:
    if (
        LinearScale is None
        or LogScale is None
        or SymmetricalLogScale is None
        or SquaredScale is None
    ):
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)
    match scale:
        case "linear":
            return LinearScale(axis=None)
        case "symlog":
            max_abs = max([np.abs(lim[0]), np.abs(lim[1])])
            return SymmetricalLogScale(
                axis=None,
                linthresh=1 if max_abs <= 0 else 1e-3 * max_abs,
            )
        case "log":
            max_abs = max([np.abs(lim[0]), np.abs(lim[1])])
            return LogScale(axis=None)
        case "squared":
            return SquaredScale(axis=None)


# https://stackoverflow.com/questions/49382105/set-different-margins-for-left-and-right-side
def set_ymargin(ax: Axes, bottom: float = 0.0, top: float = 0.3) -> None:
    ax.set_autoscale_on(b=True)
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = lim[1] - lim[0]
    bottom = lim[0] - delta * bottom
    top = lim[1] + delta * top
    ax.set_ylim(bottom, top)


def build_tuple_animation[*TS](
    fig: Figure,
    artists: Sequence[tuple[*TS]],
) -> TupleAnimation[*TS]:
    """Build a TupleAnimation from the given frames.

    Raises
    ------
        ImportError: If Matplotlib is not installed.
    """  # noqa: DOC501
    if ArtistAnimation is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)
    return cast(
        "TupleAnimation[*TS]",
        ArtistAnimation(fig, cast("Any", artists)),
    )


def combine_animations[*TS0, *TS1](
    fig: Figure, lhs: TupleAnimation[*TS0], rhs: TupleAnimation[*TS1]
) -> TupleAnimation[*TS0, *TS1]:
    """Combine multiple animations into a single animation.

    Raises
    ------
        ImportError: If Matplotlib is not installed.
    """
    artists = [
        cast("tuple[Any, ...]", a + b)
        for a, b in zip(lhs.frame_seq, rhs.frame_seq, strict=False)
    ]
    return build_tuple_animation(fig, artists)


def wait_for_close() -> None:
    """Block until the figure is closed."""  # noqa: DOC501
    try:
        from matplotlib import _pylab_helpers  # noqa: PLC0415, PLC2701
    except ImportError as e:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg) from e

    backend = _pylab_helpers.Gcf.get_active()

    if backend is not None:
        backend.start_main_loop()
