from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, override

import numpy as np

try:
    from matplotlib import pyplot as plt
except ImportError:
    plt = None


from matplotlib.axes import Axes as MPLAxes
from matplotlib.colors import Colormap, LogNorm, Normalize, SymLogNorm
from matplotlib.figure import Figure as MPLFigure
from matplotlib.scale import LinearScale, LogScale, ScaleBase, SymmetricalLogScale

from slate.plot._squared_scale import SquaredScale

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.container import ErrorbarContainer
    from matplotlib.text import Text
    from matplotlib.transforms import Transform
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike

Scale = Literal["symlog", "linear", "squared", "log"]


class Axes(MPLAxes):
    @override
    def get_figure(self) -> Figure | None: ...

    def set_xlabel(self, xlabel: str) -> None: ...  # type: ignore bad overload
    def set_ylabel(self, xlabel: str) -> None: ...  # type: ignore bad overload

    def set_yscale(self, value: str | ScaleBase) -> None: ...  # type: ignore bad overload
    def set_xscale(self, value: str | ScaleBase) -> None: ...  # type: ignore bad overload

    def errorbar(  # type: ignore bad overload  # noqa: PLR0913, PLR0917
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        yerr: float | ArrayLike | None = ...,
        xerr: float | ArrayLike | None = ...,
        fmt: str = ...,
        ecolor: ColorType | None = ...,
        elinewidth: float | None = ...,
        capsize: float | None = ...,
        barsabove: bool = ...,  # noqa: FBT001
        lolims: bool | ArrayLike = ...,  # noqa: FBT001
        uplims: bool | ArrayLike = ...,  # noqa: FBT001
        xlolims: bool | ArrayLike = ...,  # noqa: FBT001
        xuplims: bool | ArrayLike = ...,  # noqa: FBT001
        errorevery: int | tuple[int, int] = ...,
        capthick: float | None = ...,
    ) -> ErrorbarContainer: ...

    def pcolormesh(  # type: ignore bad overload  # noqa: PLR0913
        self,
        *args: ArrayLike,
        alpha: float | None = ...,
        norm: str | Normalize | None = ...,
        cmap: str | Colormap | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        shading: Literal["flat", "nearest", "gouraud", "auto"] | None = ...,
        antialiased: bool = ...,
    ) -> QuadMesh: ...
    def text(  # type: ignore bad overload  # noqa: PLR0913
        self,
        x: float,
        y: float,
        s: str,
        fontdict: dict[str, Any] | None = ...,
        *,
        transform: Transform | None = ...,
        verticalalignment: str = ...,
        bbox: dict[str, Any] | None = ...,
    ) -> Text: ...
    @override
    def twinx(self) -> Axes: ...
    @override
    def twiny(self) -> Axes: ...


class Figure(MPLFigure):
    def colorbar(  # type: ignore bad overload
        self,
        mappable: ScalarMappable,
        cax: Axes | None = ...,
        ax: Axes | Iterable[Axes] | None = ...,
        use_gridspec: bool = ...,  # noqa: FBT001
        *,
        format: str = ...,  # noqa: A002
    ) -> Colorbar: ...


def get_figure(ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """  # noqa: DOC501
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)  # noqa: RUF100

    if ax is None:
        return cast(tuple[Figure, Axes], plt.subplots())  # type: ignore plt.subplots Unknown type

    fig = ax.get_figure()
    if fig is None:
        fig = cast(Figure, plt.figure())  # type: ignore plt.figure Unknown type
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


def get_measured_data[DT: np.number[Any]](
    data: np.ndarray[Any, np.dtype[DT]],
    measure: Measure,
) -> np.ndarray[Any, np.dtype[np.float64]]:
    """Transform data with the given measure."""
    match measure:
        case "real":
            return np.real(data)  # type: ignore[no-any-return]
        case "imag":
            return np.imag(data)  # type: ignore[no-any-return]
        case "abs":
            return np.abs(data)  # type: ignore[no-any-return]
        case "angle":
            return np.unwrap(np.angle(data))  # type: ignore[no-any-return]


def _get_default_lim(
    measure: Measure, data: np.ndarray[Any, np.dtype[np.float64]]
) -> tuple[float, float]:
    if measure == "abs":
        return (0, float(np.max(data)))
    return (float(np.min(data)), float(np.max(data)))


def get_lim(
    lim: tuple[float | None, float | None],
    measure: Measure,
    data: np.ndarray[Any, np.dtype[np.float64]],
) -> tuple[float, float]:
    (default_min, default_max) = _get_default_lim(measure, data)
    l_max = default_max if lim[1] is None else lim[1]
    l_min = default_min if lim[0] is None else lim[0]
    return (l_min, l_max)


def get_norm_with_lim(
    scale: Scale,
    lim: tuple[float, float],
) -> Normalize:
    match scale:
        case "linear":
            return Normalize(vmin=lim[0], vmax=lim[1])
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
            return Normalize(vmin=lim[0], vmax=lim[1])


def get_scale_with_lim(
    scale: Scale,
    lim: tuple[float, float],
) -> ScaleBase:
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
