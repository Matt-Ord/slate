from __future__ import annotations

from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    Literal,
    TypedDict,
    Unpack,
    cast,
    overload,
    override,
)

import numpy as np

try:
    from matplotlib import pyplot as plt
    from matplotlib.animation import ArtistAnimation as MPLArtistAnimation
    from matplotlib.axes import Axes as MPLAxesBase
    from matplotlib.colors import LogNorm, SymLogNorm
    from matplotlib.colors import Normalize as BaseNorm
    from matplotlib.figure import Figure as MPLFigureBase
    from matplotlib.scale import LinearScale, LogScale, SymmetricalLogScale

    from slate.plot._squared_scale import SquaredScale
except ImportError:
    plt = None
    LogNorm, BaseNorm, SymLogNorm = (None, None, None)
    LinearScale, LogScale, SymmetricalLogScale = (None, None, None)
    SquaredScale = None

    class _MPLAxesBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
            msg = (
                "Matplotlib is not installed. Please install it with the 'plot' extra."
            )
            raise ImportError(msg)

    MPLAxesBase: type[MPLAxes] = _MPLAxesBase  # type: ignore assign

    class _MPLFigureBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
            msg = (
                "Matplotlib is not installed. Please install it with the 'plot' extra."
            )
            raise ImportError(msg)

    MPLFigureBase: type[MPLFigure] = _MPLFigureBase  # type: ignore assign

    class ArtistAnimationBase:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
            msg = (
                "Matplotlib is not installed. Please install it with the 'plot' extra."
            )
            raise ImportError(msg)

    MPLArtistAnimation: type[ArtistAnimation] = ArtistAnimationBase  # type: ignore assign


if TYPE_CHECKING:
    import os
    from collections.abc import Iterable, Sequence

    from matplotlib.animation import ArtistAnimation
    from matplotlib.artist import Artist
    from matplotlib.axes import Axes as MPLAxes
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import QuadMesh
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap, Normalize
    from matplotlib.container import ErrorbarContainer
    from matplotlib.figure import Figure as MPLFigure
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D
    from matplotlib.scale import ScaleBase
    from matplotlib.text import Text
    from matplotlib.transforms import Transform
    from matplotlib.typing import ColorType
    from numpy.typing import ArrayLike


Scale = Literal["symlog", "linear", "squared", "log"]


class LegendKwargs(TypedDict, total=False):
    loc: Literal["upper right", "upper left", "lower left", "lower right"]


class Axes(MPLAxesBase):
    @override
    def get_figure(self) -> Figure | None: ...  # type: ignore bad overload

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
    @override
    def set_title(  # type: ignore bad overload
        self,
        label: str,
        fontdict: dict[str, Any] | None = ...,
        loc: Literal["left", "center", "right"] | None = ...,
        pad: float | None = ...,
        *,
        y: float | None = ...,
    ) -> Text: ...

    @overload
    def legend(self, **kwargs: Unpack[LegendKwargs]) -> Legend: ...
    @overload
    def legend(
        self,
        handles: Iterable[Artist | tuple[Artist, ...]],
        labels: Iterable[str],
        **kwargs: Unpack[LegendKwargs],
    ) -> Legend: ...
    @overload
    def legend(
        self, labels: Iterable[str], **kwargs: Unpack[LegendKwargs]
    ) -> Legend: ...
    @override
    def legend(self, *args: Any, **kwargs: Any) -> Legend: ...  # type: ignore overload

    @overload
    def plot(  # type: ignore overload
        self,
        *args: float | ArrayLike | str,
        scalex: bool = ...,
        scaley: bool = ...,
    ) -> list[Line2D]: ...
    def axvline(self, x: float = 0, ymin: float = 0, ymax: float = 1) -> Line2D: ...  # type: ignore overload

    def axhline(self, y: float = 0, xmin: float = 0, xmax: float = 1) -> Line2D: ...  # type: ignore overload


class Figure(MPLFigureBase):
    def colorbar(  # type: ignore bad overload
        self,
        mappable: ScalarMappable,
        cax: Axes | None = ...,
        ax: Axes | Iterable[Axes] | None = ...,
        use_gridspec: bool = ...,  # noqa: FBT001
        *,
        format: str = ...,  # noqa: A002
    ) -> Colorbar: ...

    @override
    def savefig(  # type: ignore override
        self,
        fname: str | os.PathLike[Any] | IO[Any],
        *,
        transparent: bool | None = ...,
    ) -> None: ...


def get_figure(ax: Axes | None = None) -> tuple[Figure, Axes]:
    """Get the figure of the given axis.

    If no figure exists, a new figure is created
    """  # noqa: DOC501
    if plt is None:
        msg = "Matplotlib is not installed. Please install it with the 'plot' extra."
        raise ImportError(msg)  # noqa: RUF100

    if ax is None:
        return cast("tuple[Figure, Axes]", plt.subplots())  # type: ignore plt.subplots Unknown type

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


def get_scale_with_lim(
    scale: Scale,
    lim: tuple[float, float],
) -> ScaleBase:
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


class TupleAnimation[*TS](MPLArtistAnimation):
    frame_seq: Iterable[tuple[*TS]]  # type: ignore overload

    def __init__(
        self,
        fig: MPLFigure,
        artists: Sequence[tuple[*TS]],
    ) -> None:
        super().__init__(fig, artists)  # type: ignore unknown

    @override
    def new_frame_seq(self) -> Iterable[tuple[*TS]]:  # type: ignore bad parent type
        return cast("Iterable[tuple[*TS,]]", super().new_frame_seq())

    @override
    def new_saved_frame_seq(self) -> Iterable[tuple[*TS]]:  # type: ignore bad parent type
        return cast("Iterable[tuple[*TS,]]", super().new_saved_frame_seq())


def combine_animations[*TS0, *TS1](
    fig: Figure, lhs: TupleAnimation[*TS0], rhs: TupleAnimation[*TS1]
) -> TupleAnimation[*TS0, *TS1]:
    """Combine multiple animations into a single animation."""
    artists = [
        cast("tuple[Any, ...]", a + b)
        for a, b in zip(lhs.frame_seq, rhs.frame_seq, strict=False)
    ]
    return TupleAnimation(fig, artists)


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
