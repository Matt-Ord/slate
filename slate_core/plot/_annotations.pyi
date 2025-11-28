import os
from collections.abc import Iterable, Sequence
from typing import (
    IO,
    Any,
    Literal,
    TypedDict,
    Unpack,
    overload,
    override,
    type_check_only,
)

from matplotlib.animation import ArtistAnimation as MPLArtistAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes as MPLAxesBase
from matplotlib.cm import ScalarMappable
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.colors import Colormap, Normalize
from matplotlib.container import ErrorbarContainer
from matplotlib.figure import Figure as MPLFigure
from matplotlib.figure import Figure as MPLFigureBase
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.scale import ScaleBase
from matplotlib.text import Text
from matplotlib.transforms import Bbox, Transform
from matplotlib.typing import ColorType
from numpy.typing import ArrayLike

class LegendKwargs(TypedDict, total=False):
    loc: Literal["upper right", "upper left", "lower left", "lower right"]

@type_check_only
class Axes(MPLAxesBase):
    @override
    def get_figure(self) -> Figure | None: ...  # type: ignore bad overload
    def set_xlabel(self, xlabel: str) -> None: ...  # type: ignore bad overload
    def set_ylabel(self, xlabel: str) -> None: ...  # type: ignore bad overload
    def set_yscale(self, value: str | ScaleBase) -> None: ...  # type: ignore bad overload
    def set_xscale(self, value: str | ScaleBase) -> None: ...  # type: ignore bad overload
    def errorbar(  # type: ignore bad overload
        self,
        x: float | ArrayLike,
        y: float | ArrayLike,
        yerr: float | ArrayLike | None = ...,
        xerr: float | ArrayLike | None = ...,
        fmt: str = ...,
        ecolor: ColorType | None = ...,
        elinewidth: float | None = ...,
        capsize: float | None = ...,
        barsabove: bool = ...,
        lolims: bool | ArrayLike = ...,
        uplims: bool | ArrayLike = ...,
        xlolims: bool | ArrayLike = ...,
        xuplims: bool | ArrayLike = ...,
        errorevery: int | tuple[int, int] = ...,
        capthick: float | None = ...,
    ) -> ErrorbarContainer: ...
    def pcolormesh(  # type: ignore bad overload
        self,
        *args: ArrayLike,
        alpha: float | None = ...,
        norm: str | Normalize | None = ...,
        cmap: str | Colormap | None = ...,
        vmin: float | None = ...,
        vmax: float | None = ...,
        shading: Literal["flat", "nearest", "gouraud", "auto"]  # cspell:disable-line
        | None = ...,
        antialiased: bool = ...,
    ) -> QuadMesh: ...
    def text(  # type: ignore bad overload
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

@type_check_only
class Figure(MPLFigureBase):
    def colorbar(  # type: ignore bad overload
        self,
        mappable: ScalarMappable,
        cax: Axes | None = ...,
        ax: Axes | Iterable[Axes] | None = ...,
        use_gridspec: bool = ...,
        *,
        format: str = ...,  # noqa: A002
    ) -> Colorbar: ...
    @override
    def savefig(  # type: ignore override
        self,
        fname: str | os.PathLike[Any] | IO[Any],
        *,
        transparent: bool | None = ...,
        dpi: Literal["figure"] | float = "figure",
        format: str | None = None,
        bbox_inches: str | Bbox | None = None,
        pad_inches: float = 0.1,
        facecolor: Literal["none"] | None = None,
    ) -> None: ...

@type_check_only
class TupleAnimation[*TS](MPLArtistAnimation):
    frame_seq: Iterable[tuple[*TS]]  # type: ignore overload

    def __init__(
        self,
        fig: MPLFigure,
        artists: Sequence[tuple[*TS]],
    ) -> None: ...  # type: ignore unknown
    @override
    def new_frame_seq(self) -> Iterable[tuple[*TS]]:  # type: ignore bad parent type
        ...
    @override
    def new_saved_frame_seq(self) -> Iterable[tuple[*TS]]:  # type: ignore bad parent type
        ...
