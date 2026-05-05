from collections.abc import Iterable, Sequence
from typing import (
    Literal,
    TypedDict,
    override,
    type_check_only,
)

from matplotlib.animation import ArtistAnimation as MPLArtistAnimation
from matplotlib.figure import Figure as MPLFigure

class LegendKwargs(TypedDict, total=False):
    loc: Literal["upper right", "upper left", "lower left", "lower right"]

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
