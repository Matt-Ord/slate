from collections.abc import Iterable, Sequence
from typing import (
    override,
    type_check_only,
)

from matplotlib.animation import ArtistAnimation as MPLArtistAnimation
from matplotlib.figure import Figure as MPLFigure

@type_check_only
class TupleAnimation[A](MPLArtistAnimation):
    frame_seq: Iterable[tuple[A, ...]]

    def __init__(
        self,
        fig: MPLFigure,
        artists: Sequence[tuple[A, ...]],
    ) -> None: ...
    @override
    def new_frame_seq(self) -> Iterable[tuple[A, ...]]:  # type: ignore bad parent type
        ...
    @override
    def new_saved_frame_seq(self) -> Iterable[tuple[A, ...]]:  # type: ignore bad parent type
        ...
