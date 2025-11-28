from typing import Literal, Never, TypedDict


class LegendKwargs(TypedDict, total=False):
    loc: Literal["upper right", "upper left", "lower left", "lower right"]


Axes = Never
Figure = Never
TupleAnimation = Never
