from typing import Literal, Never, TypedDict


class LegendKwargs(TypedDict, total=False):
    loc: Literal["upper right", "upper left", "lower left", "lower right"]


TupleAnimation = Never
