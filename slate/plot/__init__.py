"""Functions for plotting slate data."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slate.plot._animate import (
    animate_array_over_list,
    animate_data_1d_k,
    animate_data_1d_x,
    animate_data_2d_k,
    animate_data_2d_x,
    animate_data_over_list_1d_k,
    animate_data_over_list_1d_x,
)
from slate.plot._distribution import Distribution, array_distribution
from slate.plot._plot import (
    array_against_array,
    basis_against_array,
    basis_against_array_1d_k,
    basis_against_array_1d_x,
    basis_against_array_2d_k,
    basis_against_array_2d_x,
)

try:
    from slate.plot._squared_scale import SquaredScale
except ImportError:

    class SquaredScaleMock:
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401, ARG002
            msg = (
                "Matplotlib is not installed. Please install it with the 'plot' extra."
            )
            raise ImportError(msg)

    SquaredScale: type[SquaredScaleTy] = None  # type: ignore unknown

if TYPE_CHECKING:
    from slate.plot._squared_scale import SquaredScale as SquaredScaleTy


from slate.plot._util import (
    Axes,
    Figure,
    Measure,
    Scale,
    TupleAnimation,
    combine_animations,
    get_figure,
    get_measured_data,
)

__all__ = [
    "Axes",
    "Distribution",
    "Figure",
    "Measure",
    "Scale",
    "SquaredScale",
    "TupleAnimation",
    "animate_array_over_list",
    "animate_data_1d_k",
    "animate_data_1d_x",
    "animate_data_2d_k",
    "animate_data_2d_x",
    "animate_data_2d_x",
    "animate_data_over_list_1d_k",
    "animate_data_over_list_1d_x",
    "array_against_array",
    "array_distribution",
    "basis_against_array",
    "basis_against_array_1d_k",
    "basis_against_array_1d_x",
    "basis_against_array_2d_k",
    "basis_against_array_2d_x",
    "combine_animations",
    "get_figure",
    "get_measured_data",
]
