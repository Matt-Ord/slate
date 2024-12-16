"""Functions for plotting slate data."""

from __future__ import annotations

from slate.plot._animate import (
    animate_array_over_list,
    animate_data_1d_k,
    animate_data_1d_x,
    animate_data_2d_k,
    animate_data_2d_x,
    animate_data_over_list_1d_k,
    animate_data_over_list_1d_x,
)
from slate.plot._plot import (
    array_against_array,
    basis_against_array_1d_k,
    basis_against_array_1d_x,
    basis_against_array_2d_k,
    basis_against_array_2d_x,
)
from slate.plot._squared_scale import SquaredScale
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
    "basis_against_array_1d_k",
    "basis_against_array_1d_x",
    "basis_against_array_2d_k",
    "basis_against_array_2d_x",
    "combine_animations",
    "get_figure",
    "get_measured_data",
]
