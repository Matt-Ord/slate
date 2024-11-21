"""Functions for plotting slate data."""

from __future__ import annotations

from ._squared_scale import SquaredScale
from ._util import Axes, Figure, Measure, Scale, get_figure, get_measured_data
from .animate import (
    animate_data_1d_k,
    animate_data_1d_n,
    animate_data_1d_x,
    animate_data_2d_k,
    animate_data_2d_x,
)
from .plot import (
    plot_data_1d,
    plot_data_1d_k,
    plot_data_1d_n,
    plot_data_1d_x,
    plot_data_2d,
    plot_data_2d_k,
    plot_data_2d_x,
)

__all__ = [
    "Axes",
    "Figure",
    "Measure",
    "Scale",
    "SquaredScale",
    "animate_data_1d_k",
    "animate_data_1d_n",
    "animate_data_1d_x",
    "animate_data_2d_k",
    "animate_data_2d_x",
    "animate_data_2d_x",
    "get_figure",
    "get_measured_data",
    "plot_data_1d",
    "plot_data_1d_k",
    "plot_data_1d_n",
    "plot_data_1d_x",
    "plot_data_2d",
    "plot_data_2d_k",
    "plot_data_2d_x",
]
