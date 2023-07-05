from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from matplotlib import pyplot as plt

from surface_potential_analysis.util.util import Measure, get_measured_data

from .util import AxisUtil

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from surface_potential_analysis._types import SingleFlatIndexLike
    from surface_potential_analysis.axis.axis_like import AxisLike

    _A0Inv = TypeVar("_A0Inv", bound=AxisLike[int, int, Any])


def plot_explicit_basis_states_x(
    basis: _A0Inv,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, list[Line2D]]:
    """Plot basis states against position."""
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = AxisUtil(basis)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    lines: list[Line2D] = []
    for i, vector in enumerate(util.vectors):
        data = get_measured_data(vector, measure)
        (line,) = ax.plot(x_points, data)
        line.set_label(f"State {i}")
        lines.append(line)

    ax.set_xlabel("x / m")
    ax.set_ylabel("Amplitude")
    ax.set_title("Plot of the wavefunction of the explicit basis states")
    return fig, ax, lines


def plot_explicit_basis_state_x(
    basis: _A0Inv,
    idx: SingleFlatIndexLike = 0,
    *,
    ax: Axes | None = None,
    measure: Measure = "abs",
) -> tuple[Figure, Axes, Line2D]:
    """Plot basis states against position."""
    fig, ax = (ax.get_figure(), ax) if ax is not None else plt.subplots()
    util = AxisUtil(basis)

    x_points = np.linalg.norm(util.fundamental_x_points, axis=0)
    data = get_measured_data(util.vectors[idx], measure)
    (line,) = ax.plot(x_points, data)
    return fig, ax, line