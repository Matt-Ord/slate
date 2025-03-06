from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from slate_core.basis import Basis
from slate_core.plot._util import (
    Axes,
    Figure,
    Measure,
    Scale,
    get_figure,
    get_measured_data,
    get_scale_with_lim,
)

if TYPE_CHECKING:
    from slate_core.array import Array

Distribution = Literal["normal", "exponential normal", "skew normal"]


def array_distribution[B: Basis, DT: np.dtype[np.floating]](
    array: Array[B, DT],
    *,
    ax: Axes | None = None,
    scale: Scale = "linear",
    measure: Measure = "real",
    distribution: Distribution | None = None,
) -> tuple[Figure, Axes]:
    """Plot the distribution of data in a slate array."""
    fig, ax = get_figure(ax)
    data = get_measured_data(array.as_array().ravel(), measure)

    std = np.std(data).item()
    average = np.average(data).item()
    x_range = (
        (average - 4 * std, average + 4 * std)
        if distribution is not None
        else (np.min(data).item(), np.max(data).item())
    )
    n_bins = np.max([11, data.size // 100]).item()

    ax.hist(data, bins=n_bins, range=x_range, density=True)  # type: ignore unknown arg
    ax.set_ylabel("Occupation")
    ax.set_yscale(get_scale_with_lim(scale, ax.get_ylim()))
    return fig, ax
