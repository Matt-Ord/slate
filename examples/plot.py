from __future__ import annotations

import numpy as np

from slate_core import array, basis, metadata, plot
from slate_core.array import Array

if __name__ == "__main__":
    rng = np.random.default_rng()
    raw_data = rng.random((10, 10, 10), dtype=np.float64)
    data = Array.from_array(raw_data)

    # Any data can be plotted
    fig, _, _anim0 = plot.animate_data_2d(data)
    fig.show()

    # If you provide metadata, the axes will be labeled accordingly
    square_meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])), (10, 10, 10)
    )
    data = array.cast_basis(data, basis.from_metadata(square_meta))
    fig, _, _anim1 = plot.animate_data_2d(data)
    fig.show()

    # For non-orthogonal axes, the plot will respect the relative angles
    fcc_meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        (
            np.array([0, 1, 0]),
            np.array([1 / np.sqrt(2), 1 / np.sqrt(2), 0]),
            np.array([0, 0, 1]),
        ),
        (10, 10, 10),
    )
    data = array.cast_basis(data, basis.from_metadata(fcc_meta))
    fig, _, _anim2 = plot.animate_data_2d(data, axes=(0, 1, 2))
    fig.show()

    # Non-square axes are also supported, and the plot will respect the relative lengths
    rectangle_meta = metadata.volume.spaced_volume_metadata_from_stacked_delta_x(
        (
            np.array([0, 1, 0]),
            np.array([5, 0, 0]),
            np.array([0, 0, 1]),
        ),
        (10, 10, 10),
    )
    data = array.cast_basis(data, basis.from_metadata(rectangle_meta))
    fig, _, _anim3 = plot.animate_data_2d(data, axes=(0, 1, 2))
    fig.show()
    plot.wait_for_close()
