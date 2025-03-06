from __future__ import annotations

from typing import TYPE_CHECKING, cast, override

import numpy as np
from matplotlib.scale import ScaleBase
from matplotlib.ticker import Locator
from matplotlib.transforms import Transform

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axis import Axis
    from matplotlib.pylab import ArrayLike


class SquaredLocator(Locator):
    @override
    def __call__(self):  # noqa: ANN204
        assert self.axis is not None
        min_val, max_val = self.axis.get_view_interval()  # type: ignore unknown lib type
        return self.tick_values(min_val, max_val)

    @override
    def tick_values(self, vmin: float, vmax: float) -> Sequence[float]:
        """
        Return the locations of the ticks.

        .. note::

            Because the values are fixed, vmin and vmax are not used in this
            method.

        """
        locs = cast(
            "Sequence[float]", np.sqrt(np.linspace(vmin**2, vmax**2, 8)).tolist()
        )
        return self.raise_if_exceeds(locs)


class SquaredScale(ScaleBase):
    name = "squared"

    def __init__(self, axis: Axis | None) -> None:
        super().__init__(axis)

    @override
    def get_transform(self) -> Transform:
        return self._SquaredTransform()

    @override
    def set_default_locators_and_formatters(self, axis: Axis) -> None:
        axis.set_major_locator(SquaredLocator())

    class _SquaredTransform(Transform):
        @property
        @override
        def input_dims(self) -> int:
            return 1

        @property
        @override
        def output_dims(self) -> int:
            return 1

        @property
        @override
        def is_separable(self) -> bool:
            return True

        @override
        def transform_non_affine(self, values: ArrayLike) -> ArrayLike:
            return np.square(np.real(values))

        @override
        def inverted(self) -> Transform:
            return SquaredScale._InvertedSquaredTransform()

    class _InvertedSquaredTransform(Transform):
        @property
        @override
        def input_dims(self) -> int:
            return 1

        @property
        @override
        def output_dims(self) -> int:
            return 1

        @property
        @override
        def is_separable(self) -> bool:
            return True

        @override
        def transform_non_affine(self, values: ArrayLike) -> ArrayLike:
            return np.sqrt(np.abs(values))

        @override
        def inverted(self) -> Transform:
            return SquaredScale._SquaredTransform()
