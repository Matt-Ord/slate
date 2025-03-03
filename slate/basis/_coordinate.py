from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, TypeGuard, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisConversion, BasisFeature, ctype
from slate.basis._wrapped import WrappedBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Sequence


class CoordinateBasis[  # noqa: PLW1641
    M: BasisMetadata = BasisMetadata,
    DT: ctype[Never] = ctype[Never],
](WrappedBasis[Basis[M, DT], DT]):
    """Represents a basis sampled evenly along an axis."""

    def __init__(
        self,
        points: Sequence[int] | np.ndarray[Any, np.dtype[np.int_]],
        inner: Basis[M, DT],
    ) -> None:
        self._inner_points = np.sort(points)
        super().__init__(inner)
        assert np.unique(self._inner_points).size == self._inner_points.size

    @property
    def inner_points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        """Truncation of the basis."""
        return self._inner_points

    @property
    @override
    def size(self) -> int:
        return self.inner_points.size

    @override
    def __eq__(self, other: object) -> bool:
        if is_coordinate_basis(other):
            return (
                np.allclose(self.inner_points, other.inner_points)
                and other._inner == self._inner
            )
        return False

    @override
    def __into_inner__[DT2: np.generic, DT3: np.generic](
        self: CoordinateBasis[Any, ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, DT2, DT3]:
        def _fn() -> np.ndarray[Any, np.dtype[DT2]]:
            swapped = vectors.swapaxes(axis, 0)
            out = np.zeros((self.inner.size, *swapped.shape[1:]), dtype=vectors.dtype)
            out[self.points] = swapped
            return out.swapaxes(axis, 0)

        return BasisConversion(_fn)

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic](
        self: CoordinateBasis[Any, ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, np.generic]:
        def _fn() -> np.ndarray[Any, np.dtype[DT2]]:
            return vectors.swapaxes(axis, 0)[self.inner_points].swapaxes(axis, 0)

        return BasisConversion(_fn)

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)

        return (
            cast("WrappedBasis[Any, ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


def is_coordinate_basis(basis: object) -> TypeGuard[CoordinateBasis]:
    """Check if a basis is a coordinate basis."""
    return isinstance(basis, CoordinateBasis)
