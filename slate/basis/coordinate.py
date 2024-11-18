from __future__ import annotations

from typing import Any, Callable, Self, Sequence, override

import numpy as np

from slate.basis._basis import Basis, BasisFeatures
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata


class CoordinateBasis[M: BasisMetadata, DT: np.generic](  # noqa: PLW1641
    WrappedBasis[M, DT, Basis[M, DT]]
):
    """Represents a basis sampled evenly along an axis."""

    def __init__(
        self: Self,
        points: Sequence[int] | np.ndarray[Any, np.dtype[np.int_]],
        inner: Basis[M, DT],
    ) -> None:
        self._points = np.sort(points)
        super().__init__(inner)
        assert np.unique(self._points).size == self._points.size

    @property
    def points(self: Self) -> np.ndarray[Any, np.dtype[np.int_]]:
        """Truncation of the basis."""
        return self._points

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self.points.size

    @override
    def conjugate_basis(self) -> CoordinateBasis[M, DT]:
        return self

    def __eq__(self, value: object) -> bool:
        if isinstance(value, CoordinateBasis):
            return (
                np.allclose(self.points, value.points) and value._inner == self._inner  # type: ignore unknown
            )
        return False

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        out = np.zeros((self.inner.size, *swapped.shape[1:]), dtype=vectors.dtype)
        out[self.points] = swapped
        return out.swapaxes(axis, 0)

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return vectors.swapaxes(axis, 0)[self.points].swapaxes(axis, 0)

    @override
    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](self: Self, inner: Basis[M1, DT1]) -> CoordinateBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]]
    ) -> CoordinateBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return CoordinateBasis(self.points, wrapper(self.inner))

    @property
    @override
    def features(self) -> set[BasisFeatures]:
        out = set[BasisFeatures]()
        if "SIMPLE_ADD" in self.inner.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.inner.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.inner.features:
            out.add("SUB")
            out.add("SIMPLE_SUB")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self: Self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)
