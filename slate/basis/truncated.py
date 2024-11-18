from __future__ import annotations

from typing import Any, Callable, Self, override

import numpy as np

from slate.basis._basis import Basis, BasisFeatures
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util._pad import (
    Padding,
    Truncation,
    assert_unique_indices,
    pad_along_axis,
    truncate_along_axis,
)


class TruncatedBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]],
):
    """Represents a basis sampled evenly along an axis."""

    def __init__(self: Self, truncation: Truncation, inner: Basis[M, DT]) -> None:
        self._truncation = truncation
        super().__init__(inner)
        assert_unique_indices(
            self._inner.size, self._truncation.n, self._truncation.step
        )

    def __hash__(self) -> int:
        return hash((self._inner, self._truncation))

    @property
    def truncation(self: Self) -> Truncation:
        """Truncation of the basis."""
        return self._truncation

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._truncation.n

    @override
    def conjugate_basis(self) -> TruncatedBasis[M, DT]:
        return self

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TruncatedBasis):
            return self._truncation == value._truncation and value._inner == self._inner  # type: ignore unknown
        return False

    @property
    def _inner_padding(self: Self) -> Padding:
        return Padding(self._inner.size, self._truncation.step, self._truncation.offset)

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return pad_along_axis(vectors, self._inner_padding, axis)

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return truncate_along_axis(vectors, self.truncation, axis)

    @override
    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](self: Self, inner: Basis[M1, DT1]) -> TruncatedBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]]
    ) -> TruncatedBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return TruncatedBasis(self.truncation, wrapper(self.inner))

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
