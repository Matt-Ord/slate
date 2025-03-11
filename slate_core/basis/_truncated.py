from __future__ import annotations

from typing import Any, Never, TypeGuard, cast, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._wrapped import WrappedBasis
from slate_core.util._pad import (
    Padding,
    Truncation,
    assert_unique_indices,
    pad_along_axis,
    truncate_along_axis,
)


class TruncatedBasis[
    B: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[B, CT],
):
    """Represents a basis sampled evenly along an axis."""

    def __init__[B_: Basis](
        self: TruncatedBasis[B_, Ctype[Never]], truncation: Truncation, inner: B_
    ) -> None:
        self._truncation = truncation
        super().__init__(cast("B", inner))
        assert_unique_indices(
            self._inner.size, self._truncation.n, self._truncation.step
        )

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def __hash__(self) -> int:
        return hash((self._inner, self._truncation))

    @property
    def truncation(self) -> Truncation:
        """Truncation of the basis."""
        return self._truncation

    @property
    @override
    def size(self) -> int:
        """Number of elements in the basis."""
        return self._truncation.n

    @override
    def __eq__(self, other: object) -> bool:
        if is_truncated_basis(other):
            return self._truncation == other._truncation and other._inner == self._inner
        return False

    @property
    def _inner_padding(self) -> Padding:
        return Padding(self._inner.size, self._truncation.step, self._truncation.offset)

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: TruncatedBasis[Basis[Any, Ctype[DT3]], Ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion(
            lambda: pad_along_axis(vectors, self._inner_padding, axis)
        )

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: TruncatedBasis[Basis[Any, Ctype[DT1]], Ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion(
            lambda: truncate_along_axis(vectors, self.truncation, axis)
        )

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
            cast("WrappedBasis[Any, Ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


def is_truncated_basis(basis: object) -> TypeGuard[TruncatedBasis]:
    """Check if the basis is truncated."""
    return isinstance(basis, TruncatedBasis)
