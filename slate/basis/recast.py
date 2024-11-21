from __future__ import annotations

from typing import Any, Callable, Never, Self, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata


class RecastBasis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Any] = Basis[M0, DT],
](WrappedBasis[M0, DT, B]):
    """Represents a truncated basis."""

    def __init__[
        _M1: BasisMetadata,
        _DT: np.generic,
        _B: Basis[BasisMetadata, Any],
    ](
        self: RecastBasis[Any, _M1, _DT, _B],
        inner: _B,
        inner_recast: Basis[_M1, _DT],
        outer_recast: Basis[_M1, _DT],
    ) -> None:
        self._inner_recast: Basis[M1, DT] = inner_recast
        self._outer_recast: Basis[M1, DT] = outer_recast
        super().__init__(inner)

        assert self._inner_recast.size == self.inner.size

    @property
    def size(self: Self) -> int:
        """The size of the basis."""
        return self.outer_recast.size

    @property
    def inner_recast(self: Self) -> Basis[M1, DT]:
        """The basis the inner was recast to."""
        return self._inner_recast

    @property
    def outer_recast(self: Self) -> Basis[M1, DT]:
        """The basis the inner recast was transformed to."""
        return self._outer_recast

    @override
    def conjugate_basis(self) -> Basis[M0, DT]:
        return self

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return self._outer_recast.__convert_vector_into__(
            vectors, self._inner_recast, axis
        )

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return self._inner_recast.__convert_vector_into__(
            vectors, self._outer_recast, axis
        )

    @override
    def with_inner[
        _M: BasisMetadata,
        _DT: np.generic,
        _B: Basis[BasisMetadata, Never] = Basis[_M, _DT],
    ](self: RecastBasis[_M, M1, DT, B], inner: _B) -> RecastBasis[_M, M1, DT, _B]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        _M: BasisMetadata,
        _DT: np.generic,
        _B: Basis[BasisMetadata, Never] = Basis[_M, _DT],
    ](
        self: RecastBasis[_M, M1, DT, B], wrapper: Callable[[B], _B]
    ) -> RecastBasis[_M, M1, DT, _B]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return RecastBasis(wrapper(self.inner), self.inner_recast, self.outer_recast)

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "SIMPLE_ADD" in self.outer_recast.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.outer_recast.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.outer_recast.features:
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
