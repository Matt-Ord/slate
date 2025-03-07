from __future__ import annotations

from typing import Any, Never, Self, cast, override

import numpy as np

from slate_core.metadata import BasisMetadata

from ._basis import Basis, BasisConversion, BasisFeature, NestedBool, ctype


class AsUpcast[B: Basis, M: BasisMetadata, DT: ctype[Never] = ctype[Never]](
    Basis[M, DT]
):
    def __init__[B_: Basis, M_: BasisMetadata](
        self: AsUpcast[B_, M_],
        basis: B_,
        metadata: M_,
    ) -> None:
        assert basis.metadata() == metadata
        self._basis = cast("B", basis)
        super().__init__(metadata)

    def resolve_ctype[DT_: ctype[Never]](
        self: AsUpcast[Basis[Any, DT_], Any],
    ) -> AsUpcast[B, M, DT_]:
        return cast("AsUpcast[B, M, DT_]", self)

    @property
    def inner(self) -> B:
        """Get the original basis, with a less specific metadata type."""
        return self._basis

    @property
    @override
    def size(self) -> int:
        return self._basis.size

    @property
    @override
    def features(self) -> set[BasisFeature]:
        """Features of the basis."""
        return self._basis.features

    @override
    def __into_fundamental__[DT1: np.generic, DT2: np.generic](
        self: AsUpcast[Basis[Any, Any], Any, ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, np.generic]:
        """Convert a vector in the non-conjugate basis into the fundamental basis."""
        return self._basis.__into_fundamental__(vectors, axis)

    @override
    def __from_fundamental__[DT2: np.generic, DT3: np.generic](
        self: AsUpcast[Basis[Any, Any], Any, ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, DT2, DT3]:
        """Convert a vector into the non-conjugate basis from the fundamental basis."""
        return self._basis.__from_fundamental__(vectors, axis)

    @property
    @override
    def is_dual(self) -> NestedBool:
        return self._basis.is_dual

    @override
    def dual_basis(self) -> Self:
        """Get the dual basis.

        A dual basis is a basis that is conjugate to the current basis, and
        is the basis that is used in the inner product.
        """
        return cast("Self", AsUpcast(self._basis.dual_basis(), self.metadata()))

    @override
    def __convert_vector_into__[
        M_: BasisMetadata,
        DT1: np.generic,
        DT2: np.generic,
        DT3: np.generic,
    ](
        self: AsUpcast[Basis[Any, Any], Any, ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        basis: Basis[M_, ctype[DT3]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return self._basis.__convert_vector_into__(vectors, basis, axis)

    @override
    def add_data[DT_: np.dtype[np.number]](
        self,
        lhs: np.ndarray[Any, DT_],
        rhs: np.ndarray[Any, DT_],
    ) -> np.ndarray[Any, DT_]:
        return self._basis.add_data(lhs, rhs)

    @override
    def sub_data[DT_: np.dtype[np.number]](
        self,
        lhs: np.ndarray[Any, DT_],
        rhs: np.ndarray[Any, DT_],
    ) -> np.ndarray[Any, DT_]:
        return self._basis.sub_data(lhs, rhs)

    @override
    def mul_data[DT_: np.dtype[np.number]](
        self,
        lhs: np.ndarray[Any, DT_],
        rhs: float,
    ) -> np.ndarray[Any, DT_]:
        return self._basis.mul_data(lhs, rhs)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        return self._basis.points
