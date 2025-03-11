from __future__ import annotations

import copy
from typing import Any, Self, TypeGuard, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._basis_state_metadata import BasisStateMetadata, SimpleMetadata


class FundamentalBasis[M: SimpleMetadata = SimpleMetadata](Basis[M, Ctype[np.generic]]):
    """Represents a full fundamental basis."""

    def __init__[M_: SimpleMetadata](
        self: FundamentalBasis[M_], metadata: M_, *, is_dual: bool = False
    ) -> None:
        self._is_dual = is_dual
        super().__init__(metadata)

    @property
    @override
    def ctype(self) -> Ctype[np.generic]:
        return Ctype(np.generic)

    @property
    @override
    def is_dual(self) -> bool:
        return self._is_dual

    @override
    def dual_basis(self) -> Self:
        copied = copy.copy(self)
        copied._is_dual = not copied.is_dual  # noqa: SLF001
        return copied

    @property
    @override
    def fundamental_shape(self) -> int:
        return self.metadata().fundamental_shape

    @property
    @override
    def size(self) -> int:
        return self.fundamental_size

    @override
    def __into_fundamental__[DT2: np.generic](
        self,
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, DT2, np.generic]:
        return BasisConversion[np.generic, DT2, np.generic](
            lambda: np.conj(vectors) if self.is_dual else vectors
        )

    @override
    def __from_fundamental__[DT2: np.generic](
        self,
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, DT2, np.generic]:
        return BasisConversion[np.generic, DT2, np.generic](
            lambda: np.conj(vectors) if self.is_dual else vectors
        )

    @override
    def __eq__(self, other: object) -> bool:
        if is_fundamental_basis(other):
            return other.metadata() == self.metadata() and self.is_dual == other.is_dual
        return False

    @override
    def __hash__(self) -> int:
        return hash((self.metadata(), self.is_dual))

    @staticmethod
    def from_size(
        size: int, *, is_dual: bool = False
    ) -> FundamentalBasis[SimpleMetadata]:
        """Get a fundamental basis from a size."""
        return FundamentalBasis(SimpleMetadata(size), is_dual=is_dual)

    @property
    @override
    def features(self) -> set[BasisFeature]:
        return {"ADD", "MUL", "SUB", "LINEAR_MAP", "INDEX"}

    @override
    def add_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        return np.arange(self.size)


# TODO: better name ie cast_fundamental_basis  # noqa: FIX002
def as_state_list[B: Basis](
    basis: B,
) -> FundamentalBasis[BasisStateMetadata[B]]:
    """Get the fundamental basis for a given basis."""
    return FundamentalBasis(BasisStateMetadata(basis))


def is_fundamental_basis(basis: object) -> TypeGuard[FundamentalBasis]:
    """Check if a basis is a fundamental basis."""
    return isinstance(basis, FundamentalBasis)
