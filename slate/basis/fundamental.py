from __future__ import annotations

from typing import Any, Self, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.metadata import BasisMetadata, SimpleMetadata
from slate.metadata.stacked import StackedMetadata


class FundamentalBasis[M: BasisMetadata](Basis[M, np.generic]):
    """Represents a full fundamental basis."""

    def __init__[_M: BasisMetadata](
        self: FundamentalBasis[_M], metadata: _M, *, is_dual: bool = False
    ) -> None:
        super().__init__(metadata, is_dual=is_dual)

    @property
    @override
    def size(self: Self) -> int:
        return self.fundamental_size

    @override
    def __into_fundamental__[DT1: np.generic](
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return np.conj(vectors) if self.is_dual else vectors

    @override
    def __from_fundamental__[DT1: np.generic](
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return np.conj(vectors) if self.is_dual else vectors

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, FundamentalBasis):
            return (
                other.metadata() == self.metadata()  # type: ignore unknown
                and self.is_dual == other.is_dual
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((self.metadata(), self.is_dual))

    @staticmethod
    def from_size(
        size: int,
    ) -> FundamentalBasis[SimpleMetadata]:
        """Get a fundamental basis from a size."""
        return FundamentalBasis(SimpleMetadata(size))

    @staticmethod
    def from_shape(
        shape: tuple[int, ...],
    ) -> FundamentalBasis[StackedMetadata[SimpleMetadata, None]]:
        """Get a fundamental basis from a shape."""
        return FundamentalBasis(StackedMetadata.from_shape(shape))

    @property
    @override
    def features(self) -> set[BasisFeature]:
        return {"ADD", "MUL", "SUB", "SIMPLE_ADD", "SIMPLE_MUL", "SIMPLE_SUB", "INDEX"}

    @override
    def add_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self: Self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self: Self) -> np.ndarray[Any, np.dtype[np.int_]]:
        return np.arange(self.size)


def basis_as_fundamental[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> FundamentalBasis[M]:
    """Get the fundamental basis for a given basis."""
    return FundamentalBasis(basis.metadata())
