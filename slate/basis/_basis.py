from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import starmap
from typing import (
    Any,
    Literal,
    Never,
    Self,
)

import numpy as np

from slate.metadata._metadata import BasisMetadata
from slate.metadata._shape import NestedLength, size_from_nested_shape

BasisFeature = Literal["ADD", "MUL", "SUB", "LINEAR_MAP", "INDEX"]
"""
Features of a basis.
This specify certain operations that can be performed on the basis.
- ADD: addition of data
- MUL: multiplication of data
- SUB: subtraction of data
- LINEAR_MAP: the basis can be described as the linear map from the full basis
  ie, data_i = sum_j matrix_ij fundamental_data_j
- INDEX: The basis is a simple truncation of a fundamental basis which can be represented as a CoordinateBasis basis.
"""


type NestedBool = bool | tuple[NestedBool, ...]
type NestedBoolOrNone = bool | tuple[NestedBoolOrNone, ...] | None


def are_dual_shapes(lhs: NestedBool, rhs: NestedBool) -> bool:
    """Check if two bases are dual to each other.

    The two basis must have the same shape, otherwise a `ValueError` will be raised.
    """  # noqa: DOC501
    if isinstance(lhs, tuple) and isinstance(rhs, tuple):
        return all(starmap(are_dual_shapes, zip(lhs, rhs, strict=False)))
    if isinstance(lhs, bool) and isinstance(rhs, bool):
        return lhs != rhs
    msg = "The two basis have different shapes"
    raise ValueError(msg)


def are_dual[M: BasisMetadata](lhs: Basis[M, Any], rhs: Basis[M, Any]) -> bool:
    """Check if two bases are dual to each other.

    The two basis must have the same shape, otherwise a `ValueError` will be raised.
    """
    return lhs.metadata() == rhs.metadata() and are_dual_shapes(
        lhs.is_dual, rhs.is_dual
    )


class Basis[M: BasisMetadata, DT: np.generic](ABC):
    """Base class for a basis."""

    def __init__(self, metadata: M) -> None:
        self._metadata = metadata

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of elements in the basis."""

    @property
    def fundamental_size(self) -> int:
        """Size of the full data."""
        return size_from_nested_shape(self.fundamental_shape)

    @property
    def fundamental_shape(self) -> NestedLength:
        """Shape of the full data."""
        return self.metadata().fundamental_shape

    def metadata(self) -> M:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return self._metadata

    @property
    def features(self) -> set[BasisFeature]:
        """Features of the basis."""
        return set()

    @staticmethod
    def _dtype_variance_fn(_a: DT, _b: Never) -> None:
        """
        Fix the variance of DT.

        This is a workaround for the limitation in typing __into_fundamental__
        which should be [DT1: DT]
        """
        return

    @abstractmethod
    def __into_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        """Convert a vector in the non-conjugate basis into the fundamental basis."""

    @abstractmethod
    def __from_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        """Convert a vector into the non-conjugate basis from the fundamental basis."""

    @property
    @abstractmethod
    def is_dual(self) -> NestedBool: ...

    @abstractmethod
    def dual_basis(self) -> Self:
        """Get the dual basis.

        A dual basis is a basis that is conjugate to the current basis, and
        is the basis that is used in the inner product.
        """
        ...

    def __convert_vector_into__[
        DT1: np.generic,
    ](  # [DT1: DT, B1: Basis[M1, DT]]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        basis: Basis[BasisMetadata, Never],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        assert self.metadata() == basis.metadata()

        if self == basis:
            return vectors

        fundamental = self.__into_fundamental__(vectors, axis)
        return basis.__from_fundamental__(fundamental, axis)

    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        msg = "add_data not implemented for this basis"
        raise NotImplementedError(msg)

    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        msg = "sub_data not implemented for this basis"
        raise NotImplementedError(msg)

    def mul_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: float,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        msg = "mul_data not implemented for this basis"
        raise NotImplementedError(msg)

    @property
    def points(
        self,
    ) -> np.ndarray[Any, np.dtype[np.int_]]:
        msg = "points not implemented for this basis, requires the INDEX feature"
        raise NotImplementedError(msg)
