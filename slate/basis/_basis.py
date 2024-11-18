from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Literal,
    Never,
    Self,
)

import numpy as np

from slate.metadata import BasisMetadata

BasisFeatures = Literal["ADD", "MUL", "SUB", "SIMPLE_ADD", "SIMPLE_MUL", "SIMPLE_SUB"]


class Basis[M: BasisMetadata, DT: np.generic](ABC):
    """Base class for a basis."""

    def __init__(self, metadata: M) -> None:
        self._metadata = metadata

    @property
    @abstractmethod
    def size(self: Self) -> int:
        """Number of elements in the basis."""

    @property
    def fundamental_size(self: Self) -> int:
        """Size of the full data."""
        return np.prod(self.metadata.fundamental_shape).item()

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Shape of the full data."""
        return self.metadata.fundamental_shape

    @property
    def n_dim(self: Self) -> int:
        """Number of dimensions of the full data."""
        return len(self.fundamental_shape)

    @property
    def metadata(self: Self) -> M:
        """Metadata associated with the basis."""
        return self._metadata

    @property
    def features(self: Self) -> set[BasisFeatures]:
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
    ) -> np.ndarray[Any, np.dtype[DT1]]: ...

    @abstractmethod
    def __from_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]: ...

    @abstractmethod
    def conjugate_basis(self) -> Basis[M, DT]: ...

    def __convert_vector_into__[
        DT1: np.generic,
    ](  # [DT1: DT, B1: Basis[M1, DT]]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        basis: Basis[BasisMetadata, Never],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors

        fundamental = self.__into_fundamental__(vectors, axis)
        return basis.__from_fundamental__(fundamental, axis)

    def add_data[DT1: np.number[Any]](  # noqa: PLR6301
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        msg = "add_data not implemented for this basis"
        raise NotImplementedError(msg)

    def sub_data[DT1: np.number[Any]](  # noqa: PLR6301
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        msg = "sub_data not implemented for this basis"
        raise NotImplementedError(msg)

    def mul_data[DT1: np.number[Any]](  # noqa: PLR6301
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: float,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        msg = "mul_data not implemented for this basis"
        raise NotImplementedError(msg)
