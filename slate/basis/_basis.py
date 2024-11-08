from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Never, Self

import numpy as np

from slate.metadata import BasisMetadata, SimpleMetadata


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


class FundamentalBasis[M: BasisMetadata](Basis[M, np.generic]):
    """Represents a full fundamental basis."""

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self.fundamental_size

    def __into_fundamental__[DT1: np.generic](
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return vectors

    def __from_fundamental__[DT1: np.generic](
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return vectors

    def __eq__(self, value: object) -> bool:
        if isinstance(value, FundamentalBasis):
            return value.metadata == self.metadata  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash(self.metadata)

    def conjugate_basis(self) -> FundamentalBasis[M]:
        return self

    @staticmethod
    def from_shape(
        shape: tuple[int, ...],
    ) -> FundamentalBasis[SimpleMetadata]:
        """Get a fundamental basis from a shape."""
        return FundamentalBasis(SimpleMetadata(shape))


def basis_as_fundamental[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> FundamentalBasis[M]:
    """Get the fundamental basis for a given basis."""
    return FundamentalBasis(basis.metadata)
