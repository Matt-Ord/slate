from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self

import numpy as np

from slate.basis.metadata import BasisMetadata


class Basis[_M: BasisMetadata, _DT: np.generic](ABC):
    """Base class for a basis."""

    def __init__(self, metadata: _M) -> None:
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
        """Number fo dimensions of the full data."""
        return len(self.fundamental_shape)

    @property
    def metadata(self: Self) -> _M:
        """Metadata associated with the basis."""
        return self.metadata

    @abstractmethod
    def __into_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]: ...

    @abstractmethod
    def __from_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]: ...

    def __convert_vector_into__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        basis: Basis[_M, _DT],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors

        fundamental = self.__into_fundamental__(vectors, axis)
        return basis.__from_fundamental__(fundamental, axis)


class FundamentalBasis[_M: BasisMetadata, _DT: np.generic](Basis[_M, _DT]):
    """Represents a full fundamental basis."""

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self.fundamental_size

    def __into_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        return vectors

    def __from_fundamental__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        return vectors

    def __eq__(self, value: object) -> bool:
        if isinstance(value, FundamentalBasis):
            return value.metadata == self.metadata  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash(self.metadata)


def basis_as_fundamental[_M: BasisMetadata, _DT: np.generic](
    basis: Basis[_M, _DT],
) -> FundamentalBasis[_M, _DT]:
    """Get the fundamental basis for a given basis.

    Returns
    -------
    FundamentalBasis[_M, _DT]
    """
    return FundamentalBasis(basis.metadata)
