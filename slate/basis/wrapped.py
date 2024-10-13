from __future__ import annotations

from abc import abstractmethod
from typing import Any, Self, override

import numpy as np

from slate.basis import Basis
from slate.basis.metadata import BasisMetadata


class WrappedBasis[_M: BasisMetadata, _DT: np.generic](Basis[_M, _DT]):
    """Represents a truncated basis."""

    def __init__(self: Self, inner: Basis[_M, _DT]) -> None:
        self._inner = inner
        self._metadata = inner.metadata

    @property
    def inner(self: Self) -> Basis[_M, _DT]:
        """Inner basis."""
        return self._inner

    @abstractmethod
    def __into_inner__(
        self,
        vectors: np.ndarray[
            Any,
            np.dtype[_DT],
        ],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]: ...

    @abstractmethod
    def __from_inner__(
        self,
        vectors: np.ndarray[
            Any,
            np.dtype[_DT],
        ],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]: ...

    def __into_fundamental__(
        self,
        vectors: np.ndarray[
            Any,
            np.dtype[_DT],
        ],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        transformed = self.__into_inner__(vectors, axis)
        return self._inner.__into_fundamental__(transformed, axis=axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[
            Any,
            np.dtype[_DT],
        ],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        transformed = self._inner.__from_fundamental__(vectors, axis=axis)
        return self.__from_inner__(transformed, axis)

    @override
    def __convert_vector_into__(
        self,
        vectors: np.ndarray[Any, np.dtype[_DT]],
        basis: Basis[_M, _DT],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[_DT]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors

        as_inner = self.__into_inner__(vectors, axis)

        if isinstance(basis, WrappedBasis) and self.inner == basis.inner:
            return basis.__from_inner__(as_inner, axis)
        return self._inner.__convert_vector_into__(as_inner, basis, axis=axis)
