from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, Never, Self, override

import numpy as np

from slate.basis import Basis
from slate.metadata import BasisMetadata


class WrappedBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Never] = Basis[M, DT],  # : Basis[M, DT]  # noqa: E251
](Basis[M, DT]):
    """Represents a truncated basis."""

    def __init__(self: Self, inner: Basis[M, DT]) -> None:
        self._inner = inner
        self._metadata = inner.metadata

    @property
    def inner(self: Self) -> Basis[M, DT]:
        """Inner basis."""
        return self._inner

    @abstractmethod
    def with_rewrapped_inner(
        self: Self, wrapper: Callable[[B], B]
    ) -> WrappedBasis[M, DT, B]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        WrappedBasis[M, DT]
        """

    @abstractmethod
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]: ...

    @abstractmethod
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]: ...

    def __into_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = self.__into_inner__(vectors, axis)
        return self._inner.__into_fundamental__(transformed, axis=axis)

    def __from_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = self._inner.__from_fundamental__(vectors, axis=axis)
        return self.__from_inner__(transformed, axis)

    @override
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

        as_inner = self.__into_inner__(vectors, axis)

        if isinstance(basis, WrappedBasis) and self.inner == basis.inner:
            return basis.__from_inner__(as_inner, axis)
        return self._inner.__convert_vector_into__(as_inner, basis, axis=axis)  # type: ignore unknown
