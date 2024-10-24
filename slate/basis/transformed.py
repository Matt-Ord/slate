from __future__ import annotations

from typing import Any, Callable, Self, cast, override

import numpy as np

from slate.basis import Basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata


class TransformedBasis[M: BasisMetadata](
    WrappedBasis[M, np.complex128, Basis[M, np.complex128]]
):
    """Represents a fourier transformed basis."""

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TransformedBasis):
            return self.size == value.size and value.inner == self.inner  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash((1, self.inner))

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self.inner.size

    @override
    def __into_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return cast(
            np.ndarray[Any, np.dtype[DT1]],
            np.fft.ifft(vectors, axis=axis, norm="ortho"),
        )

    @override
    def __from_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return cast(
            np.ndarray[Any, np.dtype[DT1]],
            np.fft.fft(vectors, axis=axis, norm="ortho"),
        )

    @override
    def with_inner[M1: BasisMetadata](
        self: Self, inner: Basis[M1, np.complex128]
    ) -> TransformedBasis[M1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
    ](
        self: Self,
        wrapper: Callable[[Basis[M, np.complex128]], Basis[M1, np.complex128]],
    ) -> TransformedBasis[M1]:
        """Get the wrapped basis after wrapper is applied to inner.

        Returns
        -------
        TruncatedBasis[M, DT]
        """
        return TransformedBasis(wrapper(self.inner))
