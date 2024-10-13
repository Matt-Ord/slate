from __future__ import annotations

from typing import Any, Self, override

import numpy as np

from slate.basis.metadata import BasisMetadata
from slate.basis.wrapped import WrappedBasis


class TransformedBasis[_M: BasisMetadata](WrappedBasis[_M, np.complex128]):
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
    def __into_inner__(
        self,
        vectors: np.ndarray[Any, np.dtype[np.complex128]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.fft.ifft(vectors, axis=axis, norm="ortho")

    @override
    def __from_inner__(
        self,
        vectors: np.ndarray[Any, np.dtype[np.complex128]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        return np.fft.fft(vectors, axis=axis, norm="ortho")
