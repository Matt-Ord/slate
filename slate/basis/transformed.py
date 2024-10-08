from __future__ import annotations

from typing import Any, Self

import numpy as np

from slate.basis.basis import Basis
from slate.basis.metadata import BasisMetadata


class TransformedBasis[_M: BasisMetadata](Basis[_M, np.complex128]):
    """Represents a fourier transformed basis."""

    def __init__(self: Self, inner: Basis[_M, np.complex128]) -> None:
        self._inner = inner
        self._metadata = inner.metadata

    @property
    def inner(self: Self) -> Basis[_M, np.complex128]:
        """Inner basis."""
        return self._inner

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._inner.size

    def __into_fundamental__(
        self,
        vectors: np.ndarray[
            Any,
            np.dtype[np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        transformed = np.fft.ifft(vectors, axis=axis, norm="ortho")
        return self._inner.__into_fundamental__(transformed, axis=axis)

    def __from_fundamental__(
        self,
        vectors: np.ndarray[
            Any,
            np.dtype[np.complex128],
        ],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        transformed = self._inner.__from_fundamental__(vectors, axis=axis)
        return np.fft.fft(transformed, axis=axis, norm="ortho")

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TransformedBasis):
            return value._inner == self._inner  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash((1, self._inner))
