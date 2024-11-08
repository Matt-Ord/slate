from __future__ import annotations

from typing import Any, Callable, Literal, Self, cast, override

import numpy as np

from slate.basis import Basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata

type Direction = Literal["forward", "backward"]


class TransformedBasis[M: BasisMetadata](
    WrappedBasis[M, np.complex128, Basis[M, np.complex128]]
):
    """Represents a fourier transformed basis."""

    def __init__(
        self: Self, inner: Basis[M, np.complex128], *, direction: Direction = "forward"
    ) -> None:
        self._direction: Direction = direction
        super().__init__(inner)

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TransformedBasis):
            return (
                self.size == value.size
                and value.inner == self.inner  # type: ignore unknown
                and value.direction == self.direction
            )
        return False

    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction))

    @override
    def conjugate_basis(self) -> TransformedBasis[M]:
        return TransformedBasis(
            self.inner,
            direction="forward" if self.direction == "backward" else "backward",
        )

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self.inner.size

    @property
    def direction(self: Self) -> Direction:
        """The convention used to select the direction for the forward transform."""
        return self._direction

    @classmethod
    def _transform_backward[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        cls,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return cast(
            np.ndarray[Any, np.dtype[DT1]],
            np.fft.ifft(vectors, axis=axis, norm="ortho"),
        )

    @classmethod
    def _transform_forward[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        cls,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return cast(
            np.ndarray[Any, np.dtype[DT1]],
            np.fft.fft(vectors, axis=axis, norm="ortho"),
        )

    @override
    def __into_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (
            self._transform_backward(vectors, axis)
            if self.direction == "forward"
            else self._transform_forward(vectors, axis)
        )

    @override
    def __from_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (
            self._transform_forward(vectors, axis)
            if self.direction == "forward"
            else self._transform_backward(vectors, axis)
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
        """Get the wrapped basis after wrapper is applied to inner."""
        return TransformedBasis(wrapper(self.inner))
