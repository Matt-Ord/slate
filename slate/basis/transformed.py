from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Self, cast, overload, override

import numpy as np

from slate.basis import Basis
from slate.basis._basis import FundamentalBasis
from slate.basis.stacked._tuple_basis import TupleBasis, VariadicTupleBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.metadata.stacked.stacked import StackedMetadata

if TYPE_CHECKING:
    from slate.metadata._metadata import SimpleMetadata

type TransformDirection = Literal["forward", "backward"]


class TransformedBasis[M: BasisMetadata](
    WrappedBasis[M, np.complex128, Basis[M, np.complex128]]
):
    """Represents a fourier transformed basis."""

    def __init__(
        self: Self,
        inner: Basis[M, np.complex128],
        *,
        direction: TransformDirection = "forward",
    ) -> None:
        self._direction: TransformDirection = direction
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
    def direction(self: Self) -> TransformDirection:
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


def fundamental_transformed_tuple_basis_from_metadata[M: BasisMetadata, E](
    metadata: StackedMetadata[M, E],
) -> TupleBasis[M, E, np.complexfloating[Any, Any]]:
    """Get a transformed fundamental basis with the given metadata."""
    children = tuple(TransformedBasis(FundamentalBasis(c)) for c in metadata.children)
    return TupleBasis(children, metadata.extra)


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: None = None
) -> VariadicTupleBasis[np.complex128, Basis[SimpleMetadata, np.complex128], None]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: E
) -> VariadicTupleBasis[np.complex128, Basis[SimpleMetadata, np.complex128], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: None = None
) -> VariadicTupleBasis[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: E
) -> VariadicTupleBasis[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: None = None
) -> VariadicTupleBasis[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: E
) -> VariadicTupleBasis[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: None = None
) -> TupleBasis[SimpleMetadata, None, np.complex128]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E
) -> TupleBasis[SimpleMetadata, E, np.complex128]: ...


def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E | None = None
) -> TupleBasis[SimpleMetadata, E | None, np.complex128]:
    """Get a basis with the basis at idx set to inner."""
    return fundamental_transformed_tuple_basis_from_metadata(
        StackedMetadata.from_shape(shape, extra=extra)
    )
