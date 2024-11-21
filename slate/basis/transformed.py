from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Self, cast, overload, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis.fundamental import FundamentalBasis
from slate.basis.stacked import TupleBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.metadata.stacked.stacked import StackedMetadata

if TYPE_CHECKING:
    from slate.basis.stacked._tuple import TupleBasis1D, TupleBasis2D, TupleBasis3D
    from slate.metadata import SimpleMetadata

type TransformDirection = Literal["forward", "backward"]


class TransformedBasis[M: BasisMetadata](
    WrappedBasis[M, np.complex128, Basis[M, np.complex128]],
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

    @override
    def __eq__(self, value: object) -> bool:
        if isinstance(value, TransformedBasis):
            return (
                self.size == value.size
                and value.inner == self.inner  # type: ignore unknown
                and value.direction == self.direction
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction))

    @override
    def conjugate_basis(self) -> TransformedBasis[M]:
        return TransformedBasis(
            self.inner,
            direction="forward" if self.direction == "backward" else "backward",
        )

    @property
    @override
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

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "SIMPLE_ADD" in self.inner.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.inner.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.inner.features:
            out.add("SUB")
            out.add("SIMPLE_SUB")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self: Self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)


def fundamental_transformed_tuple_basis_from_metadata[M: BasisMetadata, E](
    metadata: StackedMetadata[M, E],
) -> TupleBasis[M, E, np.complexfloating[Any, Any]]:
    """Get a transformed fundamental basis with the given metadata."""
    children = tuple(TransformedBasis(FundamentalBasis(c)) for c in metadata.children)
    return TupleBasis(children, metadata.extra)


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: None = None
) -> TupleBasis1D[np.complex128, Basis[SimpleMetadata, np.complex128], None]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: E
) -> TupleBasis1D[np.complex128, Basis[SimpleMetadata, np.complex128], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: None = None
) -> TupleBasis2D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: E
) -> TupleBasis2D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: None = None
) -> TupleBasis3D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: E
) -> TupleBasis3D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: None = None
) -> TupleBasis[BasisMetadata, None, np.complex128]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E
) -> TupleBasis[BasisMetadata, E, np.complex128]: ...


def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E | None = None
) -> TupleBasis[BasisMetadata, E | None, np.complex128]:
    """Get a basis with the basis at idx set to inner."""
    return fundamental_transformed_tuple_basis_from_metadata(
        StackedMetadata.from_shape(shape, extra=extra)
    )
