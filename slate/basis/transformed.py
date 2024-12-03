from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, Self, cast, overload, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature, NestedBoolOrNone
from slate.basis._tuple import TupleBasis, from_metadata
from slate.basis.wrapped import WrappedBasis
from slate.metadata import (
    AnyMetadata,
    BasisMetadata,
    Metadata1D,
    Metadata2D,
    Metadata3D,
    StackedMetadata,
)

if TYPE_CHECKING:
    from slate.basis._tuple import TupleBasis1D, TupleBasis2D, TupleBasis3D
    from slate.metadata import SimpleMetadata

type TransformDirection = Literal["forward", "backward"]


class TransformedBasis[M: BasisMetadata](
    WrappedBasis[M, np.complex128, Basis[M, np.complex128]],
):
    """Represents a fourier transformed basis."""

    def __init__(
        self,
        inner: Basis[M, np.complex128],
        direction: TransformDirection | None = None,
    ) -> None:
        self._direction: TransformDirection = (
            ("backward" if inner.is_dual is True else "forward")
            if direction is None
            else direction
        )
        super().__init__(inner)

    @property
    def direction(self) -> TransformDirection:
        """Direction of the transformation."""
        return self._direction

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TransformedBasis):
            return (
                self.size == other.size
                and other.inner == self.inner  # type: ignore unknown
                and other.direction == self.direction
            )
        return False

    @override
    def dual_basis(self) -> Self:
        copied = super().dual_basis()
        copied._direction = "backward" if self.direction == "forward" else "forward"  # noqa: SLF001
        return copied

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction))

    @property
    @override
    def size(self) -> int:
        """Number of elements in the basis."""
        return self.inner.size

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
        self, inner: Basis[M1, np.complex128]
    ) -> TransformedBasis[M1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
    ](
        self,
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
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)


@overload
def fundamental_transformed_tuple_basis_from_metadata[M0: SimpleMetadata, E](
    metadata: Metadata1D[M0, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.generic, TransformedBasis[M0], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    E,
](
    metadata: Metadata2D[M0, M1, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[np.generic, TransformedBasis[M0], TransformedBasis[M1], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    M2: SimpleMetadata,
    E,
](
    metadata: Metadata3D[M0, M1, M2, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis3D[
    np.generic,
    TransformedBasis[M0],
    TransformedBasis[M1],
    TransformedBasis[M2],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[M: AnyMetadata, E](
    metadata: StackedMetadata[M, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[M, E, np.generic]: ...


def fundamental_transformed_tuple_basis_from_metadata[M: AnyMetadata, E](
    metadata: StackedMetadata[M, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[M, E, np.complexfloating[Any, Any]]:
    """Get a transformed fundamental basis with the given metadata."""
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(is_dual for _ in metadata.children)
    )
    children = tuple(
        TransformedBasis(from_metadata(c, is_dual=dual))
        for c, dual in zip(metadata.children, is_dual)
    )
    return TupleBasis(children, metadata.extra)


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.complex128, Basis[SimpleMetadata, np.complex128], None]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.complex128, Basis[SimpleMetadata, np.complex128], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis3D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis3D[
    np.complex128,
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    Basis[SimpleMetadata, np.complex128],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis[BasisMetadata, None, np.complex128]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis[BasisMetadata, E, np.complex128]: ...


def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E | None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis[BasisMetadata, E | None, np.complex128]:
    """Get a basis with the basis at idx set to inner."""
    return fundamental_transformed_tuple_basis_from_metadata(
        StackedMetadata.from_shape(shape, extra=extra), is_dual=is_dual
    )
