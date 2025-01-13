from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self, cast, overload, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature, NestedBoolOrNone
from slate.basis._fundamental import FundamentalBasis
from slate.basis._tuple import TupleBasis, tuple_basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import (
    AnyMetadata,
    BasisMetadata,
    Metadata1D,
    Metadata2D,
    Metadata3D,
    SimpleMetadata,
    StackedMetadata,
)

if TYPE_CHECKING:
    from slate.basis._tuple import TupleBasis1D, TupleBasis2D, TupleBasis3D


type TransformDirection = Literal["forward", "backward"]


class TransformedBasis[M: BasisMetadata](
    WrappedBasis[M, np.complexfloating, Basis[M, np.complexfloating]],
):
    """Represents a fourier transformed basis."""

    def __init__(
        self,
        inner: Basis[M, np.complexfloating],
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
    def _transform_backward[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        cls,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return cast(
            "np.ndarray[Any, np.dtype[DT1]]",
            np.fft.ifft(vectors, axis=axis, norm="ortho"),
        )

    @classmethod
    def _transform_forward[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        cls,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return cast(
            "np.ndarray[Any, np.dtype[DT1]]",
            np.fft.fft(vectors, axis=axis, norm="ortho"),
        )

    @override
    def __into_inner__[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
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
    def __from_inner__[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return (
            self._transform_forward(vectors, axis)
            if self.direction == "forward"
            else self._transform_backward(vectors, axis)
        )

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)


@overload
def fundamental_transformed_tuple_basis_from_metadata[M0: SimpleMetadata, E](
    metadata: Metadata1D[M0, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.generic, TransformedBasis[M0], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[M0: BasisMetadata, E](
    metadata: Metadata1D[M0, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.generic, Basis[M0, np.complexfloating], E]: ...


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
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    metadata: Metadata2D[M0, M1, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[
    np.generic, Basis[M0, np.complexfloating], Basis[M1, np.complexfloating], E
]: ...


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
def fundamental_transformed_tuple_basis_from_metadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
](
    metadata: Metadata3D[M0, M1, M2, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis3D[
    np.generic,
    Basis[M0, np.complexfloating],
    Basis[M1, np.complexfloating],
    Basis[M2, np.complexfloating],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[M: SimpleMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> TransformedBasis[FundamentalBasis[M]]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[M: AnyMetadata, E](
    metadata: StackedMetadata[M, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[M, E, np.complexfloating]: ...


@overload
def fundamental_transformed_tuple_basis_from_metadata[M: AnyMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> Basis[M, np.complexfloating]: ...


def fundamental_transformed_tuple_basis_from_metadata(
    metadata: AnyMetadata, *, is_dual: NestedBoolOrNone = None
) -> Basis[AnyMetadata, np.complexfloating[Any, Any]]:
    """Get a transformed fundamental basis with the given metadata."""
    if isinstance(metadata, SimpleMetadata):
        is_dual = False if is_dual is None else is_dual
        assert isinstance(is_dual, bool)
        return TransformedBasis(FundamentalBasis(metadata, is_dual=is_dual))

    metadata = cast("StackedMetadata[AnyMetadata, Any]", metadata)
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(is_dual for _ in metadata.children)
    )

    children = tuple(
        fundamental_transformed_tuple_basis_from_metadata(c, is_dual=dual)
        for (c, dual) in zip(metadata.children, is_dual, strict=False)
    )
    return tuple_basis(children, metadata.extra)


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[
    np.complexfloating, Basis[SimpleMetadata, np.complexfloating], None
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.complexfloating, Basis[SimpleMetadata, np.complexfloating], E]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[
    np.complexfloating,
    Basis[SimpleMetadata, np.complexfloating],
    Basis[SimpleMetadata, np.complexfloating],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[
    np.complexfloating,
    Basis[SimpleMetadata, np.complexfloating],
    Basis[SimpleMetadata, np.complexfloating],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis3D[
    np.complexfloating,
    Basis[SimpleMetadata, np.complexfloating],
    Basis[SimpleMetadata, np.complexfloating],
    Basis[SimpleMetadata, np.complexfloating],
    None,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis3D[
    np.complexfloating,
    Basis[SimpleMetadata, np.complexfloating],
    Basis[SimpleMetadata, np.complexfloating],
    Basis[SimpleMetadata, np.complexfloating],
    E,
]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis[BasisMetadata, None, np.complexfloating]: ...


@overload
def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E, is_dual: NestedBoolOrNone = None
) -> TupleBasis[BasisMetadata, E, np.complexfloating]: ...


def fundamental_transformed_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E | None = None, is_dual: NestedBoolOrNone = None
) -> TupleBasis[BasisMetadata, E | None, np.complexfloating]:
    """Get a basis with the basis at idx set to inner."""
    return fundamental_transformed_tuple_basis_from_metadata(
        StackedMetadata.from_shape(shape, extra=extra), is_dual=is_dual
    )
