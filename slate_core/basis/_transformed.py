from __future__ import annotations

from typing import Any, Literal, Never, Self, TypeGuard, cast, overload, override

import numpy as np

from slate_core.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    Ctype,
    NestedBoolOrNone,
    UnionCtype,
)
from slate_core.basis._fundamental import FundamentalBasis
from slate_core.basis._tuple import TupleBasis
from slate_core.basis._wrapped import AsUpcast, WrappedBasis
from slate_core.metadata import (
    AnyMetadata,
    BasisMetadata,
    SimpleMetadata,
    TupleMetadata,
)
from slate_core.metadata._tuple import is_tuple_metadata

type TransformDirection = Literal["forward", "backward"]

type ComplexCtype[T: np.complexfloating] = Ctype[T]


class TransformedBasis[
    B: Basis = Basis,
    CT: ComplexCtype[Never] = ComplexCtype[Never],
](
    WrappedBasis[B, CT],
):
    """Represents a fourier transformed basis."""

    def __init__[B_: Basis](
        self: TransformedBasis[B_, Ctype[Never]],
        inner: B,
        direction: TransformDirection | None = None,
    ) -> None:
        self._direction: TransformDirection = (
            ("backward" if inner.is_dual is True else "forward")
            if direction is None
            else direction
        )
        super().__init__(inner)

    @property
    @override
    def ctype(self) -> CT:
        return cast(
            "CT",
            UnionCtype((self.inner.ctype, Ctype(np.complexfloating))),
        )

    @override
    def upcast[M: BasisMetadata](
        self: TransformedBasis[Basis[M]],
    ) -> AsUpcast[TransformedBasis[B, CT], M, CT]:
        return cast("AsUpcast[TransformedBasis[B, CT], M, CT]", super().upcast())

    @override
    def resolve_ctype[DT_: np.complexfloating](
        self: TransformedBasis[Basis[Any, ComplexCtype[DT_]], Any],
    ) -> TransformedBasis[B, ComplexCtype[DT_]]:
        """Upcast the wrapped basis to a more specific type."""
        # Note that np.complexfloating is the most general set of data
        # that can be used with this basis. real numbers (ie np.floating)
        # are not supported.
        return cast("TransformedBasis[B, Ctype[DT_]]", self)

    @property
    def direction(self) -> TransformDirection:
        """Direction of the transformation."""
        return self._direction

    @override
    def __eq__(self, other: object) -> bool:
        if is_transformed(other):
            return (
                self.size == other.size
                and other.inner == self.inner
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
    def _transform_backward[T: np.dtype[np.generic]](
        cls,
        vectors: np.ndarray[Any, T],
        axis: int = -1,
    ) -> np.ndarray[Any, T]:
        return cast(
            "np.ndarray[Any, T]",
            np.fft.ifft(vectors, axis=axis, norm="ortho"),
        )

    @classmethod
    def _transform_forward[T: np.dtype[np.generic]](
        cls,
        vectors: np.ndarray[Any, T],
        axis: int = -1,
    ) -> np.ndarray[Any, T]:
        return cast(
            "np.ndarray[Any, T]",
            np.fft.fft(vectors, axis=axis, norm="ortho"),
        )

    @override
    def __into_inner__[T1: np.generic, T2: np.generic](
        self: TransformedBasis[Basis[Any, Ctype[T2]]],
        vectors: np.ndarray[Any, np.dtype[T1]],
        axis: int = -1,
    ) -> BasisConversion[np.complexfloating, T1, T2]:
        return BasisConversion(
            lambda: (
                self._transform_backward(vectors, axis)
                if self.direction == "forward"
                else self._transform_forward(vectors, axis)
            )
        )

    @override
    def __from_inner__[T1: np.generic, T2: np.generic](
        self: TransformedBasis[Basis[Any, Ctype[T1]]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, np.complexfloating]:
        return BasisConversion[T1, T2, np.complexfloating](
            lambda: (
                self._transform_forward(vectors, axis)
                if self.direction == "forward"
                else self._transform_backward(vectors, axis)
            )
        )

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        return out

    @override
    def add_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
    ) -> np.ndarray[Any, np.dtype[T]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[T: np.number](
        self, lhs: np.ndarray[Any, np.dtype[T]], rhs: complex
    ) -> np.ndarray[Any, np.dtype[T]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
    ) -> np.ndarray[Any, np.dtype[T]]:
        if "LINEAR_MAP" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)


def is_transformed(
    basis: object,
) -> TypeGuard[TransformedBasis]:
    """Check if the given basis is a transformed basis."""
    return isinstance(basis, TransformedBasis)


@overload
def transformed_from_metadata[M0: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M0], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[Basis[M0, Ctype[np.complexfloating]]], E, Ctype[np.complexfloating]
]: ...


@overload
def transformed_from_metadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    metadata: TupleMetadata[tuple[M0, M1], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.complexfloating]],
        Basis[M1, Ctype[np.complexfloating]],
    ],
    E,
    Ctype[np.complexfloating],
]: ...


@overload
def transformed_from_metadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
](
    metadata: TupleMetadata[tuple[M0, M1, M2], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.complexfloating]],
        Basis[M1, Ctype[np.complexfloating]],
        Basis[M2, Ctype[np.complexfloating]],
    ],
    E,
    Ctype[np.complexfloating],
]: ...


@overload
def transformed_from_metadata[M: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M, ...], E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[
    tuple[Basis[M, Ctype[np.complexfloating]], ...], E, Ctype[np.complexfloating]
]: ...


@overload
def transformed_from_metadata[M: SimpleMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> TransformedBasis[FundamentalBasis[M], Ctype[np.complexfloating]]: ...


@overload
def transformed_from_metadata[M: AnyMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> Basis[M, Ctype[np.complexfloating]]: ...


def transformed_from_metadata(
    metadata: AnyMetadata, *, is_dual: NestedBoolOrNone = None
) -> Basis[AnyMetadata, Ctype[np.complexfloating]]:
    """Get a transformed fundamental basis with the given metadata."""
    if isinstance(metadata, SimpleMetadata):
        is_dual = False if is_dual is None else is_dual
        assert isinstance(is_dual, bool)
        return TransformedBasis(
            FundamentalBasis(metadata, is_dual=is_dual)
        ).resolve_ctype()

    assert is_tuple_metadata(metadata)
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(is_dual for _ in metadata.children)
    )

    children = tuple(
        transformed_from_metadata(c, is_dual=dual)
        for (c, dual) in zip(metadata.children, is_dual, strict=False)
    )
    return TupleBasis(children, metadata.extra).resolve_ctype()


@overload
def as_transformed[M0: BasisMetadata, E](
    basis: Basis[TupleMetadata[tuple[M0], E]],
) -> TupleBasis[
    tuple[Basis[M0, Ctype[np.complexfloating]]], E, Ctype[np.complexfloating]
]: ...


@overload
def as_transformed[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    basis: Basis[TupleMetadata[tuple[M0, M1], E]],
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.complexfloating]],
        Basis[M1, Ctype[np.complexfloating]],
    ],
    E,
    Ctype[np.complexfloating],
]: ...


@overload
def as_transformed[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
](
    basis: Basis[TupleMetadata[tuple[M0, M1, M2], E]],
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.complexfloating]],
        Basis[M1, Ctype[np.complexfloating]],
        Basis[M2, Ctype[np.complexfloating]],
    ],
    E,
    Ctype[np.complexfloating],
]: ...


@overload
def as_transformed[M: BasisMetadata, E](
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
) -> TupleBasis[
    tuple[Basis[M, Ctype[np.complexfloating]], ...], E, Ctype[np.complexfloating]
]: ...


@overload
def as_transformed[M: SimpleMetadata](
    basis: Basis[M],
) -> TransformedBasis[FundamentalBasis[M], Ctype[np.complexfloating]]: ...


@overload
def as_transformed[M: AnyMetadata](
    basis: Basis[M],
) -> Basis[M, Ctype[np.complexfloating]]: ...


def as_transformed(
    basis: Basis,
) -> Basis[AnyMetadata, Ctype[np.complexfloating]]:
    return transformed_from_metadata(basis.metadata(), is_dual=basis.is_dual)


@overload
def transformed_from_shape[E](
    shape: tuple[int], *, extra: None = None, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[tuple[FundamentalBasis], None, Ctype[np.complexfloating]]: ...
@overload
def transformed_from_shape[E](
    shape: tuple[int], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[tuple[FundamentalBasis], E, Ctype[np.complexfloating]]: ...


@overload
def transformed_from_shape[E](
    shape: tuple[int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis], None, Ctype[np.complexfloating]
]: ...
@overload
def transformed_from_shape[E](
    shape: tuple[int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis], E, Ctype[np.complexfloating]
]: ...


@overload
def transformed_from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis],
    None,
    Ctype[np.complexfloating],
]: ...
@overload
def transformed_from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis],
    E,
    Ctype[np.complexfloating],
]: ...


@overload
def transformed_from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, ...], None, Ctype[np.complexfloating]]: ...
@overload
def transformed_from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, ...], E, Ctype[np.complexfloating]]: ...


def transformed_from_shape(
    shape: tuple[int, ...],
    *,
    extra: Any | None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> Basis[Any, Ctype[np.complexfloating]]:
    """Get a basis with the basis at idx set to inner."""
    return transformed_from_metadata(
        TupleMetadata.from_shape(shape, extra=extra), is_dual=is_dual
    )
