from __future__ import annotations

from typing import Any, Literal, Never, Self, TypeGuard, cast, overload, override

import numpy as np
import scipy.fft  # type: ignore lib types

from slate_core.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    Ctype,
    NestedBoolOrNone,
)
from slate_core.basis._fundamental import FundamentalBasis
from slate_core.basis._tuple import TupleBasis
from slate_core.basis._wrapped import WrappedBasis
from slate_core.metadata._metadata import BasisMetadata, SimpleMetadata
from slate_core.metadata._tuple import AnyMetadata, TupleMetadata, is_tuple_metadata

TrigonometricTransformType = Literal["type 1", "type 2", "type 4"]
TrigonometricTransformFunction = Literal["sin", "cos"]


class TrigonometricTransformBasis[
    B: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[B, CT],
):
    """A basis with states represented by a trigonometric transform.

    This is useful for representing 'Hardwall' boundary conditions, since
    the states have a fixed zero value at the boundaries.
    """

    def __init__[B_: Basis](
        self: TrigonometricTransformBasis[B_, Ctype[Never]],
        inner: B_,
        ty: TrigonometricTransformType = "type 1",
        fn: TrigonometricTransformFunction = "sin",
    ) -> None:
        self._ty: TrigonometricTransformType = ty
        self._fn: TrigonometricTransformFunction = fn
        super().__init__(cast("B", inner))

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def resolve_ctype[DT_: np.generic](
        self: TrigonometricTransformBasis[Basis[Any, Ctype[DT_]], Any],
    ) -> TrigonometricTransformBasis[B, Ctype[DT_]]:
        """Upcast the wrapped basis to a more specific type."""
        # Note that np.complexfloating is the most general set of data
        # that can be used with this basis. real numbers (ie np.floating)
        # are not supported.
        return cast("TrigonometricTransformBasis[B, Ctype[DT_]]", self)

    @property
    def ty(self) -> TrigonometricTransformType:
        """Type of the trigonometric transform basis."""
        return self._ty

    @property
    def _ty_idx(self) -> int:
        return {"type 1": 1, "type 2": 2, "type 3": 3, "type 4": 4}[self.ty]

    @property
    def fn(self) -> TrigonometricTransformFunction:
        """Function type of the trigonometric transform basis."""
        return self._fn

    @override
    def __eq__(self, other: object) -> bool:
        if is_trigonometric_transform_basis(other):
            return (
                other.inner == self.inner
                and self.ty == other.ty
                and self.fn == other.fn
            )
        return False

    @override
    def dual_basis(self) -> Self:
        return super().dual_basis()

    @override
    def __hash__(self) -> int:
        return hash((2, self.inner, self.ty, self.fn))

    @property
    @override
    def size(self) -> int:
        """Number of elements in the basis."""
        return self.inner.size

    def _transform_backward[T: np.dtype[np.generic]](
        self,
        vectors: np.ndarray[Any, T],
        axis: int = -1,
    ) -> np.ndarray[Any, T]:
        return cast(
            "np.ndarray[Any, T]",
            scipy.fft.idct(vectors, axis=axis, type=self._ty_idx, norm="ortho")  # type: ignore lib types
            if self.fn == "cos"
            else scipy.fft.idst(vectors, axis=axis, type=self._ty_idx, norm="ortho"),  # type: ignore lib types
        )

    @override
    def __into_inner__[T1: np.generic, T2: np.generic](
        self: TrigonometricTransformBasis[Basis[Any, Ctype[T2]]],
        vectors: np.ndarray[Any, np.dtype[T1]],
        axis: int = -1,
    ) -> BasisConversion[np.complexfloating, T1, T2]:
        return BasisConversion(lambda: (self._transform_backward(vectors, axis)))

    def _transform_forward[T: np.dtype[np.generic]](
        self,
        vectors: np.ndarray[Any, T],
        axis: int = -1,
    ) -> np.ndarray[Any, T]:
        return cast(
            "np.ndarray[Any, T]",
            scipy.fft.dct(vectors, axis=axis, type=self._ty_idx, norm="ortho")  # type: ignore lib types
            if self.fn == "cos"
            else scipy.fft.dst(vectors, axis=axis, type=self._ty_idx, norm="ortho"),  # type: ignore lib types
        )

    @override
    def __from_inner__[T1: np.generic, T2: np.generic](
        self: TrigonometricTransformBasis[Basis[Any, Ctype[T1]]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, np.complexfloating]:
        return BasisConversion[T1, T2, np.complexfloating](
            lambda: (self._transform_forward(vectors, axis))
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


def is_trigonometric_transform_basis(
    basis: object,
) -> TypeGuard[TrigonometricTransformBasis]:
    """Check if the given basis is a hardwall basis."""
    return isinstance(basis, TrigonometricTransformBasis)


@overload
def trigonometric_transformed_from_metadata[M0: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M0], E],
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[Basis[M0, Ctype[np.generic]]], E, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_metadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    metadata: TupleMetadata[tuple[M0, M1], E],
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[
    tuple[Basis[M0, Ctype[np.generic]], Basis[M1, Ctype[np.generic]]],
    E,
    Ctype[np.generic],
]: ...
@overload
def trigonometric_transformed_from_metadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
](
    metadata: TupleMetadata[tuple[M0, M1, M2], E],
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.generic]],
        Basis[M1, Ctype[np.generic]],
        Basis[M2, Ctype[np.generic]],
    ],
    E,
    Ctype[np.generic],
]: ...
@overload
def trigonometric_transformed_from_metadata[M: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M, ...], E],
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[Basis[M, Ctype[np.generic]], ...], E, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_metadata[M: SimpleMetadata](
    metadata: M,
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TrigonometricTransformBasis[FundamentalBasis[M], Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_metadata[M: AnyMetadata](
    metadata: M,
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> Basis[M, Ctype[np.generic]]: ...


def trigonometric_transformed_from_metadata(
    metadata: AnyMetadata,
    *,
    is_dual: NestedBoolOrNone = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> Basis[AnyMetadata, Ctype[np.generic]]:
    """Get a trigonometric transformed basis with the given metadata."""
    if isinstance(metadata, SimpleMetadata):
        is_dual = False if is_dual is None else is_dual
        assert isinstance(is_dual, bool)
        return TrigonometricTransformBasis(
            FundamentalBasis(metadata, is_dual=is_dual), ty=ty, fn=fn
        ).resolve_ctype()

    assert is_tuple_metadata(metadata)
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(is_dual for _ in metadata.children)
    )

    children = tuple(
        trigonometric_transformed_from_metadata(c, is_dual=dual, ty=ty, fn=fn)
        for (c, dual) in zip(metadata.children, is_dual, strict=False)
    )
    return TupleBasis(children, metadata.extra).resolve_ctype()


@overload
def as_trigonometric_transformed[M0: BasisMetadata, E](
    basis: Basis[TupleMetadata[tuple[M0], E]],
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[Basis[M0, Ctype[np.generic]]], E, Ctype[np.generic]]: ...
@overload
def as_trigonometric_transformed[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    basis: Basis[TupleMetadata[tuple[M0, M1], E]],
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.generic]],
        Basis[M1, Ctype[np.generic]],
    ],
    E,
    Ctype[np.generic],
]: ...
@overload
def as_trigonometric_transformed[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
](
    basis: Basis[TupleMetadata[tuple[M0, M1, M2], E]],
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.generic]],
        Basis[M1, Ctype[np.generic]],
        Basis[M2, Ctype[np.generic]],
    ],
    E,
    Ctype[np.generic],
]: ...
@overload
def as_trigonometric_transformed[M: BasisMetadata, E](
    basis: Basis[TupleMetadata[tuple[M, ...], E]],
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[Basis[M, Ctype[np.generic]], ...], E, Ctype[np.generic]]: ...
@overload
def as_trigonometric_transformed[M: SimpleMetadata](
    basis: Basis[M],
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TrigonometricTransformBasis[FundamentalBasis[M], Ctype[np.generic]]: ...
@overload
def as_trigonometric_transformed[M: AnyMetadata](
    basis: Basis[M],
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> Basis[M, Ctype[np.generic]]: ...
def as_trigonometric_transformed(
    basis: Basis,
    *,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> Basis[AnyMetadata, Ctype[np.generic]]:
    """Get a trigonometric transformed basis with the given metadata."""
    return trigonometric_transformed_from_metadata(
        basis.metadata(), is_dual=basis.is_dual, ty=ty, fn=fn
    )


@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[FundamentalBasis], None, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[FundamentalBasis], E, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[FundamentalBasis, FundamentalBasis], None, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[FundamentalBasis, FundamentalBasis], E, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis],
    None,
    Ctype[np.generic],
]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis],
    E,
    Ctype[np.generic],
]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[FundamentalBasis, ...], None, Ctype[np.generic]]: ...
@overload
def trigonometric_transformed_from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> TupleBasis[tuple[FundamentalBasis, ...], E, Ctype[np.generic]]: ...


def trigonometric_transformed_from_shape(
    shape: tuple[int, ...],
    *,
    extra: Any | None = None,
    is_dual: tuple[bool, ...] | None = None,
    ty: TrigonometricTransformType = "type 1",
    fn: TrigonometricTransformFunction = "sin",
) -> Basis[Any, Ctype[np.generic]]:
    """Create a trigonometric transformed basis from a shape."""
    return trigonometric_transformed_from_metadata(
        TupleMetadata.from_shape(shape, extra=extra), is_dual=is_dual, ty=ty, fn=fn
    )
