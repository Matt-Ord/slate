from __future__ import annotations

from typing import Any, Literal, Never, Self, TypeGuard, cast, override

import numpy as np
import scipy.fft  # type: ignore lib types

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._wrapped import WrappedBasis


class TrigonometricTransformBasis[
    B: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[B, CT],
):
    """Represents a hardwall basis, using sin states which are guaranteed to be zero at the boundary."""

    def __init__[B_: Basis](
        self: TrigonometricTransformBasis[B_, Ctype[Never]],
        inner: B_,
        ty: Literal["sin", "cos"] = "sin",
    ) -> None:
        self._ty: Literal["sin", "cos"] = ty
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
    def ty(self) -> Literal["sin", "cos"]:
        """Type of the trigonometric transform basis."""
        return self._ty

    @override
    def __eq__(self, other: object) -> bool:
        if is_trigonometric_transform_basis(other):
            return other.inner == self.inner and self.ty == other.ty
        return False

    @override
    def dual_basis(self) -> Self:
        return super().dual_basis()

    @override
    def __hash__(self) -> int:
        return hash((2, self.inner, self.ty))

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
            scipy.fft.idct(vectors, axis=axis, norm="ortho")  # type: ignore lib types
            if self.ty == "cos"
            else scipy.fft.idst(vectors, axis=axis, norm="ortho"),  # type: ignore lib types
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
            scipy.fft.dct(vectors, axis=axis, norm="ortho")  # type: ignore lib types
            if self.ty == "cos"
            else scipy.fft.dst(vectors, axis=axis, norm="ortho"),  # type: ignore lib types
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
