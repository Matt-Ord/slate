from __future__ import annotations

from typing import Any, Never, TypeGuard, cast, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._wrapped import WrappedBasis
from slate_core.metadata import BasisMetadata
from slate_core.util import pad_ft_points


class CroppedBasis[B: Basis = Basis, CT: Ctype[Never] = Ctype[Never]](
    WrappedBasis[B, CT]
):
    """A Cropped Basis takes the first size states, using the fourier convention."""

    def __init__(self, size: int, inner: B) -> None:
        self._size = size
        assert size <= inner.fundamental_size, (
            "CroppedBasis size must be smaller than the inner basis fundamental size."
        )
        super().__init__(inner)

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @property
    @override
    def size(self) -> int:
        return self._size

    @override
    def __eq__(self, other: object) -> bool:
        if is_cropped(other):
            return self._size == other._size and other._inner == self._inner
        return False

    @override
    def __hash__(self) -> int:
        return hash((self._size, self._inner))

    @override
    def __into_inner__[T1: np.generic, T2: np.generic, T3: np.generic](
        self: CroppedBasis[Basis[Any, Ctype[T3]], Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        return BasisConversion(
            lambda: pad_ft_points(vectors, s=(self._inner.size,), axes=(axis,))
        )

    @override
    def __from_inner__[T1: np.generic, T2: np.generic, T3: np.generic](
        self: CroppedBasis[Basis[Any, Ctype[T1]], Ctype[T3]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        return BasisConversion(
            lambda: pad_ft_points(vectors, s=(self._size,), axes=(axis,))
        )

    @override
    def __convert_vector_into__[
        M_: BasisMetadata,
        T1: np.generic,
        T2: np.generic,
        T3: np.generic,
    ](
        self: CroppedBasis[Basis[M_, Ctype[T1]], Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        basis: Basis[M_, Ctype[T3]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        assert self.metadata() == basis.metadata()

        if self == basis:
            return BasisConversion(lambda: vectors)

        if is_cropped(basis) and self.inner == basis.inner:

            def fn() -> np.ndarray[Any, np.dtype[T2]]:
                out = pad_ft_points(vectors, s=(basis.size,), axes=(axis,))
                return (
                    cast("Any", np.conj(out))
                    if (self.is_dual != basis.is_dual)
                    else out
                )

            return BasisConversion(fn)
        basis = cast("Basis[M_, Ctype[T3]]", basis)
        return WrappedBasis.__convert_vector_into__(self, vectors, basis, axis)  # type: ignore bad inference

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
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

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)

        return (
            cast("WrappedBasis[Any, Ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


def is_cropped(basis: object) -> TypeGuard[CroppedBasis]:
    """Is the basis a cropped basis."""
    return isinstance(basis, CroppedBasis)
