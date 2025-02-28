from __future__ import annotations

from typing import Any, Never, TypeGuard, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisConversion, BasisFeature, ctype
from slate.basis._wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util import pad_ft_points


class CroppedBasis[
    B: Basis[BasisMetadata, ctype[Never]] = Basis[BasisMetadata, ctype[Never]],
    DT: ctype[Never] = ctype[Never],
](WrappedBasis[B, DT]):
    """A Cropped Basis takes the first size states, using the fourier convention."""

    def __init__(self, size: int, inner: B) -> None:
        self._size = size
        super().__init__(inner)

    @property
    @override
    def size(self) -> int:
        return self._size

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, CroppedBasis):
            return (
                self._size == other._size and other._inner == self._inner  # type: ignore unknown
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((self._size, self._inner))

    @override
    def __into_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: CroppedBasis[Basis[Any, ctype[DT3]], ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion(
            lambda: pad_ft_points(vectors, s=(self._inner.size,), axes=(axis,))
        )

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: CroppedBasis[Basis[Any, ctype[DT1]], ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion(
            lambda: pad_ft_points(vectors, s=(self._size,), axes=(axis,))
        )

    @override
    def __convert_vector_into__[
        M_: BasisMetadata,
        DT1: np.generic,
        DT2: np.generic,
        DT3: np.generic,
    ](
        self: CroppedBasis[Basis[M_, ctype[DT1]], ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        basis: Basis[M_, ctype[DT3]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        assert self.metadata() == basis.metadata()

        if self == basis:
            return BasisConversion(lambda: vectors)

        if is_cropped_basis(basis) and self.inner == basis.inner:

            def fn() -> np.ndarray[Any, np.dtype[DT2]]:
                out = pad_ft_points(vectors, s=(basis.size,), axes=(axis,))
                return (
                    cast("Any", np.conj(out))
                    if (self.is_dual != basis.is_dual)
                    else out
                )

            return BasisConversion(fn)

        return WrappedBasis[Basis[M_, ctype[DT1]], ctype[DT1]].__convert_vector_into__(
            self, vectors, basis, axis
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
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
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
            cast("WrappedBasis[Any, ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


def is_cropped_basis(basis: Basis) -> TypeGuard[CroppedBasis]:
    """Is the basis a cropped basis."""
    return isinstance(basis, CroppedBasis)
