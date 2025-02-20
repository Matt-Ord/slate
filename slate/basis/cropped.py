from __future__ import annotations

from typing import Any, Never, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature, ctype
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata
from slate.util import pad_ft_points


class CroppedBasis[
    B: Basis[BasisMetadata, ctype[Never]],
    DT: ctype[Never] = ctype[Never],
](WrappedBasis[B, DT]):
    """Represents a cropped basis."""

    def __init__(self, size: int, inner: B) -> None:
        self._size = size
        super().__init__(inner)

    @override
    def try_cast_ctype[DT_: np.generic](
        self,
        ctype: type[DT_],
    ) -> CroppedBasis[B, ctype[DT_]] | None:
        """Try to cast a basis into one which supports the given data type."""
        if self.inner.try_cast_ctype(ctype) is None:
            return None
        return cast("CroppedBasis[B, ctype[DT_]]", self)

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
    def __into_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        return pad_ft_points(vectors, s=(self._inner.size,), axes=(axis,))

    @override
    def __from_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        return pad_ft_points(vectors, s=(self._size,), axes=(axis,))

    @override
    def __convert_vector_into__(
        self,
        vectors: np.ndarray[Any, DT],
        basis: Basis[BasisMetadata, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        assert self.metadata() == basis.metadata()

        if self == basis:
            return vectors

        if isinstance(basis, CroppedBasis) and self.inner == basis.inner:
            out = pad_ft_points(vectors, s=(basis.size,), axes=(axis,))
            return cast("Any", np.conj(out)) if (self.is_dual != basis.is_dual) else out

        return super().__convert_vector_into__(vectors, basis, axis)

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
