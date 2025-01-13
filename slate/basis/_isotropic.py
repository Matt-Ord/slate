from __future__ import annotations

from typing import Any, overload, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis._tuple import TupleBasis2D, tuple_basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata.stacked import Metadata2D
from slate.metadata.util import nx_points


class IsotropicBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[Metadata2D[Any, Any, E], DT, TupleBasis2D[DT, B0, B1, E]],
):
    """Represents an isotropic basis."""

    def __init__[_DT: np.generic, _B0: Basis[Any, Any], _B1: Basis[Any, Any], _E](
        self: IsotropicBasis[_DT, _B0, _B1, _E], inner: TupleBasis2D[_DT, _B0, _B1, _E]
    ) -> None:
        super().__init__(inner)
        assert self.inner.children[0].size == self.inner.children[1].size

    @property
    @override
    def inner(self) -> TupleBasis2D[DT, B0, B1, E]:
        return self._inner

    @property
    @override
    def size(self) -> int:
        return self.inner.children[0].size

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        indices = nx_points(self.size)
        displacement_matrix = np.mod(indices[:, None] - indices[None, :], self.size)
        return (
            swapped[displacement_matrix]
            .reshape(-1, *swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(self.size, self.size, *swapped.shape[1:])[0]

        return stacked.swapaxes(axis, 0)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IsotropicBasis)
            and (other.inner == self.inner)  # type: ignore unknown
            and self.is_dual == other.is_dual
        )

    @override
    def __hash__(self) -> int:
        return hash((2, self.inner, self.is_dual))

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
        return self.__from_inner__(self.inner.points)


@overload
def isotropic_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any]](
    children: tuple[_B0, _B1], extra_metadata: None = None
) -> IsotropicBasis[Any, _B0, _B1, None]: ...


@overload
def isotropic_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any], E](
    children: tuple[_B0, _B1], extra_metadata: E
) -> IsotropicBasis[Any, _B0, _B1, E]: ...


def isotropic_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any], E](
    children: tuple[_B0, _B1], extra_metadata: E | None = None
) -> IsotropicBasis[Any, _B0, _B1, E | None]:
    """Build a VariadicTupleBasis from a tuple."""
    return IsotropicBasis(tuple_basis(children, extra_metadata))
