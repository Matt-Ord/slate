from __future__ import annotations

from typing import Any, Callable, Self, cast, overload, override

import numpy as np

from slate.basis._basis import Basis, BasisFeatures
from slate.basis.wrapped import WrappedBasis
from slate.metadata import StackedMetadata
from slate.metadata.util import nx_points

from ._tuple import VariadicTupleBasis


class IsotropicBasis[
    DT: np.generic,
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[StackedMetadata[Any, E], DT, VariadicTupleBasis[DT, Any, Any, E]],
):
    """Represents an isotropic basis."""

    def __init__(self: Self, inner: VariadicTupleBasis[DT, Any, Any, E]) -> None:
        super().__init__(inner)
        assert self.inner.children[0].size == self.inner.children[1].size

    def conjugate_basis(self) -> IsotropicBasis[DT, B0, B1, E]:
        return IsotropicBasis(self.inner.conjugate_basis())

    @property
    def inner(self: Self) -> VariadicTupleBasis[DT, Any, Any, E]:
        return cast(VariadicTupleBasis[DT, Any, Any, E], self._inner)

    @property
    def size(self) -> int:
        return self.inner.children[0].size

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
            .reshape(-1, swapped.shape[1:])
            .swapaxes(axis, 0)
        )

    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = vectors.swapaxes(axis, 0)
        stacked = swapped.reshape(self.size, self.size, *swapped.shape[1:])[0]

        return stacked.swapaxes(axis, 0)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, IsotropicBasis) and (other.inner == self.inner)  # type: ignore unknown

    def __hash__(self) -> int:
        return hash((2, self.inner))

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        DT1: np.generic,
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self: Self, inner: VariadicTupleBasis[DT1, B01, B11, E1]
    ) -> IsotropicBasis[DT1, B01, B11, E1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        DT1: np.generic,
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self: Self,
        wrapper: Callable[
            [VariadicTupleBasis[DT, Any, Any, E]], VariadicTupleBasis[DT1, B01, B11, E1]
        ],
    ) -> IsotropicBasis[DT1, B01, B11, E1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return IsotropicBasis(wrapper(self.inner))

    @property
    @override
    def features(self) -> set[BasisFeatures]:
        out = set[BasisFeatures]()
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
    return IsotropicBasis[Any, Any, Any, E | None](
        VariadicTupleBasis(children, extra_metadata)
    )
