from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, TypeGuard, cast, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._tuple import TupleBasis
from slate_core.basis._wrapped import AsUpcast, WrappedBasis
from slate_core.metadata._metadata import BasisMetadata
from slate_core.metadata.util import nx_points

if TYPE_CHECKING:
    from slate_core.metadata._tuple import TupleMetadata


class IsotropicBasis[
    B: TupleBasis[tuple[Basis, Basis], Any],
    CT: Ctype[Never] = Ctype[Never],
](WrappedBasis[B, CT]):
    """Represents an isotropic basis."""

    def __init__[B_: TupleBasis[tuple[Basis, Basis], Any]](
        self: IsotropicBasis[B_], inner: B_
    ) -> None:
        super().__init__(cast("B", inner))
        assert self.inner.children[0].size == self.inner.children[1].size

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def resolve_ctype[CT_: Ctype[Never]](
        self: IsotropicBasis[TupleBasis[tuple[Basis, Basis], Any, CT_], Any],
    ) -> IsotropicBasis[B, CT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("IsotropicBasis[B, CT_]", self)

    @override
    def metadata[M0: BasisMetadata, M1: BasisMetadata, E](
        self: IsotropicBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> TupleMetadata[tuple[M0, M1], E]:
        return self.inner.metadata()

    @override
    def upcast[M0: BasisMetadata, M1: BasisMetadata, E](
        self: IsotropicBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> AsUpcast[IsotropicBasis[B, CT], TupleMetadata[tuple[M0, M1], E], CT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

    @property
    @override
    def size(self) -> int:
        return self.inner.children[0].size

    @override
    def __into_inner__[T1: np.generic, T2: np.generic, T3: np.generic](
        self: WrappedBasis[Basis[Any, Ctype[T3]], Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        def fn() -> np.ndarray[Any, np.dtype[T2]]:
            swapped = vectors.swapaxes(axis, 0)
            indices = nx_points(self.size)
            displacement_matrix = np.mod(indices[:, None] - indices[None, :], self.size)
            return (
                swapped[displacement_matrix]
                .reshape(-1, *swapped.shape[1:])
                .swapaxes(axis, 0)
            )

        return BasisConversion(fn)

    @override
    def __from_inner__[T1: np.generic, T2: np.generic, T3: np.generic](
        self: WrappedBasis[Basis[Any, Ctype[T1]], Ctype[T3]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        def fn() -> np.ndarray[Any, np.dtype[T2]]:
            swapped = vectors.swapaxes(axis, 0)
            stacked = swapped.reshape(self.size, self.size, *swapped.shape[1:])[0]

            return stacked.swapaxes(axis, 0)

        return BasisConversion(fn)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            is_isotropic(other)
            and (other.inner == self.inner)
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


def is_isotropic(
    basis: object,
) -> TypeGuard[IsotropicBasis[TupleBasis[tuple[Basis, Basis], Any]]]:
    """Check if a basis is isotropic."""
    return isinstance(basis, IsotropicBasis)
