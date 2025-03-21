from __future__ import annotations

from copy import copy
from typing import Any, Never, Self, TypeGuard, cast, override

import numpy as np

from slate_core.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    Ctype,
    UnionCtype,
)
from slate_core.basis._util import get_common
from slate_core.basis._wrapped import AsUpcast, WrappedBasis
from slate_core.metadata._metadata import BasisMetadata
from slate_core.util import Padding, pad_along_axis, slice_along_axis


def _get_lhs_vectors[DT1: np.generic](
    basis: SplitBasis[Basis, Basis, Any],
    vectors: np.ndarray[Any, np.dtype[DT1]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT1]]:
    r"""Get the vector corresponding to ..math::`\hat{A}`."""
    start = 0
    end = basis.lhs.size
    return vectors[slice_along_axis(slice(start, end), axis)]


def _get_rhs_vectors[DT1: np.generic](
    basis: SplitBasis[Basis, Basis, Any],
    vectors: np.ndarray[Any, np.dtype[DT1]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT1]]:
    r"""Get the vector corresponding to ..math::`\hat{B}`."""
    start = basis.lhs.size
    return vectors[slice_along_axis(slice(start, None), axis)]


def _into_inner[DT: np.number](
    basis: SplitBasis[
        Basis[BasisMetadata, Ctype[np.generic]],
        Basis[BasisMetadata, Ctype[np.generic]],
        Ctype[np.generic],
    ],
    vectors: np.ndarray[Any, np.dtype[DT]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    lhs_fundamental = basis.lhs.__convert_vector_into__(
        _get_lhs_vectors(basis, vectors, axis),
        cast("Basis[BasisMetadata, Ctype[np.generic]]", basis.inner),
        axis,
    ).ok()
    rhs_fundamental = basis.rhs.__convert_vector_into__(
        _get_rhs_vectors(basis, vectors, axis),
        cast("Basis[BasisMetadata, Ctype[np.generic]]", basis.inner),
        axis,
    ).ok()
    return cast(
        "np.ndarray[Any, np.dtype[DT]]",
        lhs_fundamental + rhs_fundamental,
    )


def _from_inner[DT: np.number](
    basis: SplitBasis[
        Basis[BasisMetadata, Ctype[np.generic]],
        Basis[BasisMetadata, Ctype[np.generic]],
        Ctype[np.generic],
    ],
    vectors: np.ndarray[Any, np.dtype[DT]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    lhs_vector = (
        cast("Basis[BasisMetadata, Ctype[np.generic]]", basis.inner)
        .__convert_vector_into__(vectors, basis.lhs, axis)
        .ok()
    )
    return pad_along_axis(lhs_vector, Padding(basis.size, 1, 0), axis)


class SplitBasis[
    B0: Basis = Basis,
    B1: Basis = Basis,
    CT: Ctype[Never] = Ctype[np.number],
](
    WrappedBasis[Basis, CT],
):
    r"""Represents data in the split basis.

    Seprates an operator into two parts.

    .. math::
        \hat{O} = \hat{A} + \hat{B}

    where :math:`\hat{A}` and :math:`\hat{B}` may be in a different basis.
    """

    def __init__[
        B0_: Basis,
        B1_: Basis,
    ](self: SplitBasis[B0_, B1_, Ctype[Never]], lhs: B0_, rhs: B1_) -> None:
        self._lhs: B0 = lhs
        self._rhs: B1 = rhs
        super().__init__(get_common(lhs, rhs))

    @property
    @override
    def ctype(self) -> CT:
        return cast(
            "CT",
            UnionCtype((self.lhs.ctype, self.rhs.ctype)),
        )

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: SplitBasis[Basis[BasisMetadata, DT_], Basis[BasisMetadata, DT_], Any],
    ) -> SplitBasis[B0, B1, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("SplitBasis[B0, B1, DT_]", self)

    @override
    def upcast[M: BasisMetadata](
        self: SplitBasis[Basis[M], Basis[M]],
    ) -> AsUpcast[SplitBasis[B0, B1, CT], M, CT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

    @property
    def lhs(self) -> B0:
        """Left hand side of the split basis."""
        return self._lhs

    @property
    def rhs(self) -> B1:
        """Right hand side of the split basis."""
        return self._rhs

    @override
    def __eq__(self, other: object) -> bool:
        if is_split_basis(other):
            return other.lhs == self.lhs and other.rhs == self.rhs
        return False

    @override
    def dual_basis(self) -> Self:
        copied = copy(self)
        copied._lhs = self._lhs.dual_basis()  # noqa: SLF001
        copied._rhs = self._rhs.dual_basis()  # noqa: SLF001

        return copied

    @override
    def __hash__(self) -> int:
        return hash((1, hash(self._lhs), hash(self._rhs)))

    @property
    @override
    def size(self) -> int:
        return self.lhs.size + self.rhs.size

    @override
    def __into_inner__[DT1: np.number, DT2: np.generic, DT3: np.generic](
        self: SplitBasis[
            Basis[BasisMetadata, Ctype[DT3]],
            Basis[BasisMetadata, Ctype[DT3]],
            Ctype[DT1],
        ],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion(lambda: _into_inner(self, vectors, axis))  # type: ignore safe due to wrapper

    @override
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.number](
        self: SplitBasis[
            Basis[BasisMetadata, Ctype[DT1]],
            Basis[BasisMetadata, Ctype[DT1]],
            Ctype[DT3],
        ],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion(lambda: _from_inner(self, vectors, axis))  # type: ignore safe due to wrapper

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("LINEAR_MAP")
            out.add("MUL")
            out.add("SUB")
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
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: complex
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


def is_split_basis(basis: object) -> TypeGuard[SplitBasis]:
    return isinstance(basis, SplitBasis)
