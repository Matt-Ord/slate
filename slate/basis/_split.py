from __future__ import annotations

from copy import copy
from typing import Any, Never, Self, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature, ctype
from slate.basis._util import get_common_basis
from slate.basis._wrapped import WrappedBasis
from slate.util import Padding, pad_along_axis, slice_along_axis


class SplitBasis[
    B0: Basis,
    B1: Basis,
    DT: ctype[Never] = ctype[Never],
](
    WrappedBasis[Basis, DT],
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
    ](self: SplitBasis[B0_, B1_, ctype[Never]], lhs: B0_, rhs: B1_) -> None:
        self._lhs: B0 = lhs
        self._rhs: B1 = rhs
        super().__init__(get_common_basis(lhs, rhs))

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
        if isinstance(other, SplitBasis):
            return (
                other.lhs == self.lhs  # type: ignore unknown
                and other.rhs == self.rhs  # type: ignore unknown
            )
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

    def _get_lhs_vectors[DT1: np.complexfloating](
        self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{A}`."""
        start = 0
        end = self.lhs.size
        return vectors[slice_along_axis(slice(start, end), axis)]

    def _get_rhs_vectors[DT1: np.complexfloating](
        self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{B}`."""
        start = self.lhs.size
        return vectors[slice_along_axis(slice(start, None), axis)]

    @override
    def __into_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        lhs_fundamental = self.lhs.__convert_vector_into__(
            self._get_lhs_vectors(vectors, axis), self.inner, axis
        )

        rhs_fundamental = self.rhs.__convert_vector_into__(
            self._get_rhs_vectors(vectors, axis), self.inner, axis
        )

        return cast(
            "np.ndarray[Any, DT]",
            lhs_fundamental + rhs_fundamental,
        )

    @override
    def __from_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
        lhs_vector = self.inner.__convert_vector_into__(vectors, self.lhs, axis)
        return pad_along_axis(lhs_vector, Padding(self.size, 1, 0), axis)

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
