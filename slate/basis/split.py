from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, Callable, Never, Self, cast, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature
from slate.basis._tuple import get_common_basis
from slate.basis.transformed import TransformedBasis
from slate.basis.wrapped import WrappedBasis
from slate.util import Padding, pad_along_axis, slice_along_axis

if TYPE_CHECKING:
    from slate.metadata import BasisMetadata


class SplitBasis[
    M: BasisMetadata,
    DT: np.generic,
    B0: Basis[Any, Any] = Basis[M, DT],
    B1: Basis[Any, Any] = Basis[M, DT],
](
    WrappedBasis[M, DT, Basis[M, DT]],
):
    r"""Represents data in the split basis.

    Seprates an operator into two parts.

    .. math::
        \hat{O} = \hat{A} + \hat{B}

    where :math:`\hat{A}` and :math:`\hat{B}` may be in a different basis.
    """

    def __init__[
        _B0: Basis[Any, Any],
        _B1: Basis[Any, Any],
    ](
        self: SplitBasis[Any, Any, _B0, _B1],
        lhs: _B0,
        rhs: _B1,
    ) -> None:
        self._lhs: B0 = lhs
        self._rhs: B1 = rhs
        super().__init__(get_common_basis(lhs, rhs))

    @property
    def lhs(self: Self) -> B0:
        """Left hand side of the split basis."""
        return self._lhs

    @property
    def rhs(self: Self) -> B1:
        """Right hand side of the split basis."""
        return self._rhs

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TransformedBasis):
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
    def size(self: Self) -> int:
        return self.lhs.size + self.rhs.size

    def _get_lhs_vectors[DT1: np.complex128](
        self: Self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{A}`."""
        start = 0
        end = self.lhs.size
        return vectors[slice_along_axis(slice(start, end), axis)]

    def _get_rhs_vectors[DT1: np.complex128](
        self: Self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{B}`."""
        start = self.lhs.size
        return vectors[slice_along_axis(slice(start, None), axis)]

    @override
    def __into_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[np.complex128]]:
        lhs_fundamental = self.lhs.__convert_vector_into__(
            self._get_lhs_vectors(vectors, axis), self.inner, axis
        )

        rhs_fundamental = self.rhs.__convert_vector_into__(
            self._get_rhs_vectors(vectors, axis), self.inner, axis
        )

        return cast(
            np.ndarray[Any, np.dtype[DT1]],
            lhs_fundamental + rhs_fundamental,
        )

    @override
    def __from_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        lhs_vector = self.inner.__convert_vector_into__(vectors, self.lhs, axis)
        return pad_along_axis(lhs_vector, Padding(self.size, 1, 0), axis)

    @override
    def with_inner(self: Self, inner: Any) -> SplitBasis[Never, Never, Never, Never]:
        return self.with_modified_inner(lambda _: 0)

    @override
    def with_modified_inner(
        self: Self,
        wrapper: Callable[
            [Never],
            Any,
        ],
    ) -> SplitBasis[Never, Never, Never, Never]:
        msg = "Cannot modify inner of a split basis."
        raise TypeError(msg)

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
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
