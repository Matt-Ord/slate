from __future__ import annotations

from typing import Any, Callable, Self, cast, overload, override

import numpy as np

from slate.basis._basis import Basis, SupportsMulBasis
from slate.basis.stacked import DiagonalBasis, VariadicTupleBasis, tuple_basis
from slate.basis.transformed import TransformDirection, TransformedBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata.stacked.stacked import StackedMetadata
from slate.util import slice_along_axis
from slate.util._pad import Padding, pad_along_axis


class SplitBasis[
    B0: Basis[Any, Any],
    B1: Basis[Any, Any],
    E,
](
    WrappedBasis[
        StackedMetadata[Any, E], np.complex128, VariadicTupleBasis[Any, Any, Any, E]
    ],
    SupportsMulBasis,
):
    r"""Represents data in the split operator basis.

    Seprates an operator into a part acting on the fundamental basis and a part acting on the transformed basis.

    .. math::
        \hat{O} = \hat{A} + \hat{B} \hat{C} \hat{D}

    Where :math:`\hat{A}` is diagonal in the fundamental basis and :math:`\hat{B} \hat{C} \hat{D}`
    is evaluated by first applying :math:`\hat{D}` which acts on the rhs basis, then
    :math:`\hat{C}` which is diagonal in the transfromed (lhs, rhs) basis, and finally :math:`\hat{B}`
    which acts on the lhs basis.

    Internally, the basis data is stored by concatenating the individual parts of the operator.
    """

    def __init__(
        self: Self,
        inner: VariadicTupleBasis[Any, Any, Any, E],
        *,
        direction: TransformDirection = "forward",
    ) -> None:
        super().__init__(inner)
        self._direction: TransformDirection = direction
        assert self.inner.children[0].size == self.inner.children[1].size

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TransformedBasis):
            return (
                self.size == value.size
                and value.inner == self.inner  # type: ignore unknown
                and value.direction == self.direction
            )
        return False

    @property
    def direction(self: Self) -> TransformDirection:
        """The convention used to select the direction for the forward transform."""
        return self._direction

    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction))

    @override
    def conjugate_basis(self) -> SplitBasis[B0, B1, E]:
        return SplitBasis(
            self.inner.conjugate_basis(),
            direction="backward" if self.direction == "forward" else "forward",
        )

    @override
    def mul_data[DT1: np.number[Any]](
        self: Self,
        lhs: float,
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        out = rhs.copy()
        end = self.inner.children[0].size
        out[slice_along_axis(slice(0, end))] *= lhs
        out[slice_along_axis(slice(2 * end, 3 * end))] *= lhs
        return out

    @property
    @override
    def inner(self: Self) -> VariadicTupleBasis[np.complex128, Any, Any, E]:
        return cast(VariadicTupleBasis[np.complex128, Any, Any, E], self._inner)

    @property
    def transformed_inner(self: Self) -> VariadicTupleBasis[np.complex128, Any, Any, E]:
        """Inner of the transformed part of the basis."""
        return tuple_basis(
            (
                TransformedBasis(
                    self.inner.children[0],
                    direction="forward" if self.direction == "forward" else "backward",
                ),
                TransformedBasis(
                    self.inner.children[1],
                    direction="backward" if self.direction == "forward" else "forward",
                ),
            ),
            self.inner.metadata.extra,
        )

    @property
    @override
    def size(self: Self) -> int:
        return 4 * self.inner.children[0].size

    def get_a[DT1: np.complex128](
        self: Self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{A}`."""
        start = 0
        end = self.inner.children[0].size
        return vectors[slice_along_axis(slice(start, end), axis)]

    def get_b[DT1: np.complex128](
        self: Self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{B}`."""
        start = self.inner.children[0].size
        end = 2 * self.inner.children[0].size
        return vectors[slice_along_axis(slice(start, end), axis)]

    def get_c[DT1: np.complex128](
        self: Self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{C}`."""
        start = 2 * self.inner.children[0].size
        end = 3 * self.inner.children[0].size
        return vectors[slice_along_axis(slice(start, end), axis)]

    def get_d[DT1: np.complex128](
        self: Self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        r"""Get the vector corresponding to ..math::`\hat{D}`."""
        start = 3 * self.inner.children[0].size
        end = 4 * self.inner.children[0].size
        return vectors[slice_along_axis(slice(start, end), axis)]

    @override
    def __into_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        a_fundamental = DiagonalBasis(self.inner).__convert_vector_into__(
            self.get_a(vectors, axis), self.inner, axis
        )

        c_fundamental = DiagonalBasis(self.transformed_inner).__convert_vector_into__(
            self.get_c(vectors, axis), self.inner, axis
        )
        old_shape = c_fundamental.swapaxes(axis, 0).shape
        bcd_fundamental = (
            np.einsum(  # type: ignore unknown
                "i,ijk,j->ijk",
                self.get_b(vectors, axis),
                c_fundamental.swapaxes(axis, 0).reshape(*self.inner.shape, -1),
                self.get_d(vectors, axis),
            )
            .reshape(old_shape)
            .swapaxes(axis, 0)
        )
        return cast(
            np.ndarray[Any, np.dtype[DT1]],
            a_fundamental + bcd_fundamental,
        )

    @override
    def __from_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        a_vector = self.inner.__convert_vector_into__(
            vectors, DiagonalBasis[Any, Any, Any, E](self.inner), axis
        )
        return pad_along_axis(a_vector, Padding(self.size, 1, 0), axis)

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self: Self, inner: VariadicTupleBasis[Any, B01, B11, E1]
    ) -> SplitBasis[B01, B11, E1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        B01: Basis[Any, Any],
        B11: Basis[Any, Any],
        E1,
    ](
        self: Self,
        wrapper: Callable[
            [VariadicTupleBasis[Any, Any, Any, E]],
            VariadicTupleBasis[Any, B01, B11, E1],
        ],
    ) -> SplitBasis[B01, B11, E1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return SplitBasis(wrapper(self.inner), direction=self.direction)


@overload
def split_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any]](
    children: tuple[_B0, _B1],
    extra_metadata: None = None,
    *,
    direction: TransformDirection = "forward",
) -> SplitBasis[_B0, _B1, None]: ...


@overload
def split_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any], E](
    children: tuple[_B0, _B1],
    extra_metadata: E,
    *,
    direction: TransformDirection = "forward",
) -> SplitBasis[_B0, _B1, E]: ...


def split_basis[_B0: Basis[Any, Any], _B1: Basis[Any, Any], E](
    children: tuple[_B0, _B1],
    extra_metadata: E | None = None,
    *,
    direction: TransformDirection = "forward",
) -> SplitBasis[_B0, _B1, E | None]:
    """Build a VariadicTupleBasis from a tuple."""
    return SplitBasis[Any, Any, E | None](
        VariadicTupleBasis(children, extra_metadata), direction=direction
    )
