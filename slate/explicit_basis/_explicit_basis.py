from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, Never, Self, cast, override

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis._basis import Basis
from slate.basis.stacked._tuple_basis import tuple_basis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.basis.stacked._tuple_basis import VariadicTupleBasis


type Direction = Literal["forward", "backward"]


class ExplicitBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]]
):
    """Represents an explicit basis."""

    def __init__(
        self: Self,
        data: SlateArray[DT, VariadicTupleBasis[DT, Any, Any, Any]],
        *,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        self._data = data
        self._direction: Direction = direction
        self._data_id = data_id or uuid.uuid4()
        super().__init__(data.basis[1])

    def __eq__(self, value: object) -> bool:
        if isinstance(value, ExplicitBasis):
            return (
                self.size == value.size
                and value.inner == self.inner  # type: ignore unknown
                and value.direction == self.direction
                and value._data_id == self._data_id
            )
        return False

    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction, self._data_id))

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._data.basis[0].size

    @override
    def conjugate_basis(self) -> ExplicitBasis[M, DT]:
        return ExplicitBasis(
            self._data,
            direction="forward" if self.direction == "backward" else "backward",
            data_id=self._data_id,
        )

    @property
    def direction(self: Self) -> Direction:
        """The convention used to select the direction for the forward transform."""
        return self._direction

    @property
    def _transform_matrix(self) -> np.ndarray[Any, Any]:
        return (
            self._data.raw_data.reshape(self._data.basis.shape)
            if self.direction == "forward"
            else np.transpose(
                np.linalg.inv(self._data.raw_data.reshape(self._data.basis.shape))  # type: ignore unknown
            )  # type: ignore unknown
        )

    @override
    def __into_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = np.tensordot(
            cast(np.ndarray[Any, np.dtype[Never]], vectors),
            self._transform_matrix,
            axes=([axis], [0]),
        )
        return np.moveaxis(transformed, -1, axis)

    @property
    def _inverse_transform_matrix(self) -> np.ndarray[Any, Any]:
        return (
            np.linalg.inv(self._data.raw_data.reshape(self._data.basis.shape))  # type: ignore inv
            if self.direction == "forward"
            else np.transpose(self._data.raw_data.reshape(self._data.basis.shape))
        )

    @override
    def __from_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        transformed = np.tensordot(
            cast(np.ndarray[Any, np.dtype[Never]], vectors),
            self._inverse_transform_matrix,
            axes=([axis], [0]),
        )
        return np.moveaxis(transformed, -1, axis)

    @override
    def with_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](self: Self, inner: Basis[M1, DT1]) -> ExplicitBasis[M1, DT1]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
    ](
        self: Self, wrapper: Callable[[Basis[M, DT]], Basis[M1, DT1]]
    ) -> ExplicitBasis[M1, DT1]:
        """Get the wrapped basis after wrapper is applied to inner."""
        new_basis = tuple_basis(
            (self._data.basis[0], wrapper(self.inner)),
            extra_metadata=self._data.basis.metadata.extra,
        )
        return ExplicitBasis(
            cast(SlateArray[DT1, Any], convert_array(self._data, new_basis)),
            direction=self.direction,
            data_id=self._data_id,
        )


def _assert_unitary[DT: np.generic](vectors: np.ndarray[Any, np.dtype[DT]]) -> None:
    identity_matrix = np.eye(vectors.shape[0])
    result = np.dot(vectors, np.conj(np.transpose(vectors)))

    assert np.testing.assert_allclose(
        result,
        identity_matrix,
        err_msg="The provided axis vectors are not orthonormal.",
    )


class ExplicitUnitaryBasis[M: BasisMetadata, DT: np.generic](ExplicitBasis[M, DT]):
    """Represents a truncated basis."""

    def __init__(
        self: Self,
        data: SlateArray[DT, VariadicTupleBasis[DT, Any, Any, Any]],
        *,
        assert_unitary: bool = False,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        if assert_unitary:
            _assert_unitary(self._data.raw_data.reshape(self._data.basis.shape))
        super().__init__(data, direction=direction, data_id=data_id)

    @override
    def conjugate_basis(self) -> ExplicitUnitaryBasis[M, DT]:
        return ExplicitUnitaryBasis(
            self._data,
            direction="forward" if self.direction == "backward" else "backward",
            data_id=self._data_id,
        )

    @property
    def size(self: Self) -> int:
        """Number of elements in the basis."""
        return self._data.basis[0].size

    @property
    @override
    def _transform_matrix(self) -> np.ndarray[Any, Any]:
        return (
            self._data.raw_data.reshape(self._data.basis.shape)
            if self.direction == "forward"
            else np.conj(self._data.raw_data.reshape(self._data.basis.shape))
        )

    @property
    @override
    def _inverse_transform_matrix(self) -> np.ndarray[Any, Any]:
        return (
            np.transpose(np.conj(self._data.raw_data.reshape(self._data.basis.shape)))
            if self.direction == "forward"
            else np.transpose(self._data.raw_data.reshape(self._data.basis.shape))
        )
