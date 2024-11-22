from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, Never, Self, cast, override

import numpy as np

from slate.array import SlateArray
from slate.basis import (
    Basis,
    BasisFeature,
    WrappedBasis,
    as_tuple_basis,
    tuple_basis,
)
from slate.basis._diagonal import DiagonalBasis, diagonal_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate.metadata import Metadata2D


type Direction = Literal["forward", "backward"]


class ExplicitBasis[M: BasisMetadata, DT: np.generic](
    WrappedBasis[M, DT, Basis[M, DT]],
):
    """Represents an explicit basis."""

    def __init__(
        self: Self,
        eigenvectors: SlateArray[Metadata2D[BasisMetadata, M, Any], DT],
        *,
        conjugate: bool = False,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        converted = eigenvectors.with_basis(as_tuple_basis(eigenvectors.basis))
        self._eigenvectors = converted
        self._direction: Direction = direction
        self._data_id = data_id or uuid.uuid4()
        super().__init__(converted.basis[1], conjugate=conjugate)

    @property
    def data_id(self: Self) -> uuid.UUID:
        return self._data_id

    def inverse_basis(self: Self) -> ExplicitBasis[M, DT]:
        return ExplicitBasis(
            self.eigenvectors,
            direction="backward" if self.direction == "forward" else "forward",
            conjugate=self.conjugate,
            data_id=self.data_id,
        )

    @property
    def eigenvectors(
        self: Self,
    ) -> SlateArray[Metadata2D[BasisMetadata, M, None], DT]:
        return self._eigenvectors

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExplicitBasis):
            return (
                self.size == other.size
                and other.inner == self.inner  # type: ignore unknown
                and other.direction == self.direction
                and other._data_id == self._data_id
                and self.conjugate == other.conjugate
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction, self._data_id, self.conjugate))

    @property
    @override
    def size(self: Self) -> int:
        return self.eigenvectors.basis.fundamental_shape[0]

    @property
    def direction(self: Self) -> Direction:
        """The convention used to select the direction for the forward transform."""
        return self._direction

    @property
    def _transform_matrix(self) -> np.ndarray[Any, Any]:
        # TODO: We should be able to use einsum to do this, but it is not implemented yet.  # noqa: FIX002
        # TODO: inv() on sparse matrices is not implemented yet.  # noqa: FIX002
        states_tuple = self.eigenvectors.with_basis(
            as_tuple_basis(self.eigenvectors.basis)
        )

        return (
            states_tuple.raw_data.reshape(states_tuple.basis.shape)
            if self.direction == "forward"
            else np.transpose(
                np.linalg.inv(states_tuple.raw_data.reshape(states_tuple.basis.shape))  # type: ignore unknown
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
        states_tuple = self.eigenvectors.with_basis(
            as_tuple_basis(self.eigenvectors.basis)
        )
        return (
            np.linalg.inv(states_tuple.raw_data.reshape(states_tuple.basis.shape))  # type: ignore inv
            if self.direction == "forward"
            else np.transpose(states_tuple.raw_data.reshape(states_tuple.basis.shape))
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
        _B: Basis[BasisMetadata, Any] = Basis[M1, Any],
    ](self: ExplicitBasis[M1, DT1], inner: _B) -> ExplicitBasis[M1, Any]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[
        M1: BasisMetadata,
        DT1: np.generic,
        _B: Basis[BasisMetadata, Any] = Basis[M1, Any],
    ](
        self: ExplicitBasis[M1, DT1], wrapper: Callable[[Basis[M1, DT]], _B]
    ) -> ExplicitBasis[M1, Any]:
        """Get the wrapped basis after wrapper is applied to inner."""
        converted = self.eigenvectors.with_basis(
            tuple_basis(
                (as_tuple_basis(self.eigenvectors.basis)[0], wrapper(self.inner))
            )
        )
        return ExplicitBasis(
            converted,
            direction=self.direction,
            data_id=self._data_id,
        )

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
        eigenvectors: SlateArray[Metadata2D[BasisMetadata, M, Any], DT],
        *,
        assert_unitary: bool = False,
        direction: Direction = "forward",
        conjugate: bool = False,
        data_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(
            eigenvectors, direction=direction, data_id=data_id, conjugate=conjugate
        )
        if assert_unitary:
            states_tuple = self.eigenvectors.with_basis(
                as_tuple_basis(self.eigenvectors.basis)
            )
            _assert_unitary(states_tuple.raw_data.reshape(states_tuple.basis.shape))

    @override
    def inverse_basis(self: Self) -> ExplicitUnitaryBasis[M, DT]:
        return ExplicitUnitaryBasis(
            self.eigenvectors,
            direction="backward" if self.direction == "forward" else "forward",
            conjugate=self.conjugate,
            data_id=self.data_id,
        )

    @property
    @override
    def _transform_matrix(self) -> np.ndarray[Any, Any]:
        states_tuple = self.eigenvectors.with_basis(
            as_tuple_basis(self.eigenvectors.basis)
        )
        return (
            states_tuple.raw_data.reshape(states_tuple.basis.shape)
            if self.direction == "forward"
            else np.conj(states_tuple.raw_data.reshape(states_tuple.basis.shape))
        )

    @property
    @override
    def _inverse_transform_matrix(self) -> np.ndarray[Any, Any]:
        states_tuple = self.eigenvectors.with_basis(
            as_tuple_basis(self.eigenvectors.basis)
        )
        return (
            np.transpose(
                np.conj(states_tuple.raw_data.reshape(states_tuple.basis.shape))
            )
            if self.direction == "forward"
            else np.transpose(states_tuple.raw_data.reshape(states_tuple.basis.shape))
        )


class TrivialExplicitBasis[M: BasisMetadata, DT: np.generic](
    ExplicitUnitaryBasis[M, DT]
):
    def __init__(
        self: Self,
        inner: Basis[M, DT],
        *,
        conjugate: bool = False,
    ) -> None:
        self._direction: Direction = "forward"
        self._data_id = uuid.UUID(int=0)
        WrappedBasis[M, DT, Basis[M, DT]].__init__(self, inner, conjugate=conjugate)

    @override
    def __into_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return vectors

    @override
    def __from_inner__[DT1: np.complex128](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        return vectors

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TrivialExplicitBasis):
            return self.inner == other.inner and self.conjugate == other.conjugate  # type: ignore unknown
        return False

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.conjugate))

    @property
    @override
    def size(self: Self) -> int:
        return self.inner.size

    @property
    @override
    def eigenvectors(
        self: Self,
    ) -> SlateArray[
        Metadata2D[M, M, None],
        Any,
        DiagonalBasis[Any, Basis[M, DT], Basis[M, DT], Any],
    ]:
        return SlateArray(
            diagonal_basis((self.inner, self.inner), self.inner.metadata()),
            np.ones(self.size),
        )
