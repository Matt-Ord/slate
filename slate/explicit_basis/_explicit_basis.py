from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, Never, Self, cast, override

import numpy as np

from slate.array import SlateArray
from slate.array._transpose import inv, transpose
from slate.basis import (
    Basis,
    BasisFeature,
    WrappedBasis,
    as_tuple_basis,
    tuple_basis,
)
from slate.basis._diagonal import diagonal_basis
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
        matrix: SlateArray[Metadata2D[BasisMetadata, M, Any], DT],
        *,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        self._matrix = matrix
        self._direction: Direction = direction
        self._data_id = data_id or uuid.uuid4()
        super().__init__(as_tuple_basis(self.eigenvectors.basis)[1])

    @property
    def data_id(self: Self) -> uuid.UUID:
        return self._data_id

    @override
    def dual_basis(self: Self) -> Self:
        dual = super().dual_basis()
        dual._direction = "backward" if self.direction == "forward" else "forward"  # noqa: SLF001
        return dual

    @property
    def transform(
        self: Self,
    ) -> SlateArray[Metadata2D[BasisMetadata, M, None], DT]:
        return (
            self._matrix
            if self.direction == "forward"
            else transpose(inv(self._matrix))
        )

    @property
    def inverse_transform(
        self: Self,
    ) -> SlateArray[Metadata2D[BasisMetadata, M, None], DT]:
        return (
            inv(self._matrix)
            if self.direction == "forward"
            else transpose(self._matrix)
        )

    @property
    def eigenvectors(
        self: Self,
    ) -> SlateArray[Metadata2D[BasisMetadata, M, None], DT]:
        data = self.transform
        return data.with_basis(data.basis.dual_basis())

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExplicitBasis):
            return (
                self.size == other.size
                and other.inner == self.inner  # type: ignore unknown
                and other.direction == self.direction
                and other._data_id == self._data_id
                and self.is_dual == other.is_dual
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction, self._data_id, self.is_dual))

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
        states_tuple = self.transform.with_basis(as_tuple_basis(self.transform.basis))
        return states_tuple.raw_data.reshape(states_tuple.basis.shape)

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
        states_tuple = self.inverse_transform.with_basis(
            as_tuple_basis(self.inverse_transform.basis)
        )
        return states_tuple.raw_data.reshape(states_tuple.basis.shape)

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


def _dual_unitary_data[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.generic](
    array: SlateArray[Metadata2D[M1, M2, E], DT],
) -> SlateArray[Metadata2D[M1, M2, E], DT]:
    # This assumes that transposed basis is 'index-like'
    # compared to the original (list, inner) basis.
    return SlateArray(array.basis.dual_basis(), np.conjugate(array.raw_data))


class ExplicitUnitaryBasis[M: BasisMetadata, DT: np.generic](ExplicitBasis[M, DT]):
    """Represents a truncated basis."""

    def __init__(
        self: Self,
        matrix: SlateArray[Metadata2D[BasisMetadata, M, Any], DT],
        *,
        assert_unitary: bool = False,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(matrix, direction=direction, data_id=data_id)
        if assert_unitary:
            states_tuple = self.eigenvectors.with_basis(
                as_tuple_basis(self.eigenvectors.basis)
            )
            _assert_unitary(states_tuple.raw_data.reshape(states_tuple.basis.shape))

    @property
    @override
    def transform(
        self: Self,
    ) -> SlateArray[Metadata2D[BasisMetadata, M, None], DT]:
        return (
            self._matrix
            if self.direction == "forward"
            else _dual_unitary_data(self._matrix)
        )

    @property
    @override
    def inverse_transform(
        self: Self,
    ) -> SlateArray[Metadata2D[BasisMetadata, M, None], DT]:
        return (
            _dual_unitary_data(transpose(self.transform))
            if self.direction == "forward"
            else transpose(self._matrix)
        )


class TrivialExplicitBasis[M: BasisMetadata, DT: np.generic](
    ExplicitUnitaryBasis[M, DT]
):
    def __init__(self: Self, inner: Basis[M, DT]) -> None:
        super().__init__(
            SlateArray(
                diagonal_basis((inner, inner.dual_basis()), self.inner.metadata()),
                cast(np.ndarray[Any, np.dtype[DT]], np.ones(self.size)),
            ),
            data_id=uuid.UUID(int=0),
        )

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
