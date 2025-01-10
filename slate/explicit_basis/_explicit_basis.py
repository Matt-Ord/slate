from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, Literal, Self, cast, override

import numpy as np

from slate import array as _array
from slate import basis
from slate._einsum import einsum
from slate.array import Array
from slate.array._transpose import inv, transpose
from slate.basis import (
    Basis,
    BasisFeature,
    BasisStateMetadata,
    WrappedBasis,
    as_tuple_basis,
    tuple_basis,
)
from slate.basis._fundamental import FundamentalBasis
from slate.basis.recast import RecastBasis
from slate.metadata import BasisMetadata, SimpleMetadata, shallow_shape_from_nested

if TYPE_CHECKING:
    from slate.metadata import Metadata2D


type Direction = Literal["forward", "backward"]


class ExplicitBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](
    WrappedBasis[M, DT, B],
):
    """Represents an explicit basis."""

    def __init__[
        DT1: np.generic,
        B1: Basis[Any, Any],
    ](
        self: ExplicitBasis[Any, DT1, B1],
        matrix: Array[Metadata2D[SimpleMetadata, BasisStateMetadata[B1], None], DT1],
        *,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        self._matrix = cast(
            "Array[Metadata2D[SimpleMetadata, BasisStateMetadata[B], None], DT]", matrix
        )
        self._direction: Direction = direction
        self._data_id = data_id or uuid.uuid4()
        super().__init__(as_tuple_basis(self.eigenvectors.basis)[1])

    @property
    def data_id(self) -> uuid.UUID:
        return self._data_id

    @override
    def dual_basis(self) -> Self:
        dual = super().dual_basis()
        dual._direction = "backward" if self.direction == "forward" else "forward"  # noqa: SLF001
        return dual

    @property
    def transform(
        self,
    ) -> Array[Metadata2D[SimpleMetadata, BasisStateMetadata[B], None], DT]:
        return (
            self._matrix
            if self.direction == "forward"
            else transpose(inv(self._matrix))
        )

    @property
    def inverse_transform(
        self,
    ) -> Array[Metadata2D[BasisStateMetadata[B], SimpleMetadata, None], DT]:
        return (
            inv(self._matrix)
            if self.direction == "forward"
            else transpose(self._matrix)
        )

    @property
    def eigenvectors(self) -> Array[Metadata2D[BasisMetadata, M, None], DT]:
        transposed = transpose(self.inverse_transform)
        inner_recast = basis.from_metadata(transposed.basis.metadata())

        state_basis = inner_recast[1].metadata().basis
        if inner_recast[1].is_dual:
            state_basis = state_basis.dual_basis()

        inner = tuple_basis((inner_recast[0], state_basis))
        eigenvectors_basis = RecastBasis(inner, inner_recast, transposed.basis)
        return _array.cast_basis(transposed, eigenvectors_basis)

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ExplicitBasis):
            return (
                self.size == other.size
                and other.inner == self.inner  # type: ignore unknown
                and other.direction == self.direction
                and other._data_id == self._data_id
            )
        return False

    @override
    def __hash__(self) -> int:
        return hash((1, self.inner, self.direction, self._data_id))

    @property
    @override
    def size(self) -> int:
        return shallow_shape_from_nested(
            self.eigenvectors.basis.metadata().fundamental_shape
        )[0]

    @property
    def direction(self) -> Direction:
        """The convention used to select the direction for the forward transform."""
        return self._direction

    @override
    def __into_inner__[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        self, vectors: np.ndarray[Any, np.dtype[DT1]], axis: int = -1
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = cast("np.ndarray[Any, np.dtype[Any]]", vectors).swapaxes(axis, -1)
        flat = swapped.reshape(-1, vectors.shape[axis])

        transform = self.transform
        flat_basis = tuple_basis(
            (
                FundamentalBasis.from_size(flat.shape[0]),
                FundamentalBasis(
                    transform.basis.metadata()[0],
                    is_dual=not cast("tuple[bool,...]", transform.basis.is_dual)[0],
                ),
            )
        )
        swapped_array = Array(flat_basis, flat)

        transformed = einsum("(i j'),(j k)->(i k)", swapped_array, transform)
        return (
            transformed.as_array().reshape(*swapped.shape[:-1], -1).swapaxes(axis, -1)
        )

    @override
    def __from_inner__[DT1: np.complexfloating](  # type: ignore we should have stricter bound on parent
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        swapped = cast("np.ndarray[Any, np.dtype[Any]]", vectors).swapaxes(axis, -1)
        flat = swapped.reshape(-1, vectors.shape[axis])

        transform = self.inverse_transform
        flat_basis = tuple_basis(
            (
                FundamentalBasis.from_size(flat.shape[0]),
                FundamentalBasis(
                    transform.basis.metadata()[0],
                    is_dual=not cast("tuple[bool,...]", transform.basis.is_dual)[0],
                ),
            )
        )
        swapped_array = Array(flat_basis, flat)

        transform = self.inverse_transform
        transformed = einsum("(i j'),(j k)->(i k)", swapped_array, transform)
        return (
            transformed.as_array().reshape(*swapped.shape[:-1], -1).swapaxes(axis, -1)
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
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
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
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M1, M2, E], DT]:
    conj = _array.conjugate(array)
    return Array(conj.basis.dual_basis(), conj.raw_data)


class ExplicitUnitaryBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](ExplicitBasis[M, DT, B]):
    """Represents a truncated basis."""

    def __init__[
        DT1: np.generic,
        B1: Basis[Any, Any],
    ](
        self: ExplicitUnitaryBasis[Any, DT1, B1],
        matrix: Array[Metadata2D[SimpleMetadata, BasisStateMetadata[B1], Any], DT1],
        *,
        assert_unitary: bool = False,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(cast("Any", matrix), direction=direction, data_id=data_id)
        if assert_unitary:
            states_tuple = self.eigenvectors.with_basis(
                as_tuple_basis(self.eigenvectors.basis)
            )
            _assert_unitary(states_tuple.raw_data.reshape(states_tuple.basis.shape))

    @property
    @override
    def transform(
        self,
    ) -> Array[Metadata2D[SimpleMetadata, BasisStateMetadata[B], None], DT]:
        return (
            self._matrix
            if self.direction == "forward"
            else _dual_unitary_data(self._matrix)
        )

    @property
    @override
    def inverse_transform(
        self,
    ) -> Array[Metadata2D[BasisStateMetadata[B], SimpleMetadata, None], DT]:
        return (
            _dual_unitary_data(transpose(self.transform))
            if self.direction == "forward"
            else transpose(self._matrix)
        )
