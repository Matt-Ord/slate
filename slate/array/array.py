from __future__ import annotations  # noqa: A005

from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from slate.basis import Basis, FundamentalBasis
from slate.basis.stacked import (
    TupleBasis,
    fundamental_tuple_basis_from_shape,
)
from slate.basis.wrapped import as_add_basis, as_mul_basis, as_sub_basis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata import SimpleMetadata
    from slate.metadata.stacked import StackedMetadata


class SlateArray[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Any] = Basis[M, Any],
]:  # B: Basis[Any, DT]
    """An array with data stored in a given basis."""

    def __init__[DT1: np.generic, B1: Basis[BasisMetadata, Any]](
        self: SlateArray[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        assert basis.size == data.size
        self._basis = cast(B, basis)
        self._data = data.ravel()

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Datatype of the data stored in the array."""
        return self.basis.fundamental_shape

    @property
    def dtype(self: Self) -> np.dtype[DT]:
        """Datatype of the data stored in the array."""
        return self._data.dtype

    @property
    def basis(self: Self) -> B:
        """The basis of the Array."""
        return self._basis

    @property
    def raw_data(self: Self) -> np.ndarray[Any, np.dtype[DT]]:
        """The raw data for the array."""
        return self._data

    @raw_data.setter
    def raw_data[DT1: np.generic](  # [DT1: DT]
        self: Self, data: np.ndarray[Any, np.dtype[DT1]]
    ) -> None:
        """Set the raw data for the array."""
        assert self.basis.size == data.size
        self._data = cast(np.ndarray[Any, np.dtype[DT]], data).ravel()

    def as_array(self: Self) -> np.ndarray[Any, np.dtype[DT]]:
        """Get the data as a (full) np.array."""
        return self.basis.__convert_vector_into__(
            self._data.ravel(), FundamentalBasis(self.basis.metadata())
        ).reshape(self.basis.fundamental_shape)

    @staticmethod
    def from_array[DT1: np.generic](
        array: np.ndarray[tuple[int, ...], np.dtype[DT1]],
    ) -> SlateArray[
        StackedMetadata[SimpleMetadata, None],
        DT1,
        TupleBasis[BasisMetadata, None, np.generic],
    ]:
        """Get a SlateArray from an array."""
        return SlateArray(
            fundamental_tuple_basis_from_shape(array.shape),
            array,
        )

    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self: Self, basis: B1
    ) -> SlateArray[M, DT, B1]:
        """Get the SlateArray with the basis set to basis."""
        return SlateArray(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]:
        basis = as_add_basis(self.basis)
        data = basis.add_data(
            self.with_basis(basis).raw_data,
            other.with_basis(basis).raw_data,
        )

        return SlateArray(basis, data)

    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]:
        basis = as_sub_basis(self.basis)
        data = basis.sub_data(
            self.with_basis(basis).raw_data,
            other.with_basis(basis).raw_data,
        )

        return SlateArray(basis, data)

    def __mul__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: float,
    ) -> SlateArray[M1, DT1]:
        basis = as_mul_basis(self.basis)
        data = basis.mul_data(self.with_basis(basis).raw_data, other)
        return SlateArray(basis, data)
