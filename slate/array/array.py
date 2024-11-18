from __future__ import annotations  # noqa: A005

from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np

from slate.basis import Basis, FundamentalBasis
from slate.basis.stacked import (
    TupleBasis,
    fundamental_tuple_basis_from_shape,
)
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata import SimpleMetadata


class SlateArray[DT: np.generic, B: Basis[Any, Any]]:  # B: Basis[Any, DT]
    """An array with data stored in a given basis."""

    def __init__(self, basis: B, data: np.ndarray[Any, np.dtype[DT]]) -> None:
        assert basis.size == data.size
        self._basis = basis
        self._data = data.ravel()

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Datatype of the data stored in the array."""
        return self._basis.fundamental_shape

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
            self._data.ravel(), FundamentalBasis(self.basis.metadata)
        ).reshape(self.basis.fundamental_shape)

    @staticmethod
    def from_array[DT1: np.generic](
        array: np.ndarray[tuple[int, ...], np.dtype[DT1]],
    ) -> SlateArray[DT1, TupleBasis[SimpleMetadata, None, np.generic]]:
        """Get a SlateArray from an array."""
        return SlateArray(
            fundamental_tuple_basis_from_shape(array.shape),
            array,
        )

    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self: Self, basis: B1
    ) -> SlateArray[DT, B1]:
        """Get the SlateArray with the basis set to basis."""
        return SlateArray(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __add__[_DT: np.number[Any], M: BasisMetadata](
        self: SlateArray[_DT, Basis[M, Any]],
        other: SlateArray[_DT, Basis[M, Any]],
    ) -> SlateArray[_DT, Basis[M, Any]]:
        data = self.basis.add_data(self.raw_data, other.with_basis(self.basis).raw_data)

        return SlateArray[_DT, Basis[M, Any]](self.basis, data)

    def __sub__[_DT: np.number[Any], M: BasisMetadata](
        self: SlateArray[_DT, Basis[M, Any]],
        other: SlateArray[_DT, Basis[M, Any]],
    ) -> SlateArray[_DT, Basis[M, Any]]:
        data = self.basis.sub_data(self.raw_data, other.with_basis(self.basis).raw_data)

        return SlateArray[_DT, Basis[M, Any]](self.basis, data)

    def __mul__[_DT: np.number[Any], M: BasisMetadata](
        self: SlateArray[_DT, Basis[M, Any]],
        other: float,
    ) -> SlateArray[_DT, Basis[M, Any]]:
        data = self.basis.mul_data(self.raw_data, other)

        return SlateArray[_DT, Basis[M, Any]](self.basis, data)
