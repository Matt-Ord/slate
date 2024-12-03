from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate import basis
from slate.basis import Basis, TupleBasis
from slate.metadata import BasisMetadata, NestedLength, shallow_shape_from_nested

if TYPE_CHECKING:
    from slate.metadata import SimpleMetadata, StackedMetadata


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
    def fundamental_shape(self) -> NestedLength:
        """Datatype of the data stored in the array."""
        return self.basis.fundamental_shape

    @property
    def dtype(self) -> np.dtype[DT]:
        """Datatype of the data stored in the array."""
        return self._data.dtype

    @property
    def basis(self) -> B:
        """The basis of the Array."""
        return self._basis

    @property
    def raw_data(self) -> np.ndarray[Any, np.dtype[DT]]:
        """The raw data for the array."""
        return self._data

    @raw_data.setter
    def raw_data[DT1: np.generic](  # [DT1: DT]
        self, data: np.ndarray[Any, np.dtype[DT1]]
    ) -> None:
        """Set the raw data for the array."""
        assert self.basis.size == data.size
        self._data = cast(np.ndarray[Any, np.dtype[DT]], data).ravel()

    def as_array(self) -> np.ndarray[Any, np.dtype[DT]]:
        """Get the data as a (full) np.array."""
        metadata = self.basis.metadata()
        fundamental = basis.from_metadata(metadata, is_dual=self.basis.is_dual)
        shape = shallow_shape_from_nested(fundamental.fundamental_shape)
        converted = self.basis.__convert_vector_into__(self._data, fundamental)
        return converted.reshape(shape)

    @staticmethod
    def from_array[DT1: np.generic](
        array: np.ndarray[tuple[int, ...], np.dtype[DT1]],
    ) -> SlateArray[
        StackedMetadata[SimpleMetadata, None],
        DT1,
        TupleBasis[BasisMetadata, None, np.generic],
    ]:
        """Get a SlateArray from an array."""
        return SlateArray(basis.from_shape(array.shape), array)

    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> SlateArray[M, DT, B1]:
        """Get the SlateArray with the basis set to basis."""
        return SlateArray(
            basis, self.basis.__convert_vector_into__(self.raw_data, basis)
        )

    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]:
        as_add_basis = basis.as_add_basis(self.basis)
        data = as_add_basis.add_data(
            self.with_basis(as_add_basis).raw_data,
            other.with_basis(as_add_basis).raw_data,
        )

        return SlateArray(as_add_basis, data)

    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: SlateArray[M1, DT1],
    ) -> SlateArray[M1, DT1]:
        as_sub_basis = basis.as_sub_basis(self.basis)
        data = as_sub_basis.sub_data(
            self.with_basis(as_sub_basis).raw_data,
            other.with_basis(as_sub_basis).raw_data,
        )

        return SlateArray(as_sub_basis, data)

    def __mul__[M1: BasisMetadata, DT1: np.number[Any]](
        self: SlateArray[M1, DT1],
        other: float,
    ) -> SlateArray[M1, DT1]:
        as_mul_basis = basis.as_mul_basis(self.basis)
        data = as_mul_basis.mul_data(self.with_basis(as_mul_basis).raw_data, other)
        return SlateArray(as_mul_basis, data)
