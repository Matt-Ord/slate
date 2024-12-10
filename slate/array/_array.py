from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast, overload

import numpy as np

from slate import basis
from slate.basis import Basis, TupleBasis
from slate.metadata import (
    BasisMetadata,
    Metadata2D,
    NestedLength,
    shallow_shape_from_nested,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from slate.basis._tuple import TupleBasis2D
    from slate.metadata import SimpleMetadata, StackedMetadata
    from slate.metadata.stacked import Metadata1D, Metadata3D


class Array[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[BasisMetadata, Any] = Basis[M, Any],
]:  # B: Basis[Any, DT]
    """An array with data stored in a given basis."""

    def __init__[DT1: np.generic, B1: Basis[BasisMetadata, Any]](
        self: Array[Any, DT1, B1],
        basis: B1,
        data: np.ndarray[Any, np.dtype[DT]],
    ) -> None:
        assert basis.size == data.size
        self._basis = cast("B", basis)
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
        self._data = cast("np.ndarray[Any, np.dtype[DT]]", data).ravel()

    def as_array(self) -> np.ndarray[Any, np.dtype[DT]]:
        """Get the data as a (full) np.array."""
        fundamental = basis.as_fundamental(self.basis)
        shape = shallow_shape_from_nested(fundamental.fundamental_shape)
        return self.with_basis(fundamental).raw_data.reshape(shape)

    @staticmethod
    def from_array[DT1: np.generic](
        array: np.ndarray[tuple[int, ...], np.dtype[DT1]],
    ) -> Array[
        StackedMetadata[SimpleMetadata, None],
        DT1,
        TupleBasis[BasisMetadata, None, np.generic],
    ]:
        """Get a Array from an array."""
        return Array(basis.from_shape(array.shape), array)

    def with_basis[B1: Basis[Any, Any]](  # B1: B
        self, basis: B1
    ) -> Array[M, DT, B1]:
        """Get the Array with the basis set to basis."""
        return Array(basis, self.basis.__convert_vector_into__(self.raw_data, basis))

    def __add__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: Array[M1, DT1],
    ) -> Array[M1, DT1]:
        as_add_basis = basis.as_add_basis(self.basis)
        data = as_add_basis.add_data(
            self.with_basis(as_add_basis).raw_data,
            other.with_basis(as_add_basis).raw_data,
        )

        return Array(as_add_basis, data)

    def __sub__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: Array[M1, DT1],
    ) -> Array[M1, DT1]:
        as_sub_basis = basis.as_sub_basis(self.basis)
        data = as_sub_basis.sub_data(
            self.with_basis(as_sub_basis).raw_data,
            other.with_basis(as_sub_basis).raw_data,
        )

        return Array(as_sub_basis, data)

    def __mul__[M1: BasisMetadata, DT1: np.number[Any]](
        self: Array[M1, DT1],
        other: float,
    ) -> Array[M1, DT1]:
        as_mul_basis = basis.as_mul_basis(self.basis)
        data = as_mul_basis.mul_data(self.with_basis(as_mul_basis).raw_data, other)
        return Array(as_mul_basis, data)

    @overload
    def __iter__[_DT: np.generic](
        self: Array[Metadata1D[Any, Any], _DT, Basis[Any, Any]], /
    ) -> Never: ...
    @overload
    def __iter__[_M1: BasisMetadata, _DT: np.generic](
        self: Array[Metadata2D[Any, _M1, Any], _DT, Basis[Any, Any]], /
    ) -> Iterator[Array[Basis[_M1, Any], _DT]]: ...
    @overload
    def __iter__[_M1: BasisMetadata, _M2: BasisMetadata, _DT: np.generic](
        self: Array[Metadata3D[Any, _M1, _M2, Any], _DT, Basis[Any, Any]], /
    ) -> Iterator[
        Array[TupleBasis2D[Any, Basis[_M1, Any], Basis[_M2, Any], Any], _DT]
    ]: ...
    @overload
    def __iter__[_M1: BasisMetadata, _DT: np.generic](
        self: Array[StackedMetadata[_M1, Any], _DT, Basis[_M1, Any]], /
    ) -> Iterator[Array[StackedMetadata[_M1, None], _DT]]: ...

    def __iter__(
        self: Array[Any, Any, Basis[BasisMetadata, Any]], /
    ) -> Iterator[Array[Any, Any]]:
        basis_as_tuple = basis.with_modified_child(
            basis.as_tuple_basis(self.basis), basis.as_fundamental, 0
        )
        as_tuple = self.with_basis(basis_as_tuple)
        children = as_tuple.basis.children[1:]
        match len(children):
            case 0:
                msg = "Cannot iterate over a flat array."
                raise ValueError(msg)
            case 1:
                out_basis = children[0]
            case _:
                out_basis = basis.tuple_basis(children)

        return (
            Array(out_basis, row)
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )
