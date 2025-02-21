from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast, overload

import numpy as np

from slate import basis
from slate.basis import Basis, FundamentalBasis, TupleBasis
from slate.basis._tuple import TupleBasis
from slate.metadata import (
    BasisMetadata,
    NestedLength,
    shallow_shape_from_nested,
)
from slate.util._index import slice_along_axis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from slate.basis._basis import ctype
    from slate.basis._tuple import TupleBasisMetadata
    from slate.metadata import SimpleMetadata
    from slate.metadata.stacked import TupleMetadata

type Index = int | slice
type NestedIndex = Index | tuple[NestedIndex, ...]


def _index_single_raw_along_axis[DT: np.generic](
    index: Index,
    data_basis: Basis[BasisMetadata, ctype[DT]],
    data: np.ndarray[Any, np.dtype[DT]],
    *,
    axis: int = -1,
) -> tuple[Basis[BasisMetadata, Any] | None, np.ndarray[Any, np.dtype[DT]]]:
    if index == slice(None):
        return data_basis, data
    fundamental_basis = basis.as_fundamental(data_basis)
    data = data_basis.__convert_vector_into__(data, fundamental_basis, axis=axis).ok()
    out = data[slice_along_axis(index, axis=axis)]
    out_basis = (
        FundamentalBasis.from_size(out.shape[axis])
        if isinstance(index, slice)
        else None
    )
    return out_basis, out


def _index_tuple_raw_along_axis[DT: np.generic](
    index: tuple[NestedIndex, ...],
    data_basis: Basis[TupleBasisMetadata, ctype[DT]],
    data: np.ndarray[Any, np.dtype[DT]],
    *,
    axis: int = -1,
) -> tuple[Basis[BasisMetadata, Any] | None, np.ndarray[Any, np.dtype[DT]]]:
    axis &= data.ndim
    tuple_basis = basis.as_tuple_basis(basis.as_linear_map_basis(data_basis))
    children = tuple_basis.children
    stacked_shape = (
        data.shape[:axis] + tuple(c.size for c in children) + data.shape[axis + 1 :]
    )
    tuple_basis.cast_ctype(np.generic)
    data = data_basis.__convert_vector_into__(
        data, tuple_basis.cast_ctype(np.generic), axis=axis
    ).ok()
    data = data.reshape(stacked_shape)

    final_basis = list[Basis[Any, Any]]()
    for child_index, child in zip(index, children, strict=False):
        child_axis = axis + len(final_basis)
        meta, data = _index_raw_along_axis(child_index, child, data, axis=child_axis)
        if meta is not None:
            final_basis.append(meta)
    if len(final_basis) == 0:
        return None, data.reshape(data.shape[:axis] + data.shape[axis + 1 :])

    data = data.reshape(data.shape[:axis] + (-1,) + data.shape[axis + 1 :])
    if len(final_basis) == 1:
        return final_basis[0], data
    return TupleBasis(tuple(final_basis), None), data


def _index_raw_along_axis[DT: np.dtype[np.generic]](
    index: NestedIndex,
    basis: Basis[BasisMetadata, Any],
    data: np.ndarray[Any, DT],
    axis: int,
) -> tuple[Basis[BasisMetadata, Any] | None, np.ndarray[Any, DT]]:
    if isinstance(index, tuple):
        return _index_tuple_raw_along_axis(index, basis, data, axis=axis)
    return _index_single_raw_along_axis(index, basis, data, axis=axis)


class ArrayConversion[
    M0: BasisMetadata,
    DT0: ctype[np.generic],
    B1: Basis,
    DT: np.dtype[np.generic],
]:
    def __init__(
        self,
        data: np.ndarray[tuple[int], DT],
        old_basis: Basis[Any, Any],
        new_basis: B1,
    ) -> None:
        self._data = data
        self._old_basis = old_basis
        self._new_basis = new_basis

    def _metadata_variance_fn(self, value: M0, _private: Never) -> None: ...

    def ok[M_: BasisMetadata, DT_: np.generic](
        self: ArrayConversion[M_, ctype[DT_], Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[B1, DT]:
        return Array[Any, Any](
            self._new_basis,
            self._old_basis.__convert_vector_into__(self._data, self._new_basis).ok(),
        )


class Array[B: Basis, DT: np.dtype[np.generic]]:
    """An array with data stored in a given basis."""

    def __init__(self, basis: B, data: np.ndarray[Any, DT]) -> None:
        assert basis.size == data.size
        self._basis = basis
        self._data = data.ravel()

    @property
    def fundamental_shape(self) -> NestedLength:
        """Datatype of the data stored in the array."""
        return self.basis.fundamental_shape

    @property
    def dtype(self) -> DT:
        """Datatype of the data stored in the array."""
        return self._data.dtype

    @property
    def basis(self) -> B:
        """The basis of the Array."""
        return self._basis

    @property
    def raw_data(self) -> np.ndarray[Any, DT]:
        """The raw data for the array."""
        return self._data

    @raw_data.setter
    def raw_data(self, data: np.ndarray[Any, np.dtype[np.generic]]) -> None:
        """Set the raw data for the array."""
        assert self.basis.size == data.size
        self._data = cast("np.ndarray[tuple[int], Any]", data.ravel())

    def as_array[DT_: np.generic](
        self: Array[Basis[BasisMetadata, ctype[DT_]], np.dtype[DT_]],
    ) -> np.ndarray[Any, np.dtype[DT_]]:
        """Get the data as a (full) np.array."""
        fundamental = basis.as_fundamental(self.basis)
        shape = shallow_shape_from_nested(fundamental.fundamental_shape)
        return self.with_basis(fundamental).ok().raw_data.reshape(shape)

    @overload
    @staticmethod
    def from_array(
        array: np.ndarray[tuple[int,], Any],
    ) -> Array[
        TupleBasis1D[np.dtype[np.generic], FundamentalBasis[SimpleMetadata], None],
        DT,
    ]: ...

    @overload
    @staticmethod
    def from_array(
        array: np.ndarray[tuple[int, int], Any],
    ) -> Array[
        Metadata2D[SimpleMetadata, SimpleMetadata, None],
        DT,
        TupleBasis2D[
            np.generic,
            FundamentalBasis[SimpleMetadata],
            FundamentalBasis[SimpleMetadata],
            None,
        ],
    ]: ...

    @overload
    @staticmethod
    def from_array(
        array: np.ndarray[tuple[int, int, int], Any],
    ) -> Array[
        Metadata3D[SimpleMetadata, SimpleMetadata, SimpleMetadata, None],
        DT,
        TupleBasis3D[
            np.generic,
            FundamentalBasis[SimpleMetadata],
            FundamentalBasis[SimpleMetadata],
            FundamentalBasis[SimpleMetadata],
            None,
        ],
    ]: ...

    @overload
    @staticmethod
    def from_array(
        array: np.ndarray[tuple[int, ...], Any],
    ) -> Array[
        TupleMetadata[SimpleMetadata, None],
        DT,
        TupleBasis[BasisMetadata, None, np.generic],
    ]: ...

    @staticmethod
    def from_array(
        array: np.ndarray[tuple[int, ...], Any],
    ) -> Array[
        TupleMetadata[SimpleMetadata, None],
        DT,
        TupleBasis[BasisMetadata, None, np.generic],
    ]:
        """Get a Array from an array."""
        return Array(basis.from_shape(array.shape), array)

    def with_basis[
        DT_: np.generic,
        M0_: BasisMetadata,
        DT0_: ctype[Never],
        B1_: Basis,
    ](
        self: Array[Basis[M0_, DT0_], np.dtype[DT_]],
        basis: B1_,
    ) -> ArrayConversion[M0_, DT0_, B1_, np.dtype[DT_]]:
        """Get the Array with the basis set to basis."""
        return ArrayConversion(self.raw_data, self.basis, basis)

    def __add__[M_: BasisMetadata, DT_: np.number[Any]](
        self: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, ctype[DT_]], np.dtype[DT_]]:
        as_add_basis = basis.as_add_basis(self.basis)
        data = as_add_basis.add_data(
            self.with_basis(as_add_basis).raw_data,
            other.with_basis(as_add_basis).raw_data,
        )

        return Array(as_add_basis, data)

    def __sub__[M_: BasisMetadata, DT_: np.number[Any]](
        self: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, ctype[DT_]], np.dtype[DT_]]:
        as_sub_basis = basis.as_sub_basis(self.basis)
        data = as_sub_basis.sub_data(
            self.with_basis(as_sub_basis).raw_data,
            other.with_basis(as_sub_basis).raw_data,
        )

        return Array(as_sub_basis, data)

    def __mul__[M_: BasisMetadata, DT_: np.number[Any]](
        self: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
        other: float,
    ) -> Array[Basis[M_, ctype[DT_]], np.dtype[DT_]]:
        as_mul_basis = basis.as_mul_basis(self.basis)
        data = as_mul_basis.mul_data(self.with_basis(as_mul_basis).raw_data, other)
        return Array(as_mul_basis, data)

    @overload
    def __iter__[DT_: np.dtype[np.generic]](
        self: Array[Metadata1D[Any, Any], DT_, Basis[Any, Any]], /
    ) -> Never: ...
    @overload
    def __iter__[M1: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[Metadata2D[Any, M1, Any], DT_, Basis[Any, Any]], /
    ) -> Iterator[Array[Basis[M1, Any], DT_]]: ...
    @overload
    def __iter__[M1: BasisMetadata, M2: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[Metadata3D[Any, M1, M2, Any], DT_, Basis[Any, Any]], /
    ) -> Iterator[
        Array[TupleBasis2D[Any, Basis[M1, Any], Basis[M2, Any], Any], DT_]
    ]: ...
    @overload
    def __iter__[M1: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[TupleMetadata[M1, Any], DT_, Basis[M1, Any]], /
    ) -> Iterator[Array[TupleMetadata[M1, None], DT_]]: ...

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

    @overload
    def __getitem__[DT_: np.dtype[np.generic]](
        self: Array[Any, DT_],
        index: int,
    ) -> DT_: ...

    @overload
    def __getitem__[DT_: np.dtype[np.generic]](
        self: Array[Any, DT_],
        index: tuple[NestedIndex, ...] | slice,
    ) -> Array[Any, DT_]: ...

    def __getitem__[DT_: np.dtype[np.generic]](
        self: Array[Any, DT_],
        index: NestedIndex,
    ) -> Array[Any, DT_] | DT_:
        indexed_basis, indexed_data = _index_raw_along_axis(
            index, self.basis, self.raw_data.reshape(-1, 1), axis=0
        )
        if indexed_basis is None:
            return indexed_data.item()
        return Array(indexed_basis, indexed_data)
