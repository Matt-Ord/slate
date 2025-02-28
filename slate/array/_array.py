from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast, overload

import numpy as np

from slate import basis
from slate.basis import Basis, FundamentalBasis, TupleBasis
from slate.basis._tuple import is_tuple_basis_like
from slate.metadata import (
    BasisMetadata,
    NestedLength,
    shallow_shape_from_nested,
)
from slate.util._index import slice_along_axis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from slate.basis._basis import ctype
    from slate.basis._tuple import TupleBasisLike
    from slate.metadata import SimpleMetadata, TupleMetadata

type Index = int | slice
type NestedIndex = Index | tuple[NestedIndex, ...]


def _index_single_raw_along_axis(
    index: Index,
    data_basis: Basis[BasisMetadata, Any],
    data: np.ndarray[Any, Any],
    *,
    axis: int = -1,
) -> tuple[Basis[BasisMetadata, Any] | None, np.ndarray[Any, Any]]:
    if index == slice(None):
        return data_basis, data
    basis.as_fundamental(data_basis)
    data = data_basis.__into_fundamental__(data, axis=axis).ok()
    out = data[slice_along_axis(index, axis=axis)]
    out_basis = (
        FundamentalBasis.from_size(out.shape[axis])
        if isinstance(index, slice)
        else None
    )
    return out_basis, out


def _index_tuple_raw_along_axis(
    index: tuple[NestedIndex, ...],
    data_basis: TupleBasisLike,
    data: np.ndarray[Any, Any],
    *,
    axis: int = -1,
) -> tuple[Basis[BasisMetadata, Any] | None, np.ndarray[Any, Any]]:
    axis &= data.ndim
    tuple_basis = basis.as_tuple_basis(basis.as_linear_map_basis(data_basis))
    if tuple_basis is None:
        msg = "Cannot index a non-tuple basis."
        raise ValueError(msg)
    children = tuple_basis.children
    stacked_shape = (
        data.shape[:axis] + tuple(c.size for c in children) + data.shape[axis + 1 :]
    )

    data = data_basis.__convert_vector_into__(data, tuple_basis, axis=axis).ok()
    data = data.reshape(stacked_shape)

    final_basis = list[Basis]()
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
    basis: Basis,
    data: np.ndarray[Any, DT],
    axis: int,
) -> tuple[Basis[BasisMetadata, Any] | None, np.ndarray[Any, DT]]:
    if isinstance(index, tuple):
        assert is_tuple_basis_like(basis)
        return _index_tuple_raw_along_axis(index, basis, data, axis=axis)
    return _index_single_raw_along_axis(index, basis, data, axis=axis)


class ArrayConversion[
    M0: BasisMetadata,
    B1: Basis,
    DT: np.dtype[np.generic],
]:
    def __init__(
        self,
        data: np.ndarray[tuple[int], DT],
        old_basis: Basis,
        new_basis: B1,
    ) -> None:
        self._data = data
        # This is not the true type - but this is safe as it must at least be able to support
        # the data into the new basis.
        self._old_basis = cast("Basis[BasisMetadata, ctype[np.generic]]", old_basis)
        self._new_basis = new_basis

    def _metadata_variance_fn(self, value: M0, _private: Never) -> None: ...

    def ok[M_: BasisMetadata, DT_: np.generic](
        self: ArrayConversion[M_, Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[B1, DT]:
        return cast(
            "Array[B1, DT]",
            ArrayBuilder(
                self._new_basis,
                self._old_basis.__convert_vector_into__(
                    self._data, self._new_basis
                ).ok(),
            ).ok(),
        )


class ArrayBuilder[B: Basis, DT: np.dtype[np.generic]]:
    def __init__(self, basis: B, data: np.ndarray[Any, DT]) -> None:
        self._basis = basis
        self._data = data

    def ok[DT_: np.generic](
        self: ArrayBuilder[Basis[Any, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[B, DT]:
        return cast("Any", Array.__new__(self._basis, self._data, 0))  # type: ignore safe to construct


class Array[B: Basis, DT: np.dtype[np.generic]]:
    """
    An array with data stored in a given basis.

    The basis MUST be able to support the dtype of the data.
    This makes constructing an array directly unsafe - it must be done via the ArrayBuilder interface.
    """

    def __init__(self, basis: B, data: np.ndarray[Any, DT], _private: Never) -> None:
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

    def as_array(self) -> np.ndarray[Any, DT]:
        """Get the data as a (full) np.array."""
        fundamental = basis.as_fundamental(self.basis)
        shape = shallow_shape_from_nested(fundamental.fundamental_shape)
        return self.with_basis(fundamental).ok().raw_data.reshape(shape)  # type: ignore bad inference

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int,], DT_],
    ) -> Array[
        TupleBasis[
            tuple[FundamentalBasis[SimpleMetadata]], None, ctype[np.generic[Any]]
        ],
        DT_,
    ]: ...

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, int], DT_],
    ) -> Array[
        TupleBasis[
            tuple[
                FundamentalBasis[SimpleMetadata],
                FundamentalBasis[SimpleMetadata],
            ],
            None,
            ctype[np.generic[Any]],
        ],
        DT_,
    ]: ...

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, int, int], DT_],
    ) -> Array[
        TupleBasis[
            tuple[
                FundamentalBasis[SimpleMetadata],
                FundamentalBasis[SimpleMetadata],
                FundamentalBasis[SimpleMetadata],
            ],
            None,
            ctype[np.generic[Any]],
        ],
        DT_,
    ]: ...

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, ...], DT_],
    ) -> Array[
        TupleBasis[
            tuple[FundamentalBasis[SimpleMetadata], ...], None, ctype[np.generic[Any]]
        ],
        DT_,
    ]: ...

    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, ...], DT_],
    ) -> Array[
        TupleBasis[
            tuple[FundamentalBasis[SimpleMetadata], ...], None, ctype[np.generic[Any]]
        ],
        DT_,
    ]:
        """Get a Array from an array."""
        return ArrayBuilder(basis.from_shape(array.shape), array).ok()

    def with_basis[
        DT_: np.dtype[np.generic],
        M0_: BasisMetadata,
        B1_: Basis,
    ](
        self: Array[Basis[M0_, Any], DT_],
        basis: B1_,
    ) -> ArrayConversion[M0_, B1_, DT_]:
        """Get the Array with the basis set to basis."""
        return ArrayConversion(self.raw_data, self.basis, basis)

    def __add__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, ctype[DT_]], np.dtype[DT_]]:
        as_add_basis = basis.as_add_basis(self.basis)
        data = as_add_basis.add_data(
            self.with_basis(as_add_basis).ok().raw_data,
            other.with_basis(as_add_basis).ok().raw_data,
        )

        return ArrayBuilder(as_add_basis, data).ok()

    def __sub__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
        other: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
    ) -> Array[Basis[M_, ctype[DT_]], np.dtype[DT_]]:
        as_sub_basis = basis.as_sub_basis(self.basis)
        data = as_sub_basis.sub_data(
            self.with_basis(as_sub_basis).ok().raw_data,
            other.with_basis(as_sub_basis).ok().raw_data,
        )

        return ArrayBuilder(as_sub_basis, data).ok()

    def __mul__[M_: BasisMetadata, DT_: np.number](
        self: Array[Basis[M_, ctype[DT_]], np.dtype[DT_]],
        other: float,
    ) -> Array[Basis[M_, ctype[DT_]], np.dtype[DT_]]:
        as_mul_basis = basis.as_mul_basis(self.basis)
        data = as_mul_basis.mul_data(self.with_basis(as_mul_basis).ok().raw_data, other)
        return ArrayBuilder(as_mul_basis, data).ok()

    @overload
    def __iter__[DT_: np.dtype[np.generic]](
        self: Array[Basis[TupleMetadata[tuple[BasisMetadata], Any], Any], DT_],
        /,
    ) -> Never: ...
    @overload
    def __iter__[M1: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[Basis[TupleMetadata[tuple[BasisMetadata, M1], Any], Any], DT_],
        /,
    ) -> Iterator[Array[Basis[M1, Any], DT_]]: ...
    @overload
    def __iter__[M1: BasisMetadata, M2: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[Basis[TupleMetadata[tuple[BasisMetadata, M1, M2], Any], Any], DT_],
        /,
    ) -> Iterator[
        Array[TupleBasis[tuple[Basis[M1, Any], Basis[M2, Any]], Any, Any], DT_]
    ]: ...
    @overload
    def __iter__[M1: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[Basis[TupleMetadata[tuple[M1, ...], Any]], DT_], /
    ) -> Iterator[Array[Basis[TupleMetadata[tuple[M1, ...], None]], DT_]]: ...

    def __iter__(
        self: Array[Basis[TupleMetadata[tuple[BasisMetadata, ...], Any]], Any], /
    ) -> Iterator[Array[Any, Any]]:
        basis_as_tuple = basis.with_modified_child(
            basis.as_tuple_basis(self.basis), basis.as_fundamental, 0
        )
        as_tuple = self.with_basis(basis_as_tuple).ok()
        children = as_tuple.basis.children[1:]
        match len(children):
            case 0:
                msg = "Cannot iterate over a flat array."
                raise ValueError(msg)
            case 1:
                out_basis = children[0]
            case _:
                out_basis = TupleBasis(children)

        return (
            ArrayBuilder(out_basis, row).ok()
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    @overload
    def __getitem__[DT1: ctype[Never], DT_: np.dtype[np.generic]](
        self: Array[Any, DT_], index: int
    ) -> DT_: ...
    @overload
    def __getitem__[DT1: ctype[Never], DT_: np.dtype[np.generic]](
        self: Array[Basis[Any, DT1], DT_], index: tuple[NestedIndex, ...] | slice
    ) -> Array[Basis[BasisMetadata, DT1], DT_]: ...

    def __getitem__[DT1: ctype[Never], DT_: np.dtype[np.generic]](
        self: Array[Basis[Any, DT1], DT_],
        index: NestedIndex,
    ) -> Array[Basis[BasisMetadata, Any], DT_] | DT_:
        indexed_basis, indexed_data = _index_raw_along_axis(
            index, self.basis, self.raw_data.reshape(-1, 1), axis=0
        )
        if indexed_basis is None:
            return cast("DT_", indexed_data.item())
        return ArrayBuilder(indexed_basis, indexed_data).ok()
