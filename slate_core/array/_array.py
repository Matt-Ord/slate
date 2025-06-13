from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, cast, overload

import numpy as np

from slate_core import basis
from slate_core.basis import Basis, FundamentalBasis, TupleBasis, is_tuple_basis_like
from slate_core.metadata import (
    SIMPLE_FEATURE,
    BasisMetadata,
    NestedLength,
    SimpleMetadata,
    TupleMetadata,
    basis_components_from_array_points,
    shallow_shape_from_nested,
)
from slate_core.util._index import recast_along_axes, slice_along_axis

if TYPE_CHECKING:
    from collections.abc import Iterator

    from slate_core.basis import Ctype, TupleBasisLike

type Index = int | slice
type NestedIndex = Index | tuple[NestedIndex, ...]


def _index_single_raw_along_axis(
    index: Index,
    data_basis: Basis,
    data: np.ndarray[Any, Any],
    *,
    axis: int = -1,
) -> tuple[Basis | None, np.ndarray[Any, Any]]:
    if index == slice(None):
        return data_basis, data
    axis %= data.ndim

    data = data_basis.__into_fundamental__(data, axis=axis).ok()
    out = data[slice_along_axis(index, axis=axis)]

    # Scale data according to the weights of the basis
    if SIMPLE_FEATURE not in data_basis.metadata().features:
        weights = data_basis.metadata().basis_weights[index]
        out = cast(
            "np.ndarray[Any, Any]",
            out * weights.reshape(recast_along_axes((weights.size,), (axis,))),
        )

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
) -> tuple[Basis | None, np.ndarray[Any, Any]]:
    axis %= data.ndim
    tuple_basis = basis.as_tuple(basis.as_linear_map(data_basis))
    children = tuple_basis.children
    stacked_shape = (
        data.shape[:axis] + tuple(c.size for c in children) + data.shape[axis + 1 :]
    )

    data = data_basis.__convert_vector_into__(data, tuple_basis, axis=axis).ok()
    data = data.reshape(stacked_shape)

    final_basis = list[Basis]()
    for child_index, child in zip(index, children, strict=True):
        child_axis = axis + len(final_basis)
        b, data = _index_raw_along_axis(child_index, child, data, axis=child_axis)
        if b is not None:
            final_basis.append(b)
    if len(final_basis) == 0:
        return None, data.reshape(data.shape[:axis] + data.shape[axis + 1 :])

    data = data.reshape((*data.shape[:axis], -1, *data.shape[axis + 1 :]))
    if len(final_basis) == 1:
        return final_basis[0], data
    return TupleBasis(tuple(final_basis), None), data


def _index_raw_along_axis[DT: np.dtype[np.generic]](
    index: NestedIndex,
    basis: Basis,
    data: np.ndarray[Any, DT],
    axis: int,
) -> tuple[Basis | None, np.ndarray[Any, DT]]:
    if isinstance(index, tuple):
        assert is_tuple_basis_like(basis)
        return _index_tuple_raw_along_axis(index, basis, data, axis=axis)
    return _index_single_raw_along_axis(index, basis, data, axis=axis)


class Array[B: Basis, DT: np.dtype[np.generic]]:
    """
    An array with data stored in a given basis.

    The basis MUST be able to support the dtype of the data.
    This makes constructing an array directly unsafe - it must be done via the ArrayBuilder interface.
    """

    def __init__(self, basis: B, data: np.ndarray[Any, DT]) -> None:
        assert basis.size == data.size
        basis.ctype.assert_supports_dtype(data.dtype)
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
        assert self.raw_data.dtype == data.dtype
        self._data = cast("np.ndarray[tuple[int], Any]", data.ravel())

    def as_array(self) -> np.ndarray[Any, DT]:
        """Get the data as a (dense) np.array."""
        fundamental = basis.as_fundamental(self.basis)
        shape = shallow_shape_from_nested(fundamental.fundamental_shape)
        fundamental_raw = self.with_basis(fundamental).raw_data
        if SIMPLE_FEATURE not in self.basis.metadata().features:
            fundamental_raw = cast(
                "np.ndarray[Any, DT]",
                fundamental_raw * fundamental.metadata().basis_weights,  # type: ignore[return-value]
            )

        return fundamental_raw.reshape(shape)

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int,], DT_], *, metadata: None = None
    ) -> Array[
        TupleBasis[
            tuple[FundamentalBasis[SimpleMetadata]], None, Ctype[np.generic[Any]]
        ],
        DT_,
    ]: ...

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, int], DT_], *, metadata: None = None
    ) -> Array[
        TupleBasis[
            tuple[
                FundamentalBasis[SimpleMetadata],
                FundamentalBasis[SimpleMetadata],
            ],
            None,
            Ctype[np.generic[Any]],
        ],
        DT_,
    ]: ...

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, int, int], DT_], *, metadata: None = None
    ) -> Array[
        TupleBasis[
            tuple[
                FundamentalBasis[SimpleMetadata],
                FundamentalBasis[SimpleMetadata],
                FundamentalBasis[SimpleMetadata],
            ],
            None,
            Ctype[np.generic[Any]],
        ],
        DT_,
    ]: ...

    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[Any, ...], DT_], *, metadata: None = None
    ) -> Array[
        TupleBasis[
            tuple[FundamentalBasis[SimpleMetadata], ...], None, Ctype[np.generic[Any]]
        ],
        DT_,
    ]: ...
    @overload
    @staticmethod
    def from_array[DT_: np.dtype[np.generic], M: BasisMetadata](
        array: np.ndarray[tuple[Any, ...], DT_], *, metadata: M
    ) -> Array[Basis[M], DT_]: ...
    @staticmethod
    def from_array[DT_: np.dtype[np.generic]](
        array: np.ndarray[tuple[int, ...], DT_],
        *,
        metadata: BasisMetadata | None = None,
    ) -> Array[
        TupleBasis[
            tuple[FundamentalBasis[SimpleMetadata], ...],
            None,
            Ctype[np.generic[Any]],
        ]
        | Basis[BasisMetadata],
        DT_,
    ]:
        """Get a Array from an array."""
        if metadata is None:
            metadata = TupleMetadata.from_shape(array.shape)
        if array.ndim > 1:
            assert shallow_shape_from_nested(metadata.fundamental_shape) == array.shape
        raw_data = (
            array
            if SIMPLE_FEATURE in metadata.features
            else basis_components_from_array_points(metadata, array)
        )
        return Array(basis.from_metadata(metadata), raw_data)

    def with_basis[B1_: Basis](self, basis: B1_) -> Array[B1_, DT]:
        """Get the Array with the basis set to basis."""
        basis.ctype.assert_supports_dtype(self.dtype)
        assert basis.metadata() == self.basis.metadata()
        new_data = self.basis.__convert_vector_into__(self.raw_data, basis).ok()
        return Array(basis, new_data)  # type: ignore[return-value]

    def as_type[
        M_: BasisMetadata,
        T: np.number,
    ](
        self: ArrayWithMetadata[M_, np.dtype[np.generic]],
        ty: type[T],
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[T]]:
        as_type_basis = basis.as_supports_type(self.basis, ty)
        converted = self.with_basis(as_type_basis)
        return Array(converted.basis, converted.raw_data.astype(ty))

    @overload
    def __add__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]],
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[T]]: ...
    @overload
    def __add__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...

    def __add__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]] | complex,
    ) -> Any:
        if isinstance(other, (complex, float, int)):
            final_basis = basis.as_supports_type(
                basis.as_index(self.basis),
                cast("type[np.number]", np.result_type(self.dtype, other).type),
            )
            data = self.with_basis(final_basis).raw_data + other
            return Array(final_basis, data)
        # TODO: support efficient add for Coordinate-Like basis  # noqa: FIX002
        # this would involve a more precise get_common_basis
        result_type = cast(
            "type[np.number]", np.result_type(self.dtype, other.dtype).type
        )
        final_basis = basis.get_common(
            basis.as_supports_type(basis.as_linear_map(self.basis), result_type),
            basis.as_supports_type(basis.as_linear_map(other.basis), result_type),
        )
        data = (
            self.with_basis(final_basis).raw_data
            + other.with_basis(final_basis).raw_data
        )
        return Array(final_basis, data)

    @overload
    def __sub__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]],
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[T]]: ...
    @overload
    def __sub__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: complex,
    ) -> Array[Basis[M_, Ctype[T]], np.dtype[np.number]]: ...

    def __sub__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: ArrayWithMetadata[M_, np.dtype[T]] | complex,
    ) -> Any:
        if isinstance(other, (complex, float, int)):
            final_basis = basis.as_supports_type(
                basis.as_index(self.basis),
                cast("type[np.number]", np.result_type(self.dtype, other).type),
            )
            data = self.with_basis(final_basis).raw_data - other
            return Array(final_basis, data)
        # TODO: support efficient sub for Coordinate-Like basis  # noqa: FIX002
        # this would involve a more precise get_common_basis
        result_type = cast(
            "type[np.number]", np.result_type(self.dtype, other.dtype).type
        )
        final_basis = basis.get_common(
            basis.as_supports_type(basis.as_linear_map(self.basis), result_type),
            basis.as_supports_type(basis.as_linear_map(other.basis), result_type),
        )
        data = (
            self.with_basis(final_basis).raw_data
            - other.with_basis(final_basis).raw_data
        )

        return Array(final_basis, data)

    def __mul__[M_: BasisMetadata, T: np.number](
        self: ArrayWithMetadata[M_, np.dtype[T]],
        other: complex,
    ) -> ArrayWithMetadata[M_, np.dtype[np.number]]:
        final_basis = basis.as_supports_type(
            basis.as_mul(self.basis),
            cast("type[np.number]", np.result_type(self.dtype, other).type),
        )
        data = final_basis.mul_data(self.with_basis(final_basis).raw_data, other)
        return Array(final_basis, data)

    @overload
    def __iter__[DT_: np.dtype[np.generic]](
        self: Array[Basis[TupleMetadata[tuple[BasisMetadata], Any], Any], DT_],
        /,
    ) -> Never: ...
    @overload
    def __iter__[M1: BasisMetadata, DT_: np.dtype[np.generic]](
        self: Array[Basis[TupleMetadata[tuple[BasisMetadata, M1], Any], Any], DT_],
        /,
    ) -> Iterator[ArrayWithMetadata[M1, DT_]]: ...
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
            basis.as_tuple(self.basis), basis.as_fundamental, 0
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
                out_basis = TupleBasis(children)

        return (
            Array(out_basis, row)
            for row in as_tuple.raw_data.reshape(as_tuple.basis.shape)
        )

    @overload
    def __getitem__(self, index: int) -> DT: ...
    @overload
    def __getitem__[CT: Ctype[Never]](
        self: Array[Basis[BasisMetadata, CT], DT],
        index: tuple[NestedIndex, ...] | slice,
    ) -> Array[Basis[BasisMetadata, CT], DT]: ...

    def __getitem__[CT: Ctype[Never], DT_: np.dtype[np.generic]](
        self: Array[Basis[BasisMetadata, CT], DT_],
        index: NestedIndex,
    ) -> Array[Basis[BasisMetadata], DT_] | DT_:
        indexed_basis, indexed_data = _index_raw_along_axis(
            index, self.basis, self.raw_data.reshape(-1, 1), axis=0
        )
        if indexed_basis is None:
            return cast("DT_", indexed_data.item())
        return Array(indexed_basis, indexed_data)


type ArrayWithMetadata[M: BasisMetadata, DT: np.dtype[np.generic]] = Array[Basis[M], DT]
