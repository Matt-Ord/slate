from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from slate import basis
from slate.array._array import Array, ArrayBuilder
from slate.basis import TupleBasis
from slate.metadata import BasisMetadata, SimpleMetadata, is_tuple_metadata

if TYPE_CHECKING:
    from slate.basis import Basis
    from slate.metadata import TupleMetadata


@overload
def standard_deviation[M: tuple[SimpleMetadata], DT: np.dtype[np.number[Any]]](
    array: Array[Basis[TupleMetadata[tuple[BasisMetadata, *M], Any]], DT],
    *,
    axis: Literal[0],
) -> Array[Basis[TupleMetadata[tuple[*M], None]], DT]: ...


@overload
def standard_deviation[M: BasisMetadata, DT: np.dtype[np.number[Any]]](
    array: Array[Basis[TupleMetadata[tuple[M, ...], Any]], DT], *, axis: int
) -> Array[Basis[TupleMetadata[tuple[M, ...], None]], DT]: ...


@overload
def standard_deviation[DT: np.number[Any]](
    array: Array[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def standard_deviation[DT: np.number[Any]](
    array: Array[Basis, np.dtype[DT]], axis: int | None = None
) -> Array[Any, np.dtype[DT]] | DT:
    if axis is None:
        return np.std(array.as_array(), axis=axis)  # type: ignore unknown
    meta = array.basis.metadata()
    assert is_tuple_metadata(meta)
    data = np.array(
        np.std(cast("Any", array.as_array()), axis=axis),
        dtype=array.raw_data.dtype,
    )
    full_basis = basis.from_metadata(meta)

    axis %= len(full_basis.children)
    out_basis = TupleBasis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    )
    return ArrayBuilder(out_basis.upcast(), data).ok()


@overload
def average[M: tuple[SimpleMetadata], DT: np.dtype[np.number[Any]]](
    array: Array[Basis[TupleMetadata[tuple[BasisMetadata, *M], Any]], DT],
    *,
    axis: Literal[0],
) -> Array[Basis[TupleMetadata[tuple[*M], None]], DT]: ...


@overload
def average[M: BasisMetadata, DT: np.dtype[np.number[Any]]](
    array: Array[Basis[TupleMetadata[tuple[M, ...], Any]], DT], *, axis: int
) -> Array[Basis[TupleMetadata[tuple[M, ...], None]], DT]: ...


@overload
def average[DT: np.number[Any]](
    array: Array[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def average[DT: np.number[Any]](
    array: Array[Basis, np.dtype[DT]], axis: int | None = None
) -> Array[Any, np.dtype[DT]] | DT:
    if axis is None:
        return np.average(array.as_array(), axis=axis)  # type: ignore unknown
    meta = array.basis.metadata()
    assert is_tuple_metadata(meta)
    data = np.asarray(
        cast("Any", np.average(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = basis.from_metadata(meta)

    axis %= len(full_basis.children)
    out_basis = TupleBasis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    ).upcast()
    return ArrayBuilder(out_basis, data).ok()


@overload
def min[M: tuple[SimpleMetadata], DT: np.dtype[np.number[Any]]](  # noqa: A001
    array: Array[Basis[TupleMetadata[tuple[BasisMetadata, *M], Any]], DT],
    *,
    axis: Literal[0],
) -> Array[Basis[TupleMetadata[tuple[*M], None]], DT]: ...


@overload
def min[M: BasisMetadata, DT: np.dtype[np.number[Any]]](  # noqa: A001
    array: Array[Basis[TupleMetadata[tuple[M, ...], Any]], DT], *, axis: int
) -> Array[Basis[TupleMetadata[tuple[M, ...], None]], DT]: ...


@overload
def min[DT: np.number[Any]](  # noqa: A001
    array: Array[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def min[DT: np.number[Any]](  # noqa: A001
    array: Array[Basis, np.dtype[DT]], axis: int | None = None
) -> Array[Any, np.dtype[DT]] | DT:
    if axis is None:
        return np.min(array.as_array(), axis=axis)  # type: ignore unknown
    meta = array.basis.metadata()
    assert is_tuple_metadata(meta)
    data = np.asarray(
        cast("Any", np.min(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = basis.from_metadata(meta)

    axis %= len(full_basis.children)
    out_basis = TupleBasis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    ).upcast()
    return ArrayBuilder(out_basis, data).ok()


@overload
def max[M: tuple[SimpleMetadata], DT: np.dtype[np.number[Any]]](  # noqa: A001
    array: Array[Basis[TupleMetadata[tuple[BasisMetadata, *M], Any]], DT],
    *,
    axis: Literal[0],
) -> Array[Basis[TupleMetadata[tuple[*M], None]], DT]: ...


@overload
def max[M: BasisMetadata, DT: np.dtype[np.number[Any]]](  # noqa: A001
    array: Array[Basis[TupleMetadata[tuple[M, ...], Any]], DT], *, axis: int
) -> Array[Basis[TupleMetadata[tuple[M, ...], None]], DT]: ...


@overload
def max[DT: np.number[Any]](  # noqa: A001
    array: Array[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def max[DT: np.number[Any]](  # noqa: A001
    array: Array[Basis, np.dtype[DT]], axis: int | None = None
) -> Array[Any, np.dtype[DT]] | DT:
    if axis is None:
        return np.max(array.as_array(), axis=axis)  # type: ignore unknown
    meta = array.basis.metadata()
    assert is_tuple_metadata(meta)
    data = np.asarray(
        cast("Any", np.max(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = basis.from_metadata(meta)

    axis %= len(full_basis.children)
    out_basis = TupleBasis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    ).upcast()
    return ArrayBuilder(out_basis, data).ok()
