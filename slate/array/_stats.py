from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from slate import basis
from slate.array._array import Array
from slate.basis._tuple import TupleBasis, TupleBasis1D, tuple_basis
from slate.metadata._metadata import BasisMetadata, SimpleMetadata

if TYPE_CHECKING:
    from slate.basis._fundamental import FundamentalBasis
    from slate.metadata.stacked import Metadata1D, Metadata2D, StackedMetadata


@overload
def standard_deviation[M: SimpleMetadata, DT: np.number[Any]](
    array: Array[Metadata2D[Any, M, Any], DT], *, axis: Literal[0]
) -> Array[Metadata1D[M, None], DT, TupleBasis1D[DT, FundamentalBasis[M], None]]: ...


@overload
def standard_deviation[M: BasisMetadata, DT: np.number[Any]](
    array: Array[StackedMetadata[M, Any], DT], *, axis: int
) -> Array[StackedMetadata[M, None], DT]: ...


@overload
def standard_deviation[DT: np.number[Any]](
    array: Array[Any, DT], *, axis: None = ...
) -> DT: ...


def standard_deviation[DT: np.number[Any]](
    array: Array[Any, DT], axis: int | None = None
) -> Array[Any, DT] | DT:
    if axis is None:
        return np.std(array.as_array(), axis=axis)  # type: ignore unknown
    data = np.asarray(
        cast("Any", np.std(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = cast(
        "TupleBasis[Any, Any, np.generic]", basis.from_metadata(array.basis.metadata())
    )

    axis %= len(full_basis.children)
    out_basis = tuple_basis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    )
    return Array(out_basis, data)


@overload
def average[M: SimpleMetadata, DT: np.number[Any]](
    array: Array[Metadata2D[Any, M, Any], DT], *, axis: Literal[0]
) -> Array[Metadata1D[M, None], DT, TupleBasis1D[DT, FundamentalBasis[M], None]]: ...


@overload
def average[M: BasisMetadata, DT: np.number[Any]](
    array: Array[StackedMetadata[M, Any], DT], *, axis: int
) -> Array[StackedMetadata[M, None], DT]: ...


@overload
def average[DT: np.number[Any]](array: Array[Any, DT], *, axis: None = ...) -> DT: ...


def average[DT: np.number[Any]](
    array: Array[Any, DT], axis: int | None = None
) -> Array[Any, DT] | DT:
    if axis is None:
        return np.average(array.as_array(), axis=axis)  # type: ignore unknown
    data = np.asarray(
        cast("Any", np.average(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = cast(
        "TupleBasis[Any, Any, np.generic]", basis.from_metadata(array.basis.metadata())
    )

    axis %= len(full_basis.children)
    out_basis = tuple_basis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    )
    return Array(out_basis, data)


@overload
def min[M: SimpleMetadata, DT: np.number[Any]](  # noqa: A001
    array: Array[Metadata2D[Any, M, Any], DT], *, axis: Literal[0]
) -> Array[Metadata1D[M, None], DT, TupleBasis1D[DT, FundamentalBasis[M], None]]: ...


@overload
def min[M: BasisMetadata, DT: np.number[Any]](  # noqa: A001
    array: Array[StackedMetadata[M, Any], DT], *, axis: int
) -> Array[StackedMetadata[M, None], DT]: ...


@overload
def min[DT: np.number[Any]](array: Array[Any, DT], *, axis: None = ...) -> DT: ...  # noqa: A001


def min[DT: np.number[Any]](  # noqa: A001
    array: Array[Any, DT], axis: int | None = None
) -> Array[Any, DT] | DT:
    if axis is None:
        return np.min(array.as_array(), axis=axis)  # type: ignore unknown
    data = np.asarray(
        cast("Any", np.min(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = cast(
        "TupleBasis[Any, Any, np.generic]", basis.from_metadata(array.basis.metadata())
    )

    axis %= len(full_basis.children)
    out_basis = tuple_basis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    )
    return Array(out_basis, data)


@overload
def max[M: SimpleMetadata, DT: np.number[Any]](  # noqa: A001
    array: Array[Metadata2D[Any, M, Any], DT], *, axis: Literal[0]
) -> Array[Metadata1D[M, None], DT, TupleBasis1D[DT, FundamentalBasis[M], None]]: ...


@overload
def max[M: BasisMetadata, DT: np.number[Any]](  # noqa: A001
    array: Array[StackedMetadata[M, Any], DT], *, axis: int
) -> Array[StackedMetadata[M, None], DT]: ...


@overload
def max[DT: np.number[Any]](array: Array[Any, DT], *, axis: None = ...) -> DT: ...  # noqa: A001


def max[DT: np.number[Any]](  # noqa: A001
    array: Array[Any, DT], axis: int | None = None
) -> Array[Any, DT] | DT:
    if axis is None:
        return np.max(array.as_array(), axis=axis)  # type: ignore unknown
    data = np.asarray(
        cast("Any", np.max(array.as_array(), axis=axis)),  # type: ignore unknown
        dtype=array.raw_data.dtype,
    )
    full_basis = cast(
        "TupleBasis[Any, Any, np.generic]", basis.from_metadata(array.basis.metadata())
    )

    axis %= len(full_basis.children)
    out_basis = tuple_basis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    )
    return Array(out_basis, data)
