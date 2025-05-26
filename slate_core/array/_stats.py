from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from slate_core import basis
from slate_core.array._array import Array
from slate_core.basis import TupleBasis
from slate_core.metadata import BasisMetadata, SimpleMetadata, is_tuple_metadata

if TYPE_CHECKING:
    from slate_core.array._array import ArrayWithMetadata
    from slate_core.metadata import TupleMetadata


@overload
def standard_deviation[M: tuple[SimpleMetadata], DT: np.dtype[np.number]](
    array: ArrayWithMetadata[TupleMetadata[tuple[BasisMetadata, *M], Any], DT],
    *,
    axis: Literal[0],
) -> ArrayWithMetadata[TupleMetadata[tuple[*M], None], DT]: ...


@overload
def standard_deviation[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[TupleMetadata[tuple[M, ...], Any], DT], *, axis: int
) -> ArrayWithMetadata[TupleMetadata[tuple[M, ...], None], DT]: ...


@overload
def standard_deviation[DT: np.number](
    array: ArrayWithMetadata[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def standard_deviation[DT: np.number](
    array: ArrayWithMetadata[BasisMetadata, np.dtype[DT]], axis: int | None = None
) -> ArrayWithMetadata[BasisMetadata, np.dtype[DT]] | DT:
    """Calculate the standard deviation along the specified axis.

    Computes the standard deviation of array elements over the given axis.
    When axis is None, the standard deviation of all array elements is computed
    as a scalar. Otherwise, a new array with the standard deviation along the
    specified axis is returned.

    Examples
    --------
    .. literalinclude:: ../../tests/array/stats_test.py
        :language: python
        :lineno-start: 56
        :lines: 56-80
        :dedent: 4
    """
    if axis is None:
        # Ensure dtype is preserved for complex numbers too
        return np.std(array.as_array(), axis=axis, dtype=array.raw_data.dtype)  # type: ignore unknown
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
    return Array(out_basis.resolve_ctype(), data)


@overload
def average[M: tuple[SimpleMetadata], DT: np.dtype[np.number]](
    array: ArrayWithMetadata[TupleMetadata[tuple[BasisMetadata, *M], Any], DT],
    *,
    axis: Literal[0],
) -> ArrayWithMetadata[TupleMetadata[tuple[*M], None], DT]: ...


@overload
def average[M: BasisMetadata, DT: np.dtype[np.number]](
    array: ArrayWithMetadata[TupleMetadata[tuple[M, ...], Any], DT], *, axis: int
) -> ArrayWithMetadata[TupleMetadata[tuple[M, ...], None], DT]: ...


@overload
def average[DT: np.number](
    array: ArrayWithMetadata[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def average[DT: np.number](
    array: ArrayWithMetadata[BasisMetadata, np.dtype[DT]], axis: int | None = None
) -> ArrayWithMetadata[BasisMetadata, np.dtype[DT]] | DT:
    """Calculate the average (mean) along the specified axis.

    Computes the arithmetic mean of array elements over the given axis.
    When axis is None, the mean of all array elements is computed as a scalar.
    Otherwise, a new array with the mean along the specified axis is returned.

    Examples
    --------
    .. literalinclude:: ../../tests/array/stats_test.py
        :language: python
        :lineno-start: 83
        :lines: 83-107
        :dedent: 4
    """
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
    ).resolve_ctype()
    return Array(out_basis, data)


@overload
def min[M: tuple[SimpleMetadata], DT: np.dtype[np.number]](  # noqa: A001
    array: ArrayWithMetadata[TupleMetadata[tuple[BasisMetadata, *M], Any], DT],
    *,
    axis: Literal[0],
) -> ArrayWithMetadata[TupleMetadata[tuple[*M], None], DT]: ...


@overload
def min[M: BasisMetadata, DT: np.dtype[np.number]](  # noqa: A001
    array: ArrayWithMetadata[TupleMetadata[tuple[M, ...], Any], DT], *, axis: int
) -> ArrayWithMetadata[TupleMetadata[tuple[M, ...], None], DT]: ...


@overload
def min[DT: np.number](  # noqa: A001
    array: ArrayWithMetadata[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def min[DT: np.number](  # noqa: A001
    array: ArrayWithMetadata[BasisMetadata, np.dtype[DT]], axis: int | None = None
) -> ArrayWithMetadata[BasisMetadata, np.dtype[DT]] | DT:
    """Calculate the minimum value along the specified axis.

    Computes the minimum of array elements over the given axis.
    When axis is None, the minimum of all array elements is computed as a scalar.
    Otherwise, a new array with the minimum values along the specified axis is returned.

    Examples
    --------
    .. literalinclude:: ../../tests/array/stats_test.py
        :language: python
        :lineno-start: 109
        :lines: 109-131
        :dedent: 4
    """
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
    ).resolve_ctype()
    return Array(out_basis, data)


@overload
def max[M: tuple[SimpleMetadata], DT: np.dtype[np.number]](  # noqa: A001
    array: ArrayWithMetadata[TupleMetadata[tuple[BasisMetadata, *M], Any], DT],
    *,
    axis: Literal[0],
) -> ArrayWithMetadata[TupleMetadata[tuple[*M], None], DT]: ...


@overload
def max[M: BasisMetadata, DT: np.dtype[np.number]](  # noqa: A001
    array: ArrayWithMetadata[TupleMetadata[tuple[M, ...], Any], DT], *, axis: int
) -> ArrayWithMetadata[TupleMetadata[tuple[M, ...], None], DT]: ...


@overload
def max[DT: np.number](  # noqa: A001
    array: ArrayWithMetadata[Any, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def max[DT: np.number](  # noqa: A001
    array: ArrayWithMetadata[BasisMetadata, np.dtype[DT]], axis: int | None = None
) -> ArrayWithMetadata[BasisMetadata, np.dtype[DT]] | DT:
    """Calculate the maximum value along the specified axis.

    Computes the maximum of array elements over the given axis.
    When axis is None, the maximum of all array elements is computed as a scalar.
    Otherwise, a new array with the maximum values along the specified axis is returned.

    Examples
    --------
    .. literalinclude:: ../../tests/array/stats_test.py
        :language: python
        :lineno-start: 133
        :lines: 133-155
        :dedent: 4
    """
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
    ).resolve_ctype()
    return Array(out_basis, data)
