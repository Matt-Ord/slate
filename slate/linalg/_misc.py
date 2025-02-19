from __future__ import annotations

from typing import Any, Literal, cast, overload

import numpy as np

from slate import basis
from slate.array import Array
from slate.basis import (
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
    TupleBasis1D,
    tuple_basis,
)
from slate.metadata import (
    BasisMetadata,
    Metadata1D,
    Metadata2D,
    SimpleMetadata,
    StackedMetadata,
)


def extract_diagonal[M: BasisMetadata, E, DT: np.generic](
    array: Array[Metadata2D[M, M, E], DT],
) -> Array[M, DT, Basis[M, Any]]:
    b = DiagonalBasis(basis.as_tuple_basis(basis.as_fundamental(array.basis)))
    converted = array.with_basis(b)

    return Array(converted.basis.inner[1], converted.raw_data)


@overload
def norm[M: SimpleMetadata, DT: np.number[Any]](
    array: Array[Metadata2D[Any, M, Any], DT], *, axis: Literal[0]
) -> Array[Metadata1D[M, None], DT, TupleBasis1D[DT, FundamentalBasis[M], None]]: ...


@overload
def norm[M: BasisMetadata, DT: np.number[Any]](
    array: Array[StackedMetadata[M, Any], DT], *, axis: int
) -> Array[StackedMetadata[M, None], DT]: ...


@overload
def norm[DT: np.number[Any]](array: Array[Any, DT], *, axis: None = ...) -> DT: ...


def norm[DT: np.number[Any]](
    array: Array[Any, DT], axis: int | None = None
) -> Array[Any, DT] | DT:
    if axis is None:
        return np.linalg.norm(array.as_array(), axis=axis)  # type: ignore unknown
    data = np.asarray(
        cast("Any", np.linalg.norm(array.as_array(), axis=axis)),  # type: ignore unknown
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
