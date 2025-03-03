from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from slate import basis
from slate.array import build
from slate.basis import (
    Basis,
    DiagonalBasis,
    TupleBasis,
)
from slate.basis._tuple import is_tuple_basis_like
from slate.metadata import (
    BasisMetadata,
    SimpleMetadata,
    TupleMetadata,
)

if TYPE_CHECKING:
    from slate.array import Array
    from slate.basis._basis import ctype
    from slate.basis._tuple import TupleBasisLike


def extract_diagonal[M1: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[Basis[TupleMetadata[tuple[Any, M1], E]], DT],
) -> Array[Basis[M1, ctype[np.generic]], DT]:
    b = DiagonalBasis(basis.as_tuple_basis(basis.as_fundamental(array.basis)))
    converted = array.with_basis(b).ok()

    return build(converted.basis.inner.children[1], converted.raw_data).ok()


@overload
def norm[M: SimpleMetadata, DT: np.dtype[np.number]](
    array: Array[TupleBasisLike[tuple[Any, M], None], DT], *, axis: Literal[0]
) -> Array[TupleBasisLike[tuple[M], None, ctype[np.generic]], DT]: ...


@overload
def norm[M: BasisMetadata, DT: np.dtype[np.number]](
    array: Array[TupleBasisLike[tuple[M, ...]], DT], *, axis: int
) -> Array[TupleBasisLike[tuple[M, ...], None, ctype[np.generic]], DT]: ...


@overload
def norm[DT: np.dtype[np.number]](
    array: Array[Basis, DT], *, axis: None = ...
) -> DT: ...


def norm[DT: np.dtype[np.number]](
    array: Array[Basis, DT], axis: int | None = None
) -> Array[Any, DT] | DT:
    if axis is None:
        return np.linalg.norm(array.as_array(), axis=axis)  # type: ignore unknown
    data = cast("np.ndarray[Any, DT]", np.linalg.norm(array.as_array(), axis=axis))
    assert is_tuple_basis_like(array.basis)
    full_basis = basis.from_metadata(array.basis.metadata())

    axis %= len(full_basis.children)
    out_basis = TupleBasis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    ).upcast()
    return build(out_basis, data).ok()
