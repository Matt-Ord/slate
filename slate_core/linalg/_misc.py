from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast, overload

import numpy as np

from slate_core import basis
from slate_core.array import Array
from slate_core.basis import (
    Basis,
    DiagonalBasis,
    TupleBasis,
)
from slate_core.basis._tuple import is_tuple_basis_like
from slate_core.metadata import (
    BasisMetadata,
    SimpleMetadata,
    TupleMetadata,
)

if TYPE_CHECKING:
    from slate_core.array import ArrayWithMetadata
    from slate_core.basis._basis import Ctype
    from slate_core.basis._tuple import TupleBasisLike


def extract_diagonal[M1: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: ArrayWithMetadata[TupleMetadata[tuple[BasisMetadata, M1], E], DT],
) -> Array[Basis[M1, Ctype[np.generic]], DT]:
    b = DiagonalBasis(basis.as_tuple(basis.as_fundamental(array.basis))).resolve_ctype()
    converted = array.with_basis(b)

    return Array(converted.basis.inner.children[1], converted.raw_data)


@overload
def norm[M: SimpleMetadata, DT: np.dtype[np.number]](
    array: Array[TupleBasisLike[tuple[BasisMetadata, M], None], DT], *, axis: Literal[0]
) -> Array[TupleBasisLike[tuple[M], None, Ctype[np.generic]], DT]: ...


@overload
def norm[M: BasisMetadata, DT: np.dtype[np.number]](
    array: Array[TupleBasisLike[tuple[M, ...]], DT], *, axis: int
) -> Array[TupleBasisLike[tuple[M, ...], None, Ctype[np.generic]], DT]: ...


@overload
def norm[DT: np.number](
    array: Array[Basis, np.dtype[DT]], *, axis: None = ...
) -> DT: ...


def norm[DT: np.number](
    array: Array[Basis, np.dtype[DT]], axis: int | None = None
) -> Array[Basis, np.dtype[DT]] | DT:
    if axis is None:
        return np.linalg.norm(array.as_array(), axis=axis)  # type: ignore unknown
    data = cast(
        "np.ndarray[Any, np.dtype[DT]]", np.linalg.norm(array.as_array(), axis=axis)
    )
    assert is_tuple_basis_like(array.basis)
    full_basis = basis.from_metadata(array.basis.metadata())
    axis %= len(full_basis.children)
    out_basis = TupleBasis(
        tuple(b for i, b in enumerate(full_basis.children) if i != axis)
    ).resolve_ctype()
    return Array(out_basis, data)
