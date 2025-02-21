from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, overload

import numpy as np

from slate.array._array import Array
from slate.array._conversion import (
    as_diagonal_basis,
    as_index_basis,
    as_tuple_basis,
    nest,
)
from slate.metadata import BasisMetadata
from slate.util._index import get_position_in_sorted, slice_ignoring_axes

if TYPE_CHECKING:
    from slate.basis import (
        Basis,
    )
    from slate.basis._diagonal import DiagonalBasis
    from slate.basis._tuple import TupleBasis
    from slate.metadata.stacked import TupleMetadata


def conjugate[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, DT]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    raw_data = converted.raw_data
    converted_data = cast(
        "np.ndarray[Any, DT]", np.conj(raw_data).astype(raw_data.dtype)
    )
    (Array(converted.basis, converted_data).with_basis(array.basis))


def _transpose_from_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    return Array(
        diagonal_basis(
            (array.basis.inner[1], array.basis.inner[0]), array.basis.metadata().extra
        ),
        array.raw_data,
    )


def _transpose_from_tuple_simple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[np.dtype[np.generic], Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    return Array(
        tuple_basis((array.basis[1], array.basis[0]), array.basis.metadata().extra),
        array.raw_data.reshape(array.basis.shape).transpose(),
    )


def _transpose_simple[
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M2, M1, E], DT]:
    as_diagonal = as_diagonal_basis(array)
    if as_diagonal is not None:
        return _transpose_from_diagonal(as_diagonal)

    return _transpose_from_tuple_simple(as_tuple_basis(array))


def _transpose_from_tuple[M: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[
        TupleMetadata[M, E],
        DT,
        TupleBasis[M, E, Any],
    ],
    *,
    axes: tuple[int, ...] | None = None,
) -> Array[TupleMetadata[M, E], DT, TupleBasis[M, E, Any]]:
    # TODO: einsum based implementation would be preferred here...  # noqa: FIX002
    children = array.basis.children
    axes = tuple(range(len(children)))[::-1] if axes is None else axes
    out_basis = tuple_basis(
        tuple(children[i] for i in axes), array.basis.metadata().extra
    )

    return Array(out_basis, array.raw_data.reshape(array.basis.shape).transpose(axes))


@overload
def transpose[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[Metadata2D[M1, M2, E], DT],
    *,
    axes: None = None,
) -> Array[Metadata2D[M2, M1, E], DT]: ...


@overload
def transpose[M: BasisMetadata, Any, DT: np.dtype[np.generic]](
    array: Array[TupleMetadata[M, Any], DT],
    axes: tuple[int, ...] | None = None,
) -> Array[TupleMetadata[M, Any], DT]: ...


def transpose[DT: np.dtype[np.generic]](
    array: Array[TupleMetadata[Any, Any], DT],
    axes: tuple[int, ...] | None = None,
) -> Array[TupleMetadata[Any, Any], DT]:
    """Transpose a slate array."""
    array = as_index_basis(array)
    if axes is None and array.basis.metadata().n_dim == 2:  # noqa: PLR2004
        return _transpose_simple(cast("Array[Metadata2D[Any, Any, Any], DT]", array))

    return _transpose_from_tuple(as_tuple_basis(array), axes=axes)


def _inv_from_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        DiagonalBasis[np.generic, Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    return Array(
        diagonal_basis(
            (array.basis.inner[1].dual_basis(), array.basis.inner[0].dual_basis()),
            array.basis.metadata().extra,
        ),
        np.divide(1.0, array.raw_data),
    )


def _inv_from_tuple[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[np.dtype[np.generic], Basis[M0, Any], Basis[M1, Any], E],
    ],
) -> Array[Metadata2D[M1, M0, E], DT]:
    raw_data = array.raw_data.reshape(array.basis.shape)
    return Array(
        tuple_basis(
            (array.basis[1].dual_basis(), array.basis[0].dual_basis()),
            array.basis.metadata().extra,
        ),
        np.linalg.inv(raw_data),  # type: ignore unknown
    )


def inv[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M2, M1, E], DT]:
    """Inverse a slate array."""
    as_diagonal = as_diagonal_basis(array)
    if as_diagonal is not None:
        return _inv_from_diagonal(as_diagonal)

    return _inv_from_tuple(as_tuple_basis(array))


def dagger[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[Metadata2D[M1, M2, E], DT],
) -> Array[Metadata2D[M2, M1, E], DT]:
    """Conjugate Transpose a slate array."""
    return conjugate(transpose(array))


@overload
def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: Array[TupleMetadata[M, Any], DT],
    axes: tuple[int],
    idx: tuple[int, ...],
) -> Array[Metadata1D[M, Any], DT]: ...


@overload
def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: Array[TupleMetadata[M, Any], DT],
    axes: tuple[int, int],
    idx: tuple[int, ...],
) -> Array[Metadata2D[M, M, Any], DT]: ...


@overload
def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: Array[TupleMetadata[M, Any], DT],
    axes: tuple[int, ...],
    idx: tuple[int, ...],
) -> Array[TupleMetadata[M, Any], DT]: ...


def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: Array[TupleMetadata[M, Any], DT],
    axes: tuple[int, ...],
    idx: tuple[int, ...],
) -> Array[TupleMetadata[M, Any], DT]:
    """
    Given a slice, insert slice(None) everywhere given in axes.

    Parameters
    ----------
    slice : list[slice  |  _IntLike_co  |  None]
        slice excluding axes
    axes : tuple[_IntLike_co]
        axes to insert slice(None)

    Returns
    -------
    list[slice | _IntLike_co | None]
    """
    indexed = array[slice_ignoring_axes(idx, axes)]
    if len(axes) == 1:
        # Must be tuple_basis((basis,))
        indexed = nest(indexed)
    return transpose(indexed, axes=get_position_in_sorted(axes))
