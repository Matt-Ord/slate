from __future__ import annotations

from typing import TYPE_CHECKING, Any, Never, overload

import numpy as np

from slate_core.array._array import Array
from slate_core.array._conversion import (
    as_diagonal_basis,
    as_index_basis,
    as_tuple_basis,
    nest,
)
from slate_core.array._misc import is_tuple_basis_like
from slate_core.basis._basis import Ctype
from slate_core.basis._diagonal import DiagonalBasis
from slate_core.basis._tuple import TupleBasis, TupleBasisLike
from slate_core.metadata import BasisMetadata
from slate_core.util._index import get_position_in_sorted, slice_ignoring_axes

if TYPE_CHECKING:
    from slate_core.basis import (
        Basis,
    )


def conjugate[B: Basis, DT: np.dtype[np.generic]](
    array: Array[B, DT],
) -> Array[B, DT]:
    """Conjugate a slate array."""
    converted = as_index_basis(array)
    converted_data = np.conj(converted.raw_data).astype(converted.dtype)
    converted.raw_data = converted_data
    # Since b has the same dtype and metadata as the original basis
    # it is safe to use it in a conversion.
    # Unfortunately is not possible to express this invariant in the type system.
    return converted.with_basis(array.basis)


def _transpose_from_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
](
    array: Array[
        DiagonalBasis[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E], CT],
        DT,
    ],
) -> Array[TupleBasisLike[tuple[M1, M0], E, CT], DT]:
    return Array(
        DiagonalBasis(
            TupleBasis(
                (array.basis.inner.children[1], array.basis.inner.children[0]),
                array.basis.metadata().extra,
            ).resolve_ctype()
        )
        .resolve_ctype()
        .upcast(),
        array.raw_data,
    )


def _transpose_from_tuple_simple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
](
    array: Array[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E, CT], DT],
) -> Array[TupleBasisLike[tuple[M1, M0], E, CT], DT]:
    return Array(
        TupleBasis(
            (array.basis.children[1], array.basis.children[0]),
            array.basis.metadata().extra,
        )
        .resolve_ctype()
        .upcast(),
        array.raw_data.reshape(array.basis.shape).transpose(),
    )


def _transpose_simple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
](
    array: Array[TupleBasisLike[tuple[M0, M1], E, CT], DT],
) -> Array[TupleBasisLike[tuple[M1, M0], E, CT], DT]:
    as_diagonal = as_diagonal_basis(array)
    if as_diagonal is not None:
        return _transpose_from_diagonal(as_diagonal)
    return _transpose_from_tuple_simple(as_tuple_basis(array))


def _transpose_from_tuple[
    M: BasisMetadata,
    E,
    CT: Ctype[Never],
    DT: np.dtype[np.generic],
](
    array: Array[TupleBasis[tuple[Basis[M, CT], ...], E, CT], DT],
    axes: tuple[int, ...] | None = None,
) -> Array[TupleBasis[tuple[Basis[M, CT], ...], E, CT], DT]:
    # TODO: einsum based implementation would be preferred here...  # noqa: FIX002
    children = array.basis.children
    axes = tuple(range(len(children)))[::-1] if axes is None else axes
    out_basis = TupleBasis(
        tuple(children[i] for i in axes), array.basis.metadata().extra
    ).resolve_ctype()
    # SAFE, since if the original basis supports the data, the new basis will too.
    return Array(out_basis, array.raw_data.reshape(array.basis.shape).transpose(axes))


@overload
def transpose[
    M1: BasisMetadata,
    M2: BasisMetadata,
    CT: Ctype[Never],
    E,
    DT: np.dtype[np.generic],
](
    array: Array[TupleBasisLike[tuple[M1, M2], E, CT], DT],
    *,
    axes: None = None,
) -> Array[TupleBasisLike[tuple[M2, M1], E, CT], DT]: ...


@overload
def transpose[M: BasisMetadata, Any, CT: Ctype[Never], DT: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[M, ...], Any, CT], DT],
    *,
    axes: tuple[int, ...] | None = None,
) -> Array[TupleBasisLike[tuple[M, ...], Any, CT], DT]: ...


def transpose[DT: np.dtype[np.generic], CT: Ctype[Never]](
    array: Array[TupleBasisLike[tuple[BasisMetadata, ...], Any, CT], DT],
    *,
    axes: tuple[int, ...] | None = None,
) -> Array[TupleBasisLike[Any, Any, CT], DT]:
    """Transpose a slate array."""
    array = as_index_basis(array)
    if axes is None and is_tuple_basis_like(array, n_dim=2):
        return _transpose_simple(array)
    return _transpose_from_tuple(as_tuple_basis(array), axes)


def _inv_from_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.generic],
](
    array: Array[DiagonalBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E]], DT],
) -> Array[TupleBasisLike[tuple[M1, M0], E], DT]:
    return Array(
        DiagonalBasis(
            TupleBasis(
                (
                    array.basis.inner.children[1].dual_basis(),
                    array.basis.inner.children[0].dual_basis(),
                ),
                array.basis.metadata().extra,
            ).resolve_ctype()
        ),
        np.divide(1.0, array.raw_data),
    ).ok()  # type: ignore cant infer dtype


def _inv_from_tuple[M0: BasisMetadata, M1: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[TupleBasis[tuple[Basis[M0], Basis[M1]], E], DT],
) -> Array[TupleBasisLike[tuple[M1, M0], E], DT]:
    raw_data = array.raw_data.reshape(array.basis.shape)
    return Array(
        TupleBasis(
            (
                array.basis.children[1].dual_basis(),
                array.basis.children[0].dual_basis(),
            ),
            array.basis.metadata().extra,
        ),
        np.linalg.inv(raw_data),  # type: ignore unknown
    )


def inv[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[M1, M2], E], DT],
) -> Array[TupleBasisLike[tuple[M2, M1], E], DT]:
    """Inverse a slate array."""
    as_diagonal = as_diagonal_basis(array)
    if as_diagonal is not None:
        return _inv_from_diagonal(as_diagonal)

    return _inv_from_tuple(as_tuple_basis(array))


def dagger[M1: BasisMetadata, M2: BasisMetadata, E, DT: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[M1, M2], E], DT],
) -> Array[TupleBasisLike[tuple[M2, M1], E], DT]:
    """Conjugate Transpose a slate array."""
    return conjugate(transpose(array))


@overload
def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic], CT: Ctype[Never]](
    array: Array[TupleBasisLike[tuple[M, ...], Any, CT], DT],
    axes: tuple[int],
    idx: tuple[int, ...],
) -> Array[TupleBasisLike[tuple[M], Any, CT], DT]: ...


@overload
def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic], CT: Ctype[Never]](
    array: Array[TupleBasisLike[tuple[M, ...], Any, CT], DT],
    axes: tuple[int, int],
    idx: tuple[int, ...],
) -> Array[TupleBasisLike[tuple[M, M], Any, CT], DT]: ...


@overload
def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic], CT: Ctype[Never]](
    array: Array[TupleBasisLike[tuple[M, ...], Any, CT], DT],
    axes: tuple[int, ...],
    idx: tuple[int, ...],
) -> Array[TupleBasisLike[tuple[M, ...], Any, CT], DT]: ...


def get_data_in_axes[M: BasisMetadata, DT: np.dtype[np.generic]](
    array: Array[TupleBasisLike[tuple[M, ...], Any], DT],
    axes: tuple[int, ...],
    idx: tuple[int, ...],
) -> Array[TupleBasisLike[tuple[BasisMetadata, ...], Any], DT]:
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
        # Must be TupleBasis((basis,))
        indexed = nest(indexed)
    assert is_tuple_basis_like(indexed)
    return transpose(indexed, axes=get_position_in_sorted(axes))
