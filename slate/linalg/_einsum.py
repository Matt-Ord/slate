from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate.array import Array
from slate.basis._diagonal import as_diagonal_basis
from slate.basis._tuple import as_tuple_basis, tuple_basis, with_child

if TYPE_CHECKING:
    from slate.metadata import BasisMetadata


def _einsum_numpy[DT: np.number[Any]](
    idx: str,
    array_1: np.ndarray[Any, Any],
    array_2: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    return cast(
        np.ndarray[Any, Any],
        np.einsum(idx, array_1, array_2),  # type: ignore unknown
    )


def _einsum_1[DT: np.number[Any]](
    array_1: Array[BasisMetadata, DT],
    array_2: Array[BasisMetadata, DT],
) -> Array[BasisMetadata, DT]:
    array_1_diag = as_diagonal_basis(array_1.basis)
    array_1_tuple = as_tuple_basis(array_1.basis)

    array_2_tuple = as_tuple_basis(array_2.basis)
    array_2_converted_basis = with_child(
        array_2_tuple, array_1_tuple[1].dual_basis(), 0
    )
    array_2_converted = array_2.with_basis(array_2_converted_basis)

    final_basis = with_child(array_2_tuple, array_1_tuple[0], 0)
    if array_1_diag is not None:
        array_1_converted = array_1.with_basis(array_1_diag)
        data = _einsum_numpy(
            "i,i...->i...",
            array_1_converted.raw_data,
            array_2_converted.raw_data.reshape(array_2_converted.basis.shape),
        )
        return Array(final_basis, data)
    array_1_converted = array_1.with_basis(array_1_tuple)
    data = _einsum_numpy(
        "ij,i...->j...",
        array_1_converted.raw_data.reshape(array_1_converted.basis.shape),
        array_2_converted.raw_data.reshape(array_2_converted.basis.shape),
    )
    return Array(final_basis, data)


def _einsum_2[DT: np.number[Any]](
    array_1: Array[BasisMetadata, DT],
    array_2: Array[BasisMetadata, DT],
) -> Array[BasisMetadata, DT]:
    # (m (i k')),(k j) -> m (i j)

    array_1_tuple = as_tuple_basis(array_1.basis)
    array_2_tuple = as_tuple_basis(array_2.basis)
    k_basis = array_2_tuple[0]
    m_basis = array_1_tuple[0]
    i_basis = as_tuple_basis(array_1_tuple[1])[0]
    j_basis = array_2_tuple[1]

    array_1_basis = tuple_basis((m_basis, tuple_basis((i_basis, k_basis.dual_basis()))))
    array_2_basis = tuple_basis((k_basis, j_basis))
    out_basis = tuple_basis((m_basis, tuple_basis((i_basis, j_basis))))

    array_1_converted = array_1.with_basis(array_1_basis)
    array_2_converted = array_2.with_basis(array_2_basis)

    data = _einsum_numpy(
        "mik,kj->mij",
        array_1_converted.raw_data.reshape(m_basis.size, i_basis.size, k_basis.size),
        array_2_converted.raw_data.reshape(k_basis.size, j_basis.size),
    )
    return Array(out_basis, data)


def _einsum_3[DT: np.number[Any]](
    array_1: Array[BasisMetadata, DT],
    array_2: Array[BasisMetadata, DT],
) -> Array[BasisMetadata, DT]:
    # (i k'),(m (k j)) -> m (i j)

    array_1_tuple = as_tuple_basis(array_1.basis)
    array_2_tuple = as_tuple_basis(array_2.basis)
    k_basis = array_1_tuple[1].dual_basis()
    m_basis = array_2_tuple[0]
    i_basis = array_1_tuple[0]
    j_basis = as_tuple_basis(array_2_tuple[1])[1]

    array_1_basis = tuple_basis((i_basis, k_basis.dual_basis()))
    array_2_basis = tuple_basis((m_basis, tuple_basis((k_basis, j_basis))))
    out_basis = tuple_basis((m_basis, tuple_basis((i_basis, j_basis))))

    array_1_converted = array_1.with_basis(array_1_basis)
    array_2_converted = array_2.with_basis(array_2_basis)

    data = _einsum_numpy(
        "ik,mkj->mij",
        array_1_converted.raw_data.reshape(i_basis.size, k_basis.size),
        array_2_converted.raw_data.reshape(m_basis.size, k_basis.size, j_basis.size),
    )
    return Array(out_basis, data)


def einsum[DT: np.number[Any]](
    idx: str,
    array_1: Array[BasisMetadata, DT],
    array_2: Array[BasisMetadata, DT],
) -> Array[BasisMetadata, DT]:
    if idx == "(i j),i...->j...":
        return _einsum_1(array_1, array_2)
    if idx == "(m (i k')),(k j) -> (m (i j))":
        return _einsum_2(array_1, array_2)
    if idx == "(i k'),(m (k j)) -> (m (i j))":
        return _einsum_3(array_1, array_2)
    msg = "Not implemented yet."
    raise NotImplementedError(msg)
