from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate._einsum._einstein_basis import reslove_basis
from slate._einsum._einstein_index import (
    parse_einsum_specification,
)
from slate.array import Array
from slate.basis import (
    FundamentalBasis,
    as_block_diagonal_basis,
    as_diagonal_basis,
    as_linear_map_basis,
    as_tuple_basis,
    tuple_basis,
)

if TYPE_CHECKING:
    from slate.metadata import BasisMetadata


def _einsum_numpy[DT: np.dtype[np.number[Any]]](
    idx: str,
    *arrays: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    return cast(
        "np.ndarray[Any, Any]",
        np.einsum(idx, *arrays),  # type: ignore unknown
    )


def _einsum_simple[DT: np.dtype[np.number[Any]]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    specification = parse_einsum_specification(idx)
    resolved = reslove_basis(specification, arrays)

    raw_arrays = resolved.get_part_data(*arrays)
    raw_idx = resolved.part_index

    final_idx = ",".join("".join(i) for i in raw_idx) + "->"
    result_basis = resolved.result_basis
    if result_basis is None:
        return Array(
            FundamentalBasis.from_size(1),
            _einsum_numpy(final_idx, *raw_arrays),
        )

    final_idx += "".join(resolved.result_index)

    return Array(result_basis, _einsum_numpy(final_idx, *raw_arrays))


def _einsum_smart[DT: np.dtype[np.number[Any]]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    assert idx == "(i j'),(j k)->(i k)"
    as_linear = as_linear_map_basis(arrays[1].basis)
    as_tuple_0 = as_tuple_basis(arrays[0].basis)
    as_diagonal = as_diagonal_basis(as_linear)
    if as_diagonal is not None:
        out_basis = tuple_basis((as_tuple_0[0], as_diagonal.inner[1]))
        array_0 = arrays[0].with_basis(as_tuple_0)
        array_1 = arrays[1].with_basis(as_diagonal)

        return Array(
            out_basis,
            _einsum_numpy(
                "ij,j->ij",
                array_0.raw_data.reshape(array_0.basis.shape),
                array_1.raw_data,
            ),
        )

    as_block_diagonal = as_block_diagonal_basis(as_linear)
    if as_block_diagonal is not None:
        out_basis = tuple_basis((as_tuple_0[0], as_block_diagonal.inner[1]))
        array_0 = arrays[0].with_basis(as_tuple_0)
        array_1 = arrays[1].with_basis(as_block_diagonal)

        array_1_raw = array_1.raw_data.reshape(
            as_block_diagonal.n_repeats, *as_block_diagonal.block_shape
        )
        array_0_raw = array_0.raw_data.reshape(
            as_tuple_0[0].size,
            as_block_diagonal.n_repeats,
            as_block_diagonal.block_shape[0],
        )

        return Array(
            out_basis,
            # Diagonal on index j but not on index (k,l)
            _einsum_numpy("ijk,jkl->ijl", array_0_raw, array_1_raw),
        )
    return _einsum_simple("(i j'),(j k)->(i k)", *arrays)


def einsum[DT: np.dtype[np.number[Any]]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    # Eventually we will want to support fast einsum for an arbitrary index-like
    # matrix. For now though, we just support the simple case that is
    # required for ExplicitBasis.
    if idx == "(i j'),(j k)->(i k)":
        _einsum_smart(idx, *arrays)

    return _einsum_simple(idx, *arrays)
