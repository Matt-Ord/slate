from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate._einsum._einstein_basis import reslove_basis
from slate._einsum._einstein_index import (
    parse_einsum_specification,
)
from slate.array._array import build
from slate.basis import (
    FundamentalBasis,
    as_block_diagonal_basis,
    as_diagonal_basis,
    as_linear_map_basis,
)
from slate.basis._tuple import TupleBasis, as_tuple_basis, is_tuple_basis_like

if TYPE_CHECKING:
    from slate.array import Array
    from slate.basis._basis import Basis, ctype


def _einsum_numpy[DT: np.dtype[np.number]](
    idx: str,
    *arrays: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    return cast("np.ndarray[Any, Any]", np.einsum(idx, *arrays))  # type: ignore unknown


def _einsum_simple[DT: np.dtype[np.number]](
    idx: str,
    *arrays: Array[Basis, DT],
) -> Array[Any, DT]:
    specification = parse_einsum_specification(idx)
    resolved = reslove_basis(specification, arrays)

    raw_arrays = resolved.get_part_data(*arrays)
    raw_idx = resolved.part_index

    final_idx = ",".join("".join(i) for i in raw_idx) + "->"
    result_basis = resolved.result_basis
    if result_basis is None:
        return build(
            FundamentalBasis.from_size(1),
            _einsum_numpy(final_idx, *raw_arrays),
        ).ok()

    final_idx += "".join(resolved.result_index)

    return build(result_basis, _einsum_numpy(final_idx, *raw_arrays)).ok()


def _einsum_smart[DT: np.dtype[np.number]](
    idx: str,
    *arrays: Array[Basis, DT],
) -> Array[Any, DT]:
    assert idx == "(i j'),(j k)->(i k)"
    assert is_tuple_basis_like(arrays[0].basis)
    as_linear = as_linear_map_basis(arrays[1].basis)
    as_tuple_0 = as_tuple_basis(arrays[0].basis)
    as_diagonal = (
        as_diagonal_basis(as_linear)
        if is_tuple_basis_like(as_linear, n_dim=2)
        else None
    )
    if as_diagonal is not None:
        out_basis = TupleBasis(
            (as_tuple_0.children[0], as_diagonal.inner.children[1])
        ).upcast()
        array_0 = (
            arrays[0].with_basis(cast("Basis[Any, ctype[np.generic]]", as_tuple_0)).ok()
        )
        array_1 = (
            arrays[1]
            .with_basis(cast("Basis[Any, ctype[np.generic]]", as_diagonal))
            .ok()
        )

        return build(
            out_basis,
            _einsum_numpy(
                "ij,j->ij",
                array_0.raw_data.reshape(as_tuple_0.shape),
                array_1.raw_data,
            ),
        ).ok()

    as_block_diagonal = (
        as_block_diagonal_basis(as_linear)
        if is_tuple_basis_like(as_linear, n_dim=2)
        else None
    )
    if as_block_diagonal is not None:
        out_basis = TupleBasis(
            (as_tuple_0.children[0], as_block_diagonal.inner.children[1])
        )
        array_0 = (
            arrays[0].with_basis(cast("Basis[Any, ctype[np.generic]]", as_tuple_0)).ok()
        )
        array_1 = (
            arrays[1]
            .with_basis(cast("Basis[Any, ctype[np.generic]]", as_block_diagonal))
            .ok()
        )

        array_1_raw = array_1.raw_data.reshape(
            as_block_diagonal.n_repeats, *as_block_diagonal.block_shape
        )
        array_0_raw = array_0.raw_data.reshape(
            as_tuple_0.children[0].size,
            as_block_diagonal.n_repeats,
            as_block_diagonal.block_shape[0],
        )

        return build(
            out_basis,
            # Diagonal on index j but not on index (k,l)
            _einsum_numpy("ijk,jkl->ijl", array_0_raw, array_1_raw),
        ).ok()
    return _einsum_simple("(i j'),(j k)->(i k)", *arrays)


def einsum[DT: np.dtype[np.number]](
    idx: str,
    *arrays: Array[Basis, DT],
) -> Array[Any, DT]:
    # Eventually we will want to support fast einsum for an arbitrary index-like
    # matrix. For now though, we just support the simple case that is
    # required for ExplicitBasis.
    if idx == "(i j'),(j k)->(i k)":
        return _einsum_smart(idx, *arrays)

    return _einsum_simple(idx, *arrays)
