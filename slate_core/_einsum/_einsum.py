from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate_core._einsum._einstein_basis import InvalidBasisError, reslove_basis
from slate_core._einsum._einstein_index import (
    parse_einsum_specification,
)
from slate_core.array._array import Array
from slate_core.basis import (
    FundamentalBasis,
    as_linear_map,
)
from slate_core.basis import (
    as_block_diagonal as as_block_diagonal_basis,
)
from slate_core.basis import (
    as_diagonal as as_diagonal_basis,
)
from slate_core.basis._tuple import TupleBasis, as_tuple, is_tuple_basis_like

if TYPE_CHECKING:
    from slate_core.basis._basis import Basis, Ctype


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
        return Array(
            FundamentalBasis.from_size(1),
            _einsum_numpy(final_idx, *raw_arrays),
        )

    final_idx += "".join(resolved.result_index)

    return Array(result_basis, _einsum_numpy(final_idx, *raw_arrays))


def _einsum_smart[DT: np.dtype[np.number]](
    idx: str,
    *arrays: Array[Basis, DT],
) -> Array[Any, DT]:
    assert idx == "(i j'),(j k)->(i k)"
    assert is_tuple_basis_like(arrays[0].basis)
    as_linear = as_linear_map(arrays[1].basis)
    as_tuple_0 = as_tuple(arrays[0].basis)
    as_diagonal = (
        as_diagonal_basis(as_linear)
        if is_tuple_basis_like(as_linear, n_dim=2)
        else None
    )
    if as_diagonal is not None:
        out_basis = TupleBasis(
            (as_tuple_0.children[0], as_diagonal.inner.children[1])
        ).resolve_ctype()
        array_0 = arrays[0].with_basis(
            cast("Basis[Any, Ctype[np.generic]]", as_tuple_0)
        )
        array_1 = arrays[1].with_basis(
            cast("Basis[Any, Ctype[np.generic]]", as_diagonal)
        )

        return Array(
            out_basis,
            _einsum_numpy(
                "ij,j->ij",
                array_0.raw_data.reshape(as_tuple_0.shape),
                array_1.raw_data,
            ),
        )

    as_block_diagonal = (
        as_block_diagonal_basis(as_linear)
        if is_tuple_basis_like(as_linear, n_dim=2)
        else None
    )
    if as_block_diagonal is not None:
        out_basis = TupleBasis(
            (as_tuple_0.children[0], as_block_diagonal.inner.children[1])
        )
        array_0 = arrays[0].with_basis(
            cast("Basis[Any, Ctype[np.generic]]", as_tuple_0)
        )
        array_1 = arrays[1].with_basis(
            cast("Basis[Any, Ctype[np.generic]]", as_block_diagonal)
        )

        array_1_raw = array_1.raw_data.reshape(
            as_block_diagonal.n_repeats, *as_block_diagonal.block_shape
        )
        array_0_raw = array_0.raw_data.reshape(
            as_tuple_0.children[0].size,
            as_block_diagonal.n_repeats,
            as_block_diagonal.block_shape[0],
        )

        return Array(
            out_basis,
            # Diagonal on index j but not on index (k,l)
            _einsum_numpy("ijk,jkl->ijl", array_0_raw, array_1_raw),
        )
    return _einsum_simple("(i j'),(j k)->(i k)", *arrays)


def einsum[DT: np.dtype[np.number]](
    idx: str,
    *arrays: Array[Basis, DT],
) -> Array[Any, DT]:
    # Eventually we will want to support fast einsum for an arbitrary index-like
    # matrix. For now though, we just support the simple case that is
    # required for ExplicitBasis.
    # Ideally this support should also be for a more general contracted basis
    if idx == "(i j'),(j k)->(i k)":
        return _einsum_smart(idx, *arrays)
    try:
        return _einsum_simple(idx, *arrays)
    except InvalidBasisError as e:
        raise e from None
