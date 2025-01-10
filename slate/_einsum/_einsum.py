from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate._einsum._einstein_index import (
    EinsteinIndex,
    NestedEinsteinIndex,
    parse_einsum_specification,
)
from slate.array import Array
from slate.basis import (
    Basis,
    as_fundamental,
    as_index_basis,
    as_tuple_basis,
    tuple_basis,
)
from slate.basis._block_diagonal import as_block_diagonal_basis
from slate.basis._diagonal import as_diagonal_basis
from slate.basis._fundamental import FundamentalBasis
from slate.metadata import NestedLength, StackedMetadata

if TYPE_CHECKING:
    from slate.metadata import BasisMetadata


def _einsum_numpy[DT: np.number[Any]](
    idx: str,
    *arrays: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    return cast(
        "np.ndarray[Any, Any]",
        np.einsum(idx, *arrays),  # type: ignore unknown
    )


def _get_einsum_result_basis(
    idx: NestedEinsteinIndex, basis_map: EinsumBasisMap
) -> Basis[Any, Any]:
    if isinstance(idx, EinsteinIndex):
        return basis_map[idx]
    return tuple_basis(tuple(_get_einsum_result_basis(c, basis_map) for c in idx), None)


class EinsumBasisMap:
    def __init__(self) -> None:
        self._map = dict[str, Basis[Any, Any]]()

    def __getitem__(self, idx: EinsteinIndex) -> Basis[Any, Any]:
        current = self._map.get(idx.label, None)
        if current is None:
            raise KeyError(idx.label)
        return current.dual_basis() if idx.is_dual else current

    def __setitem__(self, idx: EinsteinIndex, basis: Basis[Any, Any]) -> None:
        self._map[idx.label] = basis.dual_basis() if idx.is_dual else basis


def _resolve_einsum_basis(
    idx: NestedEinsteinIndex, basis_map: EinsumBasisMap
) -> tuple[Basis[Any, Any], NestedLength]:
    if isinstance(idx, EinsteinIndex):
        resolved = basis_map[idx]
        return resolved, resolved.size

    children = list[Basis[Any, Any]]()
    lengths = list[NestedLength]()
    for i in idx:
        resolved = _resolve_einsum_basis(i, basis_map)
        children.append(resolved[0])
        lengths.append(resolved[1])

    return tuple_basis(tuple(children), None), tuple(lengths)


class EinsumBasisHints:
    def __init__(self) -> None:
        self._map = defaultdict[str, set[Basis[Any, Any]]](set)

    def add_hint(self, idx: EinsteinIndex, basis: Basis[Any, Any]) -> None:
        self._map[idx.label].add(basis.dual_basis() if idx.is_dual else basis)

    def resolve_basis_map(self) -> EinsumBasisMap:
        basis_map = EinsumBasisMap()
        for idx, bases in self._map.items():
            bases_list = list(bases)
            if len(bases_list) == 1:
                basis_map[EinsteinIndex(idx)] = as_index_basis(bases_list[0])
            else:
                basis_map[EinsteinIndex(idx)] = as_fundamental(bases_list[0])
        return basis_map


def _collect_einsum_basis_hints(
    basis: Basis[Any, Any], idx: NestedEinsteinIndex, hints: EinsumBasisHints
) -> None:
    if isinstance(idx, EinsteinIndex):
        hints.add_hint(idx, basis)
        return

    basis = as_tuple_basis(cast("Basis[StackedMetadata[Any, Any], Any]", basis))

    for n, i in enumerate(idx):
        child = basis.children[n]
        _collect_einsum_basis_hints(child, i, hints)


type NestedData[T] = T | tuple[NestedData[T], ...]


def _flatten_nested[T](nested: NestedData[T]) -> tuple[T, ...]:
    if isinstance(nested, tuple):
        return tuple(item for subtuple in nested for item in _flatten_nested(subtuple))  # type: ignore unknown
    return (nested,)


def _einsum_simple[DT: np.number[Any]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    specification = parse_einsum_specification(idx)
    # For now, we don't support any optimization
    # we just do the naive einsum in the fundamental basis
    hints = EinsumBasisHints()
    for arr, part in zip(arrays, specification.parts, strict=False):
        _collect_einsum_basis_hints(arr.basis, part, hints)
    basis_map = hints.resolve_basis_map()
    raw_arrays = list[np.ndarray[Any, Any]]()
    raw_idx = list[str]()
    for arr, part in zip(arrays, specification.parts, strict=False):
        basis, shape = _resolve_einsum_basis(part, basis_map)
        converted = arr.with_basis(basis)
        raw_arrays.append(converted.raw_data.reshape(_flatten_nested(shape)))

        flat_idx = _flatten_nested(part)
        raw_idx.append("".join(i.label for i in flat_idx))

    final_idx = ",".join(raw_idx) + "->"

    if specification.result is None:
        return Array(
            FundamentalBasis.from_size(1),
            _einsum_numpy(final_idx, *raw_arrays),
        )

    out_basis, _shape = _resolve_einsum_basis(specification.result, basis_map)
    out_shape_flat = _flatten_nested(specification.result)

    final_idx += "".join(i.label for i in out_shape_flat)
    return Array(out_basis, _einsum_numpy(final_idx, *raw_arrays))


def _einsum_smart[DT: np.number[Any]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    assert idx == "(i j'),(j k)->(i k)"
    as_index = as_index_basis(arrays[1].basis)
    as_tuple_0 = as_tuple_basis(arrays[0].basis)
    as_diagonal = as_diagonal_basis(as_index)
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

    as_block_diagonal = as_block_diagonal_basis(as_index)
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


def einsum[DT: np.number[Any]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    # Eventually we will want to support fast einsum for an arbitrary index-like
    # matrix. For now though, we just support the simple case that is
    # required for ExplicitBasis.
    if idx == "(i j'),(j k)->(i k)":
        _einsum_smart(idx, *arrays)

    return _einsum_simple(idx, *arrays)
