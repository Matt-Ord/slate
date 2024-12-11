from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate.array import Array
from slate.basis import (
    Basis,
    as_fundamental,
    as_index_basis,
    as_tuple_basis,
    tuple_basis,
)
from slate.linalg._einstein_index import (
    EinsteinIndex,
    NestedEinsteinIndex,
    parse_einsum_specification,
)
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


def einsum[DT: np.number[Any]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    specification = parse_einsum_specification(idx)
    # For now, we don't support any optimization
    # we just do the naive einsum in the fundamental basis
    hints = EinsumBasisHints()
    for arr, part in zip(arrays, specification.parts):
        _collect_einsum_basis_hints(arr.basis, part, hints)
    basis_map = hints.resolve_basis_map()
    raw_arrays = list[np.ndarray[Any, Any]]()
    raw_idx = list[str]()
    for arr, part in zip(arrays, specification.parts):
        basis, shape = _resolve_einsum_basis(part, basis_map)
        converted = arr.with_basis(basis)
        raw_arrays.append(converted.raw_data.reshape(_flatten_nested(shape)))

        flat_idx = _flatten_nested(part)
        raw_idx.append("".join(i.label for i in flat_idx))

    out_basis, _shape = _resolve_einsum_basis(specification.result, basis_map)
    out_shape_flat = _flatten_nested(specification.result)

    final_idx = ",".join(raw_idx) + "->" + "".join(i.label for i in out_shape_flat)
    return Array(out_basis, _einsum_numpy(final_idx, *raw_arrays))
