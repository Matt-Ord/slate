from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from slate._einsum._einstein_index import (
    EinsteinIndex,
    EinsumSpecification,
    NestedEinsteinIndex,
    parse_einsum_specification,
)
from slate.array import Array
from slate.basis import (
    Basis,
    as_fundamental,
    as_tuple_basis,
    tuple_basis,
)
from slate.basis._block_diagonal import as_block_diagonal_basis
from slate.basis._diagonal import as_diagonal_basis
from slate.basis._fundamental import FundamentalBasis
from slate.basis._util import as_linear_map_basis, from_shape
from slate.basis.recast import RecastBasis
from slate.metadata import BasisMetadata, NestedLength, StackedMetadata


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
        return basis_map.resolve_single(idx)[0]
    return tuple_basis(tuple(_get_einsum_result_basis(c, basis_map) for c in idx), None)


class EinsumBasisMap:
    def __init__(self) -> None:
        self._single_map = dict[str, Basis[Any, Any]]()
        self._tuple_map = dict[tuple[str, ...], Basis[Any, Any]]()

    def resolve_single(self, idx: EinsteinIndex) -> tuple[Basis[Any, Any], int]:
        current = self._single_map.get(idx.label, None)
        if current is None:
            raise KeyError(idx.label)
        return current.dual_basis() if idx.is_dual else current, current.size

    def set_single(self, idx: EinsteinIndex, basis: Basis[Any, Any]) -> None:
        self._single_map[idx.label] = basis.dual_basis() if idx.is_dual else basis


def _resolve_einsum_basis(
    idx: NestedEinsteinIndex, basis_map: EinsumBasisMap
) -> tuple[Basis[Any, Any], NestedLength]:
    if isinstance(idx, EinsteinIndex):
        return basis_map.resolve_single(idx)

    children = list[Basis[Any, Any]]()
    lengths = list[NestedLength]()
    for i in idx:
        resolved = _resolve_einsum_basis(i, basis_map)
        children.append(resolved[0])
        lengths.append(resolved[1])

    return tuple_basis(tuple(children), None), tuple(lengths)


class EinsumBasisHints:
    def __init__(self) -> None:
        self._single_map = defaultdict[str, set[Basis[Any, Any]]](set)
        self._diag_map = defaultdict[tuple[str, ...], set[Basis[Any, Any]]](set)

    def add_hint(self, idx: EinsteinIndex, basis: Basis[Any, Any]) -> None:
        self._single_map[idx.label].add(basis.dual_basis() if idx.is_dual else basis)

    def add_diag_hint(
        self, idx: tuple[EinsteinIndex, ...], basis: Basis[Any, Any]
    ) -> None:
        diag = basis.dual_basis() if idx[0].is_dual else basis
        self._diag_map[tuple(i.label for i in idx)].add(diag)

    def resolve_basis_map(self) -> EinsumBasisMap:
        # The resolved inner basis for the einsum operation
        inner_basis_map = EinsumBasisMap()
        for idx, bases in self._single_map.items():
            bases_list = list(bases)
            if len(bases_list) == 1:
                as_linear = as_linear_map_basis(bases_list[0])
                inner_basis_map.set_single(EinsteinIndex(idx), as_linear)
            else:
                common_parent = as_fundamental(bases_list[0])
                inner_basis_map.set_single(EinsteinIndex(idx), common_parent)

        # TODO: The strategy here. Resolve the inner basis as before, then  # noqa: FIX002
        # wrap them all in a recast basis as we resolve the diagonals such
        # that we now index according to the outer recast. we also need a new
        # set of indices that are used to index the outer recast.
        # The resolution process works to resolve an outer recast basis
        # which is a simple TupleBasis as the outer index.
        # I think to do this coherently we need information about
        # the complete einsum shape, which we don't have here.

        return inner_basis_map


def _collect_einsum_basis_hints(
    basis: Basis[Any, Any], idx: NestedEinsteinIndex, hints: EinsumBasisHints
) -> None:
    if isinstance(idx, EinsteinIndex):
        hints.add_hint(idx, basis)
        return

    basis = cast("Basis[StackedMetadata[Any, Any], Any]", basis)
    as_diagonal = as_diagonal_basis(basis)
    # TODO: Add generalized diagonal so we can also support block diagonal  # noqa: FIX002
    if as_diagonal is not None and all(isinstance(i, EinsteinIndex) for i in idx):
        idx = cast("tuple[EinsteinIndex, ...]", idx)
        hints.add_diag_hint(idx, basis)

    basis = as_tuple_basis(cast("Basis[StackedMetadata[Any, Any], Any]", basis))

    for n, i in enumerate(idx):
        child = basis.children[n]
        _collect_einsum_basis_hints(child, i, hints)


type NestedData[T] = T | tuple[NestedData[T], ...]


def _flatten_nested[T](nested: NestedData[T]) -> tuple[T, ...]:
    if isinstance(nested, tuple):
        return tuple(item for subtuple in nested for item in _flatten_nested(subtuple))  # type: ignore unknown
    return (nested,)


@dataclass(frozen=True)
class BasisSpecification:
    result_index: tuple[str, ...]
    result_basis: Basis[Any, Any] | None
    part_index: tuple[tuple[str, ...], ...]
    part_basis: tuple[RecastBasis[BasisMetadata, StackedMetadata[Any, Any], Any], ...]

    def get_part_data(
        self, *arrays: Array[BasisMetadata, Any]
    ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[Any]], ...]:
        return tuple(
            arr.with_basis(b.inner).raw_data.reshape(b.outer_recast.metadata().shape)
            for (arr, b) in zip(arrays, self.part_basis, strict=True)
        )


def _reslove_basis(
    specification: EinsumSpecification, arrays: tuple[Array[Any, Any], ...]
) -> BasisSpecification:
    # For now, we don't support any optimization
    # we just do the naive einsum in the fundamental basis
    hints = EinsumBasisHints()
    for arr, part in zip(arrays, specification.parts, strict=False):
        _collect_einsum_basis_hints(arr.basis, part, hints)
    basis_map = hints.resolve_basis_map()
    part_basis = list[
        RecastBasis[
            BasisMetadata,
            StackedMetadata[Any, Any],
            Any,
            Basis[BasisMetadata, Any],
            Basis[StackedMetadata[Any, Any], Any],
        ],
    ]()
    part_index = list[tuple[str, ...]]()

    for _arr, part in zip(arrays, specification.parts, strict=False):
        basis, shape = _resolve_einsum_basis(part, basis_map)

        part_basis.append(
            RecastBasis(
                basis,
                from_shape(_flatten_nested(shape)),
                from_shape(_flatten_nested(shape)),
            )
        )

        part_index.append(tuple(i.label for i in _flatten_nested(part)))

    result_basis = (
        None
        if specification.result is None
        else _resolve_einsum_basis(specification.result, basis_map)[0]
    )
    out_shape_flat = (
        tuple[EinsteinIndex]()
        if specification.result is None
        else _flatten_nested(specification.result)
    )

    return BasisSpecification(
        result_index=tuple(i.label for i in out_shape_flat),
        result_basis=result_basis,
        part_index=tuple(part_index),
        part_basis=tuple(part_basis),
    )


def _einsum_simple[DT: np.number[Any]](
    idx: str,
    *arrays: Array[BasisMetadata, DT],
) -> Array[Any, DT, Any]:
    specification = parse_einsum_specification(idx)
    resolved = _reslove_basis(specification, arrays)

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


def _einsum_smart[DT: np.number[Any]](
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
