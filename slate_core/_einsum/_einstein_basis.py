from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from slate_core._einsum._einstein_index import (
    EinsteinIndex,
    EinsumSpecification,
    NestedEinsteinIndex,
)
from slate_core.basis import (
    Basis,
    RecastBasis,
    as_fundamental,
    as_linear_map,
    as_tuple,
    from_shape,
)
from slate_core.basis import (
    as_diagonal as as_diagonal_basis,
)
from slate_core.basis._tuple import TupleBasis, TupleBasisLike, is_tuple_basis_like
from slate_core.metadata import NestedLength

if TYPE_CHECKING:
    import numpy as np

    from slate_core.array import Array
    from slate_core.basis._basis import Ctype


def _get_einsum_result_basis(
    idx: NestedEinsteinIndex, basis_map: EinsumBasisMap
) -> Basis:
    if isinstance(idx, EinsteinIndex):
        return basis_map.resolve_single(idx)[0]
    return TupleBasis(tuple(_get_einsum_result_basis(c, basis_map) for c in idx), None)


class EinsumBasisMap:
    def __init__(self) -> None:
        self._single_map = dict[str, Basis]()
        self._tuple_map = dict[tuple[str, ...], Basis]()

    def resolve_single(self, idx: EinsteinIndex) -> tuple[Basis, int]:
        current = self._single_map.get(idx.label, None)
        if current is None:
            raise KeyError(idx.label)
        return current.dual_basis() if idx.is_dual else current, current.size

    def set_single(self, idx: EinsteinIndex, basis: Basis) -> None:
        self._single_map[idx.label] = basis.dual_basis() if idx.is_dual else basis


def _resolve_einsum_basis(
    idx: NestedEinsteinIndex, basis_map: EinsumBasisMap
) -> tuple[Basis, NestedLength]:
    if isinstance(idx, EinsteinIndex):
        return basis_map.resolve_single(idx)

    children = list[Basis]()
    lengths = list[NestedLength]()
    for i in idx:
        resolved = _resolve_einsum_basis(i, basis_map)
        children.append(resolved[0])
        lengths.append(resolved[1])

    return TupleBasis(tuple(children), None), tuple(lengths)


class EinsumBasisHints:
    def __init__(self) -> None:
        self._single_map = defaultdict[str, list[Basis]](list)
        self._diag_map = defaultdict[tuple[str, ...], list[Basis]](list)

    def add_hint(self, idx: EinsteinIndex, basis: Basis) -> None:
        self._single_map[idx.label].append(basis.dual_basis() if idx.is_dual else basis)

    def add_diag_hint(self, idx: tuple[EinsteinIndex, ...], basis: Basis) -> None:
        diag = basis.dual_basis() if idx[0].is_dual else basis
        self._diag_map[tuple(i.label for i in idx)].append(diag)

    def resolve_basis_map(self) -> EinsumBasisMap:
        # The resolved inner basis for the einsum operation
        inner_basis_map = EinsumBasisMap()
        for idx, bases in self._single_map.items():
            # Get unique bases, but preserve insertion order
            bases_list = list(dict.fromkeys(bases))
            if len(bases_list) == 1:
                as_linear = as_linear_map(bases_list[0])
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
    basis: Basis, idx: NestedEinsteinIndex, hints: EinsumBasisHints
) -> None:
    if isinstance(idx, EinsteinIndex):
        hints.add_hint(idx, basis)
        return

    as_diagonal = (
        as_diagonal_basis(basis) if is_tuple_basis_like(basis, n_dim=2) else None
    )
    # TODO: Add generalized diagonal so we can also support block diagonal  # noqa: FIX002
    if as_diagonal is not None and all(isinstance(i, EinsteinIndex) for i in idx):
        idx = cast("tuple[EinsteinIndex, ...]", idx)
        hints.add_diag_hint(idx, basis)
    assert is_tuple_basis_like(basis)
    basis_as_tuple = as_tuple(basis)
    for n, i in enumerate(idx):
        child = basis_as_tuple.children[n]
        _collect_einsum_basis_hints(child, i, hints)


type NestedData[T] = T | tuple[NestedData[T], ...]


def _flatten_nested[T](nested: NestedData[T]) -> tuple[T, ...]:
    if isinstance(nested, tuple):
        return tuple(item for subtuple in nested for item in _flatten_nested(subtuple))  # type: ignore unknown
    return (nested,)


@dataclass(frozen=True)
class BasisSpecification:
    result_index: tuple[str, ...]
    result_basis: Basis | None
    part_index: tuple[tuple[str, ...], ...]
    part_basis: tuple[RecastBasis[Basis, Basis, TupleBasisLike], ...]

    def get_part_data(
        self, *arrays: Array[Basis, Any]
    ) -> tuple[np.ndarray[tuple[int, ...], np.dtype[np.generic]], ...]:
        return tuple(
            arr.with_basis(b.inner).raw_data.reshape(b.outer_recast.metadata().shape)
            for (arr, b) in zip(arrays, self.part_basis, strict=True)
        )


def reslove_basis(
    specification: EinsumSpecification, arrays: tuple[Array[Any, Any], ...]
) -> BasisSpecification:
    # For now, we don't support any optimization
    # we just do the naive einsum in the fundamental basis
    hints = EinsumBasisHints()
    # Sort by the size of the array to minimize the transformation cost
    parts_iter = sorted(
        zip(arrays, specification.parts, strict=False),
        key=lambda x: x[0].basis.fundamental_size,
        reverse=True,
    )
    for arr, part in parts_iter:
        _collect_einsum_basis_hints(arr.basis, part, hints)
    basis_map = hints.resolve_basis_map()
    part_basis = list[RecastBasis[Basis, Basis, TupleBasisLike],]()
    part_index = list[tuple[str, ...]]()

    for _arr, part in zip(arrays, specification.parts, strict=False):
        basis, shape = _resolve_einsum_basis(part, basis_map)

        part_basis.append(
            RecastBasis(
                cast("Basis[Any, Ctype[np.generic]]", basis),
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
