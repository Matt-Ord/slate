from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Self, cast

import numpy as np

from slate.array import Array
from slate.basis._diagonal import as_diagonal_basis
from slate.basis._tuple import as_tuple_basis, tuple_basis, with_child

if TYPE_CHECKING:
    from slate.metadata import BasisMetadata


def _einsum_numpy[DT: np.number[Any]](
    idx: str,
    *arrays: np.ndarray[Any, Any],
) -> np.ndarray[Any, Any]:
    return cast(
        np.ndarray[Any, Any],
        np.einsum(idx, *arrays),  # type: ignore unknown
    )


@dataclass(frozen=True)
class EinsteinIndex:
    label: str
    is_dual: bool


class EinsumBasisMap:
    def __init__(self):
        self._map = dict[str, Basis[Any, Any]]()

    def __getitem__(self, idx: EinsteinIndex) -> Basis[Any, Any]:
        current = self._map.get(idx.label, None)
        if current is None:
            raise KeyError(idx.label)
        return current.dual_basis() if idx.is_dual else current


type NestedEinsteinIndex = EinsteinIndex | tuple[NestedEinsteinIndex, ...]


def _remap_einstein_index(
    idx: NestedEinsteinIndex, mapped: dict[str, str]
) -> NestedEinsteinIndex:
    if isinstance(idx, EinsteinIndex):
        return EinsteinIndex(mapped.get(idx.label, idx.label), idx.is_dual)
    return tuple(_remap_einstein_index(c, mapped) for c in idx)


# To get an optimal solve of einsum we need to split it into several passes
# - Pass 1: Identify any diagonal basis and simplify the EinsteinIndex accordingly
# - Pass 2: Identify the smallest possible basis for each index in the einsum
# - Pass 3: resolve the shape and basis of the data in this final configuration

# How does diagonal basis interact with the smallest basis?
# what if the smallest basis is not the same as the basis which is diagonal?
# Can we use a recast basis to make use of the smaller basis?


@dataclass
class EinsumBasisHint:
    basis: Basis[Any, Any]
    cost_hint: int

    def dual_basis(self) -> EinsumBasisHint:
        return EinsumBasisHint(self.basis.dual_basis(), self.cost_hint)


class EinsumBasisHints:
    def __init__(self):
        self._hints = dict[str, list[EinsumBasisHint]]()

    def add_hint(self, idx: EinsteinIndex, hint: EinsumBasisHint) -> None:
        hints = self._hints.get(idx.label, [])
        hint = hint.dual_basis() if idx.is_dual else hint
        hints.append(hint)

    def __iadd__(self, other: EinsumBasisHints) -> Self:
        for k, v in other._hints.items():
            self._hints.setdefault(k, []).extend(v)
        return self


def _collect_basis_hints_tuple(
    basis: Basis[Any, Any],
    idx: tuple[NestedEinsteinIndex, ...],
    multiplier: int = 1,
) -> EinsumBasisHints:
    as_tuple = as_tuple_basis(basis)
    hints = EinsumBasisHints()
    for i, c in zip(idx, as_tuple.children):
        multiplier = int(as_tuple.size / c.size) * multiplier
        hints += _collect_basis_hints(basis, i, multiplier=multiplier)
    return hints


def _collect_basis_hints_pair(
    basis: Basis[Any, Any],
    idx: tuple[EinsteinIndex, EinsteinIndex],
    multiplier: int = 1,
) -> EinsumBasisHints:
    as_diagonal = as_diagonal_basis(basis)
    if as_diagonal is not None:
        # We save the full cost of the diagonal basis - not just the cost of the
        # axis itself
        cost = multiplier * (basis.fundamental_size - basis.size)

        hints = EinsumBasisHints()
        basis_0 = as_diagonal.inner[0]
        hints.add_hint(idx[0], EinsumBasisHint(basis_0, cost))

        basis_1 = as_diagonal.inner[1]
        hints.add_hint(idx[1], EinsumBasisHint(basis_1, cost))
        return hints

    return _collect_basis_hints_tuple(basis, idx, multiplier=multiplier)


def _collect_basis_hints(
    basis: Basis[Any, Any], idx: NestedEinsteinIndex, multiplier: int = 1
) -> EinsumBasisHints:
    if isinstance(idx, EinsteinIndex):
        cost = multiplier * (basis.fundamental_size - basis.size)
        hints = EinsumBasisHints()
        hints.add_hint(idx, EinsumBasisHint(basis, cost))
        return hints

    if (
        len(idx) == 2
        and isinstance(idx[0], EinsteinIndex)
        and isinstance(idx[1], EinsteinIndex)
    ):
        lhs, rhs = idx
        if isinstance(lhs, EinsteinIndex) and isinstance(rhs, EinsteinIndex):
            return _collect_basis_hints_pair(basis, (lhs, rhs), multiplier)

    return _collect_basis_hints_tuple(basis, idx, multiplier=multiplier)


@dataclass
class EinsumBasisSolution:
    basis: Basis[Any, Any]
    shape: NestedLength
    basis_map: EinsumBasisMap
    mapped: dict[str, str] = field(default_factory=dict)


def _get_basis_for_einsum_single[DT: np.number[Any]](
    idx: EinsteinIndex,
    basis: Basis[Any, DT],
    basis_map: EinsumBasisMap,
) -> EinsumBasisSolution:
    basis = basis_map.get(idx, basis)
    return EinsumBasisSolution(basis, basis.size, basis_map)


def _get_basis_for_einsum_tuple[DT: np.number[Any]](
    idx: tuple[NestedEinsteinIndex, ...],
    basis: Basis[Any, DT],
    basis_map: EinsumBasisMap,
) -> EinsumBasisSolution: ...


def _get_basis_for_einsum_pair[DT: np.number[Any]](
    idx: tuple[EinsteinIndex, EinsteinIndex],
    basis: Basis[Any, DT],
    basis_map: EinsumBasisMap,
) -> EinsumBasisSolution:
    as_diagonal = as_diagonal_basis(basis)
    if as_diagonal is not None:
        current_basis = basis_map.try_get(idx[0])
        return EinsumBasisSolution(basis, basis.size, basis_map)
    as_tuple = as_tuple_basis(basis)

    return EinsumBasisSolution(basis, basis.size, basis_map)


def _get_basis_for_einsum[DT: np.number[Any]](
    idx: NestedEinsteinIndex,
    basis: Basis[Any, DT],
    basis_map: EinsumBasisMap,
) -> EinsumBasisSolution:
    if isinstance(idx, EinsteinIndex):
        return _get_basis_for_einsum_single(idx, basis, basis_map)

    if (
        len(idx) == 2
        and isinstance(idx[0], EinsteinIndex)
        and isinstance(idx[1], EinsteinIndex)
    ):
        lhs, rhs = idx
        if isinstance(lhs, EinsteinIndex) or not isinstance(rhs, EinsteinIndex):
            return _get_basis_for_einsum_pair(idx, basis, basis_map)

    children = list[Basis[Any, DT]]()
    for c in as_tuple.children:
        child, basis_map = _get_basis_for_einsum(einsum_map, c, basis_map)
        children.append(child)

    return (tuple_basis(tuple(children), as_tuple.metadata().extra), basis_map)


_einsum_map = dict[
    str, Callable[[SlateArray[Any, Any], SlateArray[Any, Any]], SlateArray[Any, Any]]
]()


def _einsum_1[DT: np.number[Any]](
    array_1: Array[BasisMetadata, DT],
    array_2: Array[BasisMetadata, DT],
) -> Array[BasisMetadata, DT]:
    array_1_diag = as_diagonal_basis(array_1.basis)
    if array_1_diag is not None:
        return einsum("(ii),(i...)->(i...)", array_1, array_2)

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


_einsum_map["(ij),(i...)->(j...)"] = _einsum_1


def _einsum_2[DT: np.number[Any]](
    array_1: SlateArray[Any, DT],
    array_2: SlateArray[Any, DT],
) -> SlateArray[BasisMetadata, DT]:
    # Einsum for (ik),(m(kj))->(m(ij))
    array_1_diag = as_diagonal_basis(array_1.basis)
    if array_1_diag is not None:
        return einsum("(ii),(m(ij))->(m(ij))", array_1, array_2)

    array_2 = array_as_tuple_basis(array_2)
    array_2_diag = as_diagonal_basis(array_2.basis[1])
    if array_2_diag is not None:
        return einsum("(ij),(m(jj))->(m(ij))", array_1, array_2)

    array_1 = array_as_tuple_basis(array_1)
    array_2 = array_as_flatten_basis(array_2)
    # No diagonal basis efficient implementation
    final_basis = tuple_basis(
        (array_2.basis[0], tuple_basis((array_1.basis[0], array_2.basis[2]))), None
    )

    data = _einsum_numpy(
        "ik,mkj->mij",
        array_1.raw_data.reshape(array_1.basis.shape),
        array_2.raw_data.reshape(array_2.basis.shape),
    )
    return SlateArray(final_basis, data)


_einsum_map["(ik),(m(kj))->(m(ij))"] = _einsum_2


# TODO: need to do a more general approach for einsum


def _einsum_3[DT: np.number[Any]](
    array_1: SlateArray[Any, DT],
    array_2: SlateArray[Any, DT],
) -> SlateArray[BasisMetadata, DT]:
    # Einsum for (m(ik)),(kj)->(m(ij))
    array_1_tuple = as_tuple_basis(array_1.basis)
    array_1_diag = as_diagonal_basis(array_1_tuple[1])
    if array_1_diag is not None:
        return einsum("(m(ii)),(ij)->(m(ij))", array_1, array_2)

    array_2_diag = as_diagonal_basis(array_2.basis)
    if array_2_diag is not None:
        return einsum("(m(ij)),(jj)->(m(ij))", array_1, array_2)

    array_1 = array_as_flatten_basis(array_1)
    array_2 = array_as_tuple_basis(array_2)
    # No diagonal basis efficient implementation
    final_basis = tuple_basis(
        (array_1.basis[0], tuple_basis((array_1.basis[1], array_2.basis[1]))), None
    )

    data = _einsum_numpy(
        "mik,kj->mij",
        array_1.raw_data.reshape(array_1.basis.shape),
        array_2.raw_data.reshape(array_2.basis.shape),
    )
    return SlateArray(final_basis, data)


_einsum_map["(m(ik)),(kj)->(m(ij))"] = _einsum_3


def _einsum_4[DT: np.number[Any]](
    array_1: SlateArray[Any, DT],
    array_2: SlateArray[Any, DT],
) -> SlateArray[BasisMetadata, DT]:
    # Einsum for (m(ij)),(jj)->(m(ij))
    array_1 = array_as_tuple_basis(array_1)
    array_1_diag = as_diagonal_basis(array_1.basis[1])
    if array_1_diag is not None:
        return einsum("(m(jj)),(jj)->(m(jj))", array_1, array_2)
    j_basis = as_diagonal_basis(array_2.basis) or DiagonalBasis(
        as_tuple_basis(array_2.basis)  # type: ignore a
    )
    array_1 = array_1.with_basis(
        tuple_basis((array_1.basis[0], tuple_basis((j_basis, j_basis.inner[1]))))
    )
    array_2 = array_2.with_basis(j_basis)
    data = _einsum_numpy(
        "mj,j,mj",
        array_1.raw_data.reshape(-1, j_basis.size),
        array_2.raw_data.reshape(j_basis.size),
    )
    return SlateArray(array_1.basis, data)


_einsum_map["(m(ij)),(jj)->(m(ij))"] = _einsum_4


def _einsum_5[DT: np.number[Any]](
    array_1: SlateArray[Any, DT],
    array_2: SlateArray[Any, DT],
) -> SlateArray[BasisMetadata, DT]:
    # Einsum for (m(jj)),(jj)->(m(jj))
    # NOTE: we could maybe convert this to mj,j,mj
    array_1 = array_as_tuple_basis(array_1)
    array_1_diag = as_diagonal_basis(array_1.basis[1])

    array_2_diag = as_diagonal_basis(array_2.basis)

    j_basis = (
        array_1_diag or array_2_diag or DiagonalBasis(as_tuple_basis(array_2.basis))  # type: ignore any
    )
    array_1 = array_1.with_basis(tuple_basis((array_1.basis[0], j_basis)))
    array_2 = array_2.with_basis(j_basis)
    data = _einsum_numpy(
        "mj,j,mj",
        array_1.raw_data.reshape(-1, j_basis.size),
        array_2.raw_data.reshape(j_basis.size),
    )
    return SlateArray(array_1.basis, data)


_einsum_map["(m(jj)),(jj)->(m(jj))"] = _einsum_5


@dataclass
class EinsumSpecification:
    parts: tuple[NestedEinsteinIndex, ...]
    result: NestedEinsteinIndex


type NestedEinsteinIndex = EinsteinIndex | tuple[NestedEinsteinIndex, ...]

type NestedTokens = str | list[NestedTokens]


def parse_tokens(tokens: list[str], idx: int) -> tuple[NestedTokens, int]:
    out = list[NestedTokens]()

    while idx < len(tokens):
        token = tokens[idx]
        if token == "(":
            # Start of deeper nesting
            inner, idx = parse_tokens(tokens, idx + 1)
            out.append(inner)
            continue
        if token == ")":
            # End of current nesting
            return (out, idx + 1)

        # Add token to current nesting
        out.append(token)
        idx += 1

    return (out, idx + 1)


def process_tokens(tokens: NestedTokens) -> NestedEinsteinIndex:
    if isinstance(tokens, str):
        if tokens.endswith("'"):
            return EinsteinIndex(tokens[:-1], is_dual=True)
        return EinsteinIndex(tokens, is_dual=False)

    return tuple(process_tokens(t) for t in tokens)


def parse_einsum_index(idx: str) -> NestedEinsteinIndex:
    """
    Parse an einsum index string into a nested einstein index.

    An einsum index is a series of labels in a nested structure.

    eg
    (i j)
    (m (i j))
    (m (i j) k)
    (m (i (k l)) (k l))

    Labels are separated by spaces and nested by brackets.
    """
    idx_split = idx.replace("(", " ( ").replace(")", " ) ").split()
    return process_tokens(parse_tokens(idx_split, 0)[0][0])


def parse_einsum_specification(specification: str) -> EinsumSpecification:
    parts, result = specification.split("->")
    parts = tuple(parse_einsum_index(p) for p in parts.split(","))
    result = parse_einsum_index(result)
    return EinsumSpecification(parts, result)


def _get_einsum_data() -> tuple[np.ndarray[Any, Any], str]:
    return tuple()


def _get_einsum_result_basis(
    idx: NestedEinsteinIndex, basis_map: EinsumBasisMap
) -> Basis[Any, Any]:
    if isinstance(idx, EinsteinIndex):
        return basis_map[idx]
    return tuple_basis(tuple(_get_einsum_result_basis(c, basis_map) for c in idx), None)


def einsum[DT: np.number[Any]](
    idx: str,
    *arrays: SlateArray[BasisMetadata, DT],
) -> SlateArray[BasisMetadata, DT]:
    if idx == "(ij),(i...)->(j...)":
        return _einsum_map[idx](arrays[0], arrays[1])

    # TODO: how do we get the final basis for the einsum?
    specification = parse_einsum_specification(idx)
    hints = EinsumBasisHints()
    for index, array in zip(specification.parts, arrays):
        hints += _collect_basis_hints(array.basis, index)
    basis_map = get_basis_map(hints)

    np_einsum_data = list[np.ndarray[Any, Any]]()
    np_einsum_idx = list[str]()
    for index, array in zip(specification.parts, arrays):
        data, np_idx = _get_einsum_data(array, basis_map)
        np_einsum_data.append(data)
        np_einsum_idx.append(np_idx)

    out_data = _einsum_numpy("", *einsum_data)
    out_basis = _get_einsum_result_basis(specification.result, basis_map)
    return SlateArray(out_basis, out_data)
