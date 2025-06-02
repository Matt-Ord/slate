from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Never, TypeGuard, cast, overload, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._tuple import is_tuple
from slate_core.basis._wrapped import AsUpcast, WrappedBasis, wrapped_basis_iter_inner
from slate_core.metadata import BasisMetadata
from slate_core.util._diagonal import (
    apply_contractions,
    expand_contractions,
)

if TYPE_CHECKING:
    from slate_core.metadata._shape import NestedLength


def _flatten_nested_index(
    index: NestedIndex,
) -> tuple[int, ...]:
    if isinstance(index, int):
        return (index,)
    return tuple(i for idx in index for i in _flatten_nested_index(idx))


type NestedIndex = int | tuple[NestedIndex, ...]


def _get_contracted_index(
    index: int,
    contractions: tuple[set[int], ...],
) -> int:
    """Get the outer index for a contraction."""
    for contraction in contractions:
        if index in contraction:
            return min(contraction)
    return index


def _get_contracted_indices(
    index: NestedIndex,
    contractions: tuple[set[int], ...],
) -> NestedIndex:
    """Get the outer index for a contraction."""
    if isinstance(index, int):
        return _get_contracted_index(index, contractions)

    return tuple(_get_contracted_indices(i, contractions) for i in index)


def _get_outer_indices(
    index: NestedIndex,
) -> tuple[int, ...]:
    """Get the outer index for a contraction."""
    return tuple(sorted(set(_flatten_nested_index(index))))


def _build_contraction_size_map(
    basis: Basis,
    index: NestedIndex,
    *,
    size_map: dict[int, int] | None = None,
) -> dict[int, int]:
    size_map = size_map or {}
    if isinstance(index, int):
        outer_size = size_map.get(index, basis.size)
        size_map[index] = outer_size
        if outer_size != basis.size:
            msg = (
                f"Contraction has inconsistent sizes for contracted index."
                f"The size at {index} is {outer_size}, but the basis size is {basis.size}. "
            )
            raise ValueError(msg)
        return size_map
    assert is_tuple(basis), "Basis must be a tuple basis for contraction."
    for i, child in zip(index, basis.children, strict=True):
        size_map = _build_contraction_size_map(child, i, size_map=size_map)
    return size_map


def _build_shape(
    index: NestedIndex,
    size_map: dict[int, int],
) -> NestedLength:
    if isinstance(index, int):
        return size_map[index]
    return tuple(_build_shape(idx, size_map) for idx in index)


def _get_flattened_shape(length: NestedLength) -> tuple[int, ...]:
    if isinstance(length, int):
        return (length,)
    return tuple(i for idx in length for i in _get_flattened_shape(idx))


class ContractedBasis[
    B: Basis = Basis,
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[B, CT],
):
    """Represents a basis which has been contracted along an axis."""

    def __init__[B_: Basis](
        self: ContractedBasis[B_, Ctype[Never]],
        inner: B_,
        index: NestedIndex,
    ) -> None:
        super().__init__(cast("B", inner))

        self._index = index
        # Assert that it is possible to build the size map
        _ = self._size_map

    @cached_property
    def _size_map(self) -> dict[int, int]:
        """The size map for the indices."""
        return _build_contraction_size_map(self.inner, self._index)

    @cached_property
    def _inner_shape(self) -> NestedLength:
        """The shape of the contracted basis."""
        return _build_shape(self._index, self._size_map)

    @cached_property
    def _outer_indices(self) -> NestedIndex:
        """The outer indices of the contraction."""
        return _get_outer_indices(self._index)

    @cached_property
    def _outer_shape(self) -> NestedLength:
        """The shape of the contracted basis."""
        return _build_shape(self._outer_indices, self._size_map)

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: ContractedBasis[Basis[Any, DT_], Any],
    ) -> ContractedBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("ContractedBasis[B, DT_]", self)

    @override
    def upcast[M: BasisMetadata](
        self: ContractedBasis[Basis[M], Any],
    ) -> AsUpcast[ContractedBasis[B, CT], M, CT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

    @property
    @override
    def size(self) -> int:
        return np.prod(_get_flattened_shape(self._outer_shape)).item()

    @override
    def __into_inner__[T1: np.generic, T2: np.generic](
        self: ContractedBasis[Any, Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T1]:
        axis %= vectors.ndim

        s0 = int(np.prod(vectors.shape[:axis]))  # type: ignore[assignment]
        s1 = int(np.prod(vectors.shape[axis + 1 :]))  # type: ignore[assignment]
        flat_index = _flatten_nested_index(self._index)
        flat_out_index = _flatten_nested_index(self._outer_indices)

        def fn() -> np.ndarray[Any, np.dtype[T2]]:
            stacked = vectors.reshape(s0, *_get_flattened_shape(self._outer_shape), s1)

            out_axes: tuple[int, ...] = (
                0,
                *(
                    next(i + 1 for i, val in enumerate(flat_out_index) if val == idx)
                    for idx in flat_index
                ),
                stacked.ndim - 1,
            )
            contracted = expand_contractions(stacked, out_axes)
            return contracted.reshape(
                *vectors.shape[:axis], -1, *vectors.shape[axis + 1 :]
            )

        return BasisConversion(fn)

    @override
    def __from_inner__[T1: np.generic, T2: np.generic](
        self: ContractedBasis[Any, Ctype[T2]],
        vectors: np.ndarray[Any, np.dtype[T1]],
        axis: int = -1,
    ) -> BasisConversion[T2, T1, T2]:
        axis %= vectors.ndim

        s0 = int(np.prod(vectors.shape[:axis]))  # type: ignore[assignment]
        s1 = int(np.prod(vectors.shape[axis + 1 :]))  # type: ignore[assignment]
        flat_index = _flatten_nested_index(self._index)
        flat_out_index = _flatten_nested_index(self._outer_indices)

        def fn() -> np.ndarray[Any, np.dtype[T1]]:
            stacked = vectors.reshape(s0, *_get_flattened_shape(self._inner_shape), s1)

            mapped_contractions = (
                {0},
                *(
                    {i + 1 for i, val in enumerate(flat_index) if val == idx}
                    for idx in flat_out_index
                ),
                {stacked.ndim - 1},
            )
            contracted = apply_contractions(stacked, mapped_contractions)
            return contracted.reshape(
                *vectors.shape[:axis], -1, *vectors.shape[axis + 1 :]
            )

        return BasisConversion(fn)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            is_contracted(other)
            and (other.inner == self.inner)
            and self.is_dual == other.is_dual
            and self._index == other._index
        )

    @override
    def __hash__(self) -> int:
        return hash((3, self.inner, self.is_dual, self._index))

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("LINEAR_MAP")
            out.add("MUL")
            out.add("SUB")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
    ) -> np.ndarray[Any, np.dtype[T]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[T: np.number](
        self, lhs: np.ndarray[Any, np.dtype[T]], rhs: complex
    ) -> np.ndarray[Any, np.dtype[T]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
    ) -> np.ndarray[Any, np.dtype[T]]:
        if "LINEAR_MAP" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)

        return (
            cast("WrappedBasis[Any, Ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


@overload
def is_contracted[
    M: BasisMetadata,
    CT: Ctype[Never],
](
    basis: Basis[M, CT],
) -> TypeGuard[ContractedBasis[Basis[M], CT]]: ...
@overload
def is_contracted(
    basis: object,
) -> TypeGuard[ContractedBasis[Basis, Ctype[Never]]]: ...


def is_contracted(
    basis: object,
) -> TypeGuard[ContractedBasis[Basis, Ctype[Never]]]:
    return isinstance(basis, ContractedBasis)


def as_contracted[
    M: BasisMetadata,
    CT: Ctype[Never],
](
    basis: Basis[M, CT],
) -> ContractedBasis[Basis[M], CT] | None:
    """Get the closest basis that is block diagonal."""
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if is_contracted(b)),
        None,
    )
