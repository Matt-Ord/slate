from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, cast, override

import numpy as np

from slate import BasisMetadata, TupleBasis
from slate.basis import BasisFeature, WrappedBasis
from slate.metadata.stacked import StackedMetadata
from slate.util._diagonal import build_diagonal, extract_diagonal

if TYPE_CHECKING:
    from collections.abc import Callable

# TODO: can we get rid of special block diagonal and generalize Diagonal to support
# arbitrary nested axes? Can we do this in a way which doesn't make the 'SimpleDiagonal'
# case much harder to use?
# block diagonal in its current impl is equivalent to re-casting a basis
# as [(list,other), (list,other),...] and then taking this
# general diagonal over the list basis
# The full diagonal basis can also be a special case of the general diagonal basis
# in this case...


class BlockDiagonalBasis[
    DT: np.generic,
    M: BasisMetadata,
    E,
    B: TupleBasis[Any, Any, Any] = TupleBasis[M, E, DT],
](
    WrappedBasis[StackedMetadata[M, E], DT, B],
):
    """Represents a diagonal basis."""

    def __init__[
        _DT: np.generic,
        _B: TupleBasis[Any, Any, Any],
    ](
        self: BlockDiagonalBasis[_DT, Any, Any, _B],
        inner: _B,
        block_shape: tuple[int, ...],
    ) -> None:
        super().__init__(cast("Any", inner))
        for child, s in zip(inner.children, block_shape, strict=True):
            assert child.size % s == 0

        self._block_shape = block_shape

    @property
    @override
    def inner(self) -> B:
        return self._inner

    @property
    def block_shape(self) -> tuple[int, ...]:
        """The shape of each block matrix along the diagonal."""
        return self._block_shape

    @property
    def repeat_shape(self) -> tuple[int, int]:
        """The shape of the repeats of blocks."""
        return (
            self.inner.children[0].size // self.block_shape[0],
            self.inner.children[1].size // self.block_shape[1],
        )

    @property
    def n_repeats(self) -> int:
        """Total number of repeats."""
        return min(self.repeat_shape)

    @property
    @override
    def size(self) -> int:
        return np.prod(self.block_shape).item() * self.n_repeats

    @override
    def __into_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

        stacked = vectors.reshape(
            *vectors.shape[:axis],
            self.n_repeats,
            *self.block_shape,
            *vectors.shape[axis + 1 :],
        )
        return build_diagonal(
            stacked,
            axis,
            out_shape=self.repeat_shape,
            out_axes=tuple(range(axis, axis + 2 * len(self.block_shape), 2)),
        )

    @override
    def __from_inner__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        axis %= vectors.ndim

        # The vectors in the inner basis are stored in
        # the shape [(list,momentum), (list,momentum),...]
        k_shape = self.block_shape
        list_shape = self.repeat_shape
        inner_shape = tuple(
            (n_list, n_k) for (n_k, n_list) in zip(k_shape, list_shape, strict=False)
        )
        stacked = vectors.reshape(
            *vectors.shape[:axis],
            *(itertools.chain(*inner_shape)),
            *vectors.shape[axis + 1 :],
        )

        return extract_diagonal(
            stacked, tuple(range(axis, axis + 2 * len(inner_shape), 2)), axis
        )

    @override
    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, BlockDiagonalBasis)
            and (other.inner == self.inner)  # type: ignore unknown
            and self.is_dual == other.is_dual
        )

    @override
    def __hash__(self) -> int:
        return hash((3, self.inner, self.is_dual))

    @override
    def with_inner[  # type: ignore there is no way to bound inner in parent
        _B: TupleBasis[Any, Any, Any],
    ](self, inner: _B) -> BlockDiagonalBasis[DT, M, E, _B]:
        return self.with_modified_inner(lambda _: inner)

    @override
    def with_modified_inner[  # type: ignore there is no way to bound the wrapper function in the parent class
        _DT: np.generic,
        _M: BasisMetadata,
        _E,
        _B: TupleBasis[Any, Any, Any] = TupleBasis[_M, _E, _DT],
    ](
        self,
        wrapper: Callable[[TupleBasis[_M, _E, _DT]], _B],
    ) -> BlockDiagonalBasis[_DT, _M, _E, _B]:
        """Get the wrapped basis after wrapper is applied to inner."""
        return BlockDiagonalBasis[_DT, _M, _E, _B](
            wrapper(self.inner), self.block_shape
        )

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "SIMPLE_ADD" in self.inner.features:
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if "SIMPLE_MUL" in self.inner.features:
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if "SIMPLE_SUB" in self.inner.features:
            out.add("SUB")
            out.add("SIMPLE_SUB")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)
        return self.__from_inner__(self.inner.points)
