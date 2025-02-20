from __future__ import annotations

import itertools
from typing import Any, Never, TypeGuard, cast, overload, override

import numpy as np

from slate.basis._basis import Basis, BasisFeature, ctype
from slate.basis._tuple import TupleBasis, TupleBasis2D
from slate.basis.wrapped import WrappedBasis, wrapped_basis_iter_inner
from slate.metadata import BasisMetadata, Metadata2D, TupleMetadata
from slate.util._diagonal import build_diagonal, extract_diagonal

# TODO: can we get rid of special block diagonal and generalize Diagonal to support  # noqa: FIX002
# arbitrary nested axes? Can we do this in a way which doesn't make the 'SimpleDiagonal'
# case much harder to use?
# block diagonal in its current impl is equivalent to re-casting a basis
# as [(list,other), (list,other),...] and then taking this
# general diagonal over the list basis
# The full diagonal basis can also be a special case of the general diagonal basis
# in this case...


class BlockDiagonalBasis[
    C: tuple[Basis[BasisMetadata, ctype[Never]], ...],
    E,
    DT: ctype[Never],
](
    WrappedBasis[TupleBasis[C, E, DT], DT],
):
    """Represents a diagonal basis."""

    def __init__(
        self,
        inner: TupleBasis[C, E, DT],
        block_shape: tuple[int, ...],
    ) -> None:
        super().__init__(cast("Any", inner))
        for child, s in zip(inner.children, block_shape, strict=True):
            assert child.size % s == 0

        self._block_shape = block_shape

    @property
    def block_shape(self) -> tuple[int, ...]:
        """The shape of each block matrix along the diagonal."""
        return self._block_shape

    @property
    def repeat_shape(self) -> tuple[int, ...]:
        """The shape of the repeats of blocks."""
        return tuple(
            self.inner.children[i].size // self.block_shape[i]
            for i in range(len(self.block_shape))
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
    def __into_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
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
        ).reshape(
            *vectors.shape[:axis],
            -1,
            *vectors.shape[axis + 1 :],
        )

    @override
    def __from_inner__(
        self,
        vectors: np.ndarray[Any, DT],
        axis: int = -1,
    ) -> np.ndarray[Any, DT]:
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
    def add_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
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
            cast("WrappedBasis[Any, ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


def is_block_diagonal_basis(
    basis: Basis[Any, Any],
) -> TypeGuard[
    BlockDiagonalBasis[
        tuple[Basis[BasisMetadata, ctype[Never]], ...], Never, ctype[Never]
    ]
]:
    return isinstance(basis, BlockDiagonalBasis)


@overload
def as_block_diagonal_basis[
    DT: np.dtype[np.generic],
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    basis: Basis[Metadata2D[M0, M1, E], DT],
) -> (
    BlockDiagonalBasis[DT, Any, E, TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E]]
    | None
): ...


@overload
def as_block_diagonal_basis[
    DT: np.dtype[np.generic],
    M: BasisMetadata,
    E,
](
    basis: Basis[TupleMetadata[M, E], DT],
) -> BlockDiagonalBasis[DT, M, E] | None: ...


@overload
def as_block_diagonal_basis[DT: np.dtype[np.generic]](
    basis: Basis[Any, DT],
) -> BlockDiagonalBasis[DT, BasisMetadata, Any] | None: ...


def as_block_diagonal_basis(
    basis: Any,
) -> Any:
    """Get the closest basis that is block diagonal."""
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if is_block_diagonal_basis(b)),
        None,
    )
