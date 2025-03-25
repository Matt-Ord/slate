from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any, Never, TypeGuard, cast, overload, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, BasisFeature, Ctype
from slate_core.basis._tuple import TupleBasis
from slate_core.basis._wrapped import AsUpcast, WrappedBasis, wrapped_basis_iter_inner
from slate_core.metadata import BasisMetadata
from slate_core.util._diagonal import build_diagonal, extract_diagonal

if TYPE_CHECKING:
    from slate_core.metadata._tuple import TupleMetadata


# TODO: can we get rid of special block diagonal and generalize Diagonal to support  # noqa: FIX002
# arbitrary nested axes? Can we do this in a way which doesn't make the 'SimpleDiagonal'
# case much harder to use?
# block diagonal in its current impl is equivalent to re-casting a basis
# as [(list,other), (list,other),...] and then taking this
# general diagonal over the list basis
# The full diagonal basis can also be a special case of the general diagonal basis
# in this case...


class BlockDiagonalBasis[
    B: TupleBasis[tuple[Basis, ...], Any] = TupleBasis[tuple[Basis, ...], Any],
    CT: Ctype[Never] = Ctype[Never],
](
    WrappedBasis[B, CT],
):
    """Represents a diagonal basis."""

    def __init__[B_: TupleBasis[tuple[Basis, ...], Any]](
        self: BlockDiagonalBasis[B_, Ctype[Never]],
        inner: B_,
        block_shape: tuple[int, ...],
    ) -> None:
        super().__init__(cast("B", inner))
        for child, s in zip(inner.children, block_shape, strict=True):
            assert child.size % s == 0

        self._block_shape = block_shape

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: BlockDiagonalBasis[TupleBasis[Any, Any, DT_], Any],
    ) -> BlockDiagonalBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("BlockDiagonalBasis[B, DT_]", self)

    @overload
    def upcast[M0: BasisMetadata, E](
        self: BlockDiagonalBasis[TupleBasis[tuple[Basis[M0]], E], Any],
    ) -> AsUpcast[BlockDiagonalBasis[B, CT], TupleMetadata[tuple[M0], E], CT]: ...
    @overload
    def upcast[M0: BasisMetadata, M1: BasisMetadata, E](
        self: BlockDiagonalBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> AsUpcast[BlockDiagonalBasis[B, CT], TupleMetadata[tuple[M0, M1], E], CT]: ...
    @overload
    def upcast[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E](
        self: BlockDiagonalBasis[
            TupleBasis[tuple[Basis[M0], Basis[M1], Basis[M2]], E], Any
        ],
    ) -> AsUpcast[
        BlockDiagonalBasis[B, CT], TupleMetadata[tuple[M0, M1, M2], E], CT
    ]: ...
    @overload
    def upcast[M: BasisMetadata, E](
        self: BlockDiagonalBasis[TupleBasis[tuple[Basis[M], ...], E], Any],
    ) -> AsUpcast[BlockDiagonalBasis[B, CT], TupleMetadata[tuple[M, ...], E], CT]: ...
    @override
    def upcast(
        self,
    ) -> AsUpcast[
        BlockDiagonalBasis[B, CT], TupleMetadata[tuple[BasisMetadata, ...], Any], CT
    ]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

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
    def __into_inner__[T1: np.generic, T2: np.generic](
        self: BlockDiagonalBasis[Any, Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T1]:
        axis %= vectors.ndim

        def fn() -> np.ndarray[Any, np.dtype[T2]]:
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

        return BasisConversion(fn)

    @override
    def __from_inner__[T1: np.generic, T2: np.generic](
        self: BlockDiagonalBasis[Any, Ctype[T2]],
        vectors: np.ndarray[Any, np.dtype[T1]],
        axis: int = -1,
    ) -> BasisConversion[T2, T1, T2]:
        axis %= vectors.ndim

        def fn() -> np.ndarray[Any, np.dtype[T1]]:
            # The vectors in the inner basis are stored in
            # the shape [(list,momentum), (list,momentum),...]
            k_shape = self.block_shape
            list_shape = self.repeat_shape
            inner_shape = tuple(
                (n_list, n_k)
                for (n_k, n_list) in zip(k_shape, list_shape, strict=False)
            )
            stacked = vectors.reshape(
                *vectors.shape[:axis],
                *(itertools.chain(*inner_shape)),
                *vectors.shape[axis + 1 :],
            )

            return extract_diagonal(
                stacked, tuple(range(axis, axis + 2 * len(inner_shape), 2)), axis
            )

        return BasisConversion(fn)

    @override
    def __eq__(self, other: object) -> bool:
        return (
            is_block_diagonal_basis(other)
            and (other.inner == self.inner)
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
def is_block_diagonal_basis[  # type: ignore not incompatible
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
](
    basis: Basis[TupleMetadata[tuple[M0, M1], E], CT],
) -> TypeGuard[
    BlockDiagonalBasis[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E], CT]
]: ...
@overload
def is_block_diagonal_basis(
    basis: object,
) -> TypeGuard[
    BlockDiagonalBasis[TupleBasis[tuple[Basis, Basis], Never], Ctype[Never]]
]: ...


def is_block_diagonal_basis(
    basis: object,
) -> TypeGuard[BlockDiagonalBasis[TupleBasis[tuple[Basis, Basis], Any], Ctype[Never]]]:
    return isinstance(basis, BlockDiagonalBasis)


def as_block_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[Never],
](
    basis: Basis[TupleMetadata[tuple[M0, M1], E], CT],
) -> BlockDiagonalBasis[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E], CT] | None:
    """Get the closest basis that is block diagonal."""
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if is_block_diagonal_basis(b)),
        None,
    )
