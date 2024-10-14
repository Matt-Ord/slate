from __future__ import annotations

from typing import Any, Iterator, Literal, Never, Self, cast, overload, override

import numpy as np

from slate.basis import Basis, FundamentalBasis
from slate.basis.metadata import BasisMetadata


class TupleMetadata[M: BasisMetadata, E](BasisMetadata):
    """Metadata built from a tuple of individual metadata entries."""

    def __init__(self: Self, children: tuple[M, ...], extra: E) -> None:
        self._children = children
        self._extra = extra

    @property
    def children(self: Self) -> tuple[M, ...]:
        """Children metadata."""
        return self._children

    @property
    def extra(self: Self) -> E:
        """Extra metadata."""
        return self._extra

    @property
    def fundamental_shape(self: Self) -> tuple[int, ...]:
        """Shape of the full data."""
        return tuple(np.prod(i.fundamental_shape).item() for i in self.children)

    def __getitem__(self: Self, index: int) -> M:
        return self.children[index]

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TupleMetadata):
            return (self.extra == value.extra) and all(  # type: ignore unknown
                a == b
                for (a, b) in zip(self.children, value.children)  # type: ignore unknown
            )
        return False

    def __hash__(self) -> int:
        return hash((self.extra, self.children))


type StackedBasis[M: BasisMetadata, E, DT: np.generic] = Basis[TupleMetadata[M, E], DT]


def stacked_basis_as_fundamental[M: BasisMetadata, E, DT: np.generic](
    basis: StackedBasis[M, E, DT],
) -> TupleBasis[M, E, DT]:
    """Get the equivalent Tuple of Fundamental Basis.

    Returns
    -------
    TupleBasis[M, E, DT]
    """
    return TupleBasis[M, E, DT](
        tuple(FundamentalBasis(basis.metadata[i]) for i in range(basis.n_dim)),
        basis.metadata.extra,
    )


def _convert_tuple_basis_vector[M: BasisMetadata, E, DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    initial_basis: TupleBasis[M, E, DT],
    final_basis: TupleBasis[M, E, DT],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    """
    Convert a vector, expressed in terms of the given basis from_config in the basis to_config.

    Parameters
    ----------
    vector : np.ndarray[tuple[int], np.dtype[np.complex_] | np.dtype[np.float_]]
        the vector to convert
    from_config : B3d0Inv
    to_config : B3d1Inv
    axis : int, optional
        axis along which to convert, by default -1

    Returns
    -------
    np.ndarray[tuple[int], np.dtype[np.complex_]]
    """
    swapped = vectors.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = initial.__convert_vector_into__(stacked, final, axis=ax)

    return stacked.reshape(-1, *swapped.shape[1:]).swapaxes(axis, 0)


class TupleBasis[M: BasisMetadata, E, DT: np.generic](Basis[TupleMetadata[M, E], DT]):
    """Represents a Tuple of independent basis."""

    def __init__(
        self: Self, children: tuple[Basis[M, DT], ...], extra_metadata: E
    ) -> None:
        self._children = children
        self._metadata = TupleMetadata(
            tuple(i.metadata for i in children), extra_metadata
        )

    @property
    def children(self) -> tuple[Basis[M, DT], ...]:
        """Children basis."""
        return self._children

    @property
    def size(self) -> int:
        """Number of elements in the basis."""
        return np.prod(self.shape).item()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the children data."""
        return tuple(x.size for x in self.children)

    @override
    def __into_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        basis = stacked_basis_as_fundamental(self)
        return _convert_tuple_basis_vector(
            vectors,
            cast(TupleBasis[M, E, DT1], basis),
            cast(TupleBasis[M, E, DT1], self),
            axis,
        )

    @override
    def __from_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        basis = stacked_basis_as_fundamental(self)
        return _convert_tuple_basis_vector(
            vectors,
            cast(TupleBasis[M, E, DT1], basis),
            cast(TupleBasis[M, E, DT1], self),
            axis,
        )

    def __iter__(self) -> Iterator[Basis[M, DT]]:
        return self.children.__iter__()

    def __getitem__(self: Self, index: int) -> Basis[M, DT]:
        return self.children[index]

    def __convert_vector_into__[
        DT1: np.generic
    ](  # [DT1: DT, B1: Basis[TupleMetadata[M, E], DT]]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        basis: Basis[BasisMetadata, Never],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        assert self.metadata == basis.metadata

        if self == basis:
            return vectors

        if not isinstance(basis, TupleBasis):
            return super().__convert_vector_into__(vectors, basis, axis)

        # We overload __convert_vector_into__, more likely to get the 'happy path'
        return _convert_tuple_basis_vector(vectors, self, basis, axis)  # type: ignore unknown


class VariadicTupleBasis[*TS, E, DT: np.generic](TupleBasis[BasisMetadata, E, DT]):
    """A variadic alternative to tuple basis.

    Note all sub basis must have the same datatype (DT), but it is not
    currently possible to add this information to the type system.
    """

    def __init__(self: Self, children: tuple[*TS], extra_metadata: E) -> None:
        super().__init__(
            cast(tuple[Basis[BasisMetadata, DT], ...], children), extra_metadata
        )

    @property
    def children(self) -> tuple[*TS]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast(tuple[*TS], super().children)

    @overload
    def __getitem__[B: Basis[Any, Any]](
        self: VariadicTupleBasis[Any, Any, B, *tuple[Any, ...], E, DT],
        index: Literal[2],
    ) -> B: ...
    @overload
    def __getitem__[B: Basis[Any, Any]](
        self: VariadicTupleBasis[Any, B, *tuple[Any, ...], E, DT], index: Literal[1]
    ) -> B: ...
    @overload
    def __getitem__[B: Basis[Any, Any]](
        self: VariadicTupleBasis[B, *tuple[Any, ...], E, DT], index: Literal[0]
    ) -> B: ...
    @overload
    def __getitem__(self: Self, index: int) -> Basis[Any, Any]: ...

    def __getitem__(self: Self, index: int) -> Basis[Any, Any]: ...
