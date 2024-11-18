from __future__ import annotations

from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    Literal,
    Never,
    Self,
    TypeGuard,
    cast,
    overload,
    override,
)

import numpy as np

from slate.basis import Basis, FundamentalBasis
from slate.basis._basis import SimpleBasis
from slate.basis.wrapped import WrappedBasis, get_wrapped_basis_super_inner
from slate.metadata import BasisMetadata
from slate.metadata.stacked.stacked import StackedMetadata

if TYPE_CHECKING:
    from slate.metadata import SimpleMetadata

type StackedBasis[M: BasisMetadata, E, DT: np.generic] = Basis[
    StackedMetadata[M, E], DT
]


def stacked_basis_as_fundamental[M: BasisMetadata, E, DT: np.generic](
    basis: StackedBasis[M, E, DT],
) -> TupleBasis[M, E, DT]:
    """Get the equivalent Tuple of Fundamental Basis."""
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
    """
    swapped = vectors.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = initial.__convert_vector_into__(stacked, final, axis=ax)

    return stacked.reshape(-1, *swapped.shape[1:]).swapaxes(axis, 0)


class TupleBasis[M: BasisMetadata, E, DT: np.generic](
    SimpleBasis[StackedMetadata[M, E], DT],
):
    """Represents a Tuple of independent basis."""

    def __init__(
        self: Self, children: tuple[Basis[M, Any], ...], extra_metadata: E
    ) -> None:
        self._children = children
        assert all(isinstance(c, SimpleBasis) for c in children)
        self._metadata = StackedMetadata(
            tuple(i.metadata for i in children), extra_metadata
        )

    def conjugate_basis(self) -> TupleBasis[M, E, DT]:
        return self

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
            cast(TupleBasis[M, E, DT1], self),
            cast(TupleBasis[M, E, DT1], basis),
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

    def __eq__(self, value: object) -> bool:
        if isinstance(value, TupleBasis):
            return value.children == self.children  # type: ignore unknown
        return False

    def __hash__(self) -> int:
        return hash((self.metadata.extra, self.children))


class VariadicTupleBasis[DT: np.generic, *TS, E](TupleBasis[Any, E, DT]):
    """A variadic alternative to tuple basis.

    Note all sub basis must have the same datatype (DT), but it is not
    currently possible to add this information to the type system.
    """

    def __init__(self: Self, children: tuple[*TS], extra_metadata: E) -> None:
        super().__init__(
            cast(tuple[Basis[BasisMetadata, DT], ...], children), extra_metadata
        )

    def conjugate_basis(self) -> VariadicTupleBasis[DT, *TS, E]:
        return self

    @property
    def children(self) -> tuple[*TS]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast(tuple[*TS], super().children)

    @overload
    def __getitem__[B: Basis[Any, Any]](
        self: VariadicTupleBasis[DT, Any, Any, B, *tuple[Any, ...], E],
        index: Literal[2],
    ) -> B: ...

    @overload
    def __getitem__[B: Basis[Any, Any]](
        self: VariadicTupleBasis[DT, Any, B, *tuple[Any, ...], E], index: Literal[1]
    ) -> B: ...

    @overload
    def __getitem__[B: Basis[Any, Any]](
        self: VariadicTupleBasis[DT, B, *tuple[Any, ...], E], index: Literal[0]
    ) -> B: ...

    @overload
    def __getitem__(self: Self, index: int) -> Basis[Any, Any]: ...

    def __getitem__(self: Self, index: int) -> Basis[Any, Any]:
        return super().__getitem__(index)


@overload
def tuple_basis[*TS, E](
    children: tuple[*TS], extra_metadata: None = None
) -> VariadicTupleBasis[Any, *TS, None]: ...


@overload
def tuple_basis[*TS, E](
    children: tuple[*TS], extra_metadata: E
) -> VariadicTupleBasis[Any, *TS, E]: ...


def tuple_basis[*TS, E](
    children: tuple[*TS], extra_metadata: E | None = None
) -> VariadicTupleBasis[np.generic, *TS, E | None]:
    """Build a VariadicTupleBasis from a tuple."""
    return VariadicTupleBasis[Any, *TS, E | None](children, extra_metadata)


def tuple_basis_is_variadic[M: BasisMetadata, E, DT: np.generic](
    _basis: TupleBasis[M, E, DT],
) -> TypeGuard[VariadicTupleBasis[Any, *tuple[Basis[M, DT], ...], E]]:
    """Cast a TupleBasis as a VariadicTupleBasis."""
    return True


def tuple_basis_with_modified_children[
    M: BasisMetadata,
    E,
    DT: np.generic,
    DT1: np.generic,
](
    basis: TupleBasis[M, E, DT], wrapper: Callable[[int, Basis[M, DT1]], Basis[M, DT]]
) -> TupleBasis[M, E, DT1]:
    """Get the basis with modified children."""
    return TupleBasis[M, E, DT1](
        tuple(starmap(wrapper, enumerate(basis.children))), basis.metadata.extra
    )


def tuple_basis_with_modified_child[
    M: BasisMetadata,
    E,
    DT: np.generic,
    DT1: np.generic,
](
    basis: TupleBasis[M, E, DT],
    wrapper: Callable[[Basis[M, DT]], Basis[M, DT1]],
    idx: int,
) -> TupleBasis[M, E, DT | DT1]:
    """Get the basis with modified child."""
    return tuple_basis_with_modified_children(
        basis, lambda i, b: cast(Basis[M, DT | DT1], b if i != idx else wrapper(b))
    )


def tuple_basis_with_child[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], inner: Basis[M, DT], idx: int
) -> TupleBasis[M, E, DT]:
    """Get a basis with the basis at idx set to inner."""
    return tuple_basis_with_modified_child(basis, lambda _: inner, idx)


def fundamental_tuple_basis_from_metadata[M: BasisMetadata, E](
    metadata: StackedMetadata[M, E],
) -> TupleBasis[M, E, np.generic]:
    """Get a basis with the basis at idx set to inner."""
    children = tuple(FundamentalBasis(c) for c in metadata.children)
    return TupleBasis(children, metadata.extra)


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: None = None
) -> VariadicTupleBasis[np.generic, Basis[SimpleMetadata, np.generic], None]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: E
) -> VariadicTupleBasis[np.generic, Basis[SimpleMetadata, np.generic], E]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: None = None
) -> VariadicTupleBasis[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    None,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: E
) -> VariadicTupleBasis[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    E,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: None = None
) -> VariadicTupleBasis[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    None,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: E
) -> VariadicTupleBasis[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    E,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: None = None
) -> TupleBasis[SimpleMetadata, None, np.generic]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E
) -> TupleBasis[SimpleMetadata, E, np.generic]: ...


def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E | None = None
) -> TupleBasis[SimpleMetadata, E | None, np.generic]:
    """Get a basis with the basis at idx set to inner."""
    return fundamental_tuple_basis_from_metadata(
        StackedMetadata.from_shape(shape, extra=extra)
    )


def as_tuple_basis[M: BasisMetadata, E, DT: np.generic](
    basis: Basis[StackedMetadata[M, E], DT],
) -> TupleBasis[M, E, np.generic]:
    """Get the closest basis to basis that is a TupleBasis.

    - For a wrapped TupleBasis, this will return the unwrapped basis
    - For a basis which is not a wrapped tuple_basis, this returns the fundamental stacked basis

    This is useful for functions which require a TupleBasis, and ensures that
    only the minimal basis conversion is required.
    """
    if isinstance(basis, WrappedBasis):
        super_inner = get_wrapped_basis_super_inner(basis)  # type: ignore unknown
        if isinstance(super_inner, TupleBasis):
            return cast(TupleBasis[M, E, np.generic], super_inner)

    return fundamental_tuple_basis_from_metadata(basis.metadata)
