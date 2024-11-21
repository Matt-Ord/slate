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

from slate.basis._basis import Basis, BasisFeature
from slate.basis.fundamental import FundamentalBasis
from slate.basis.wrapped import get_wrapped_basis_super_inner
from slate.metadata import (
    BasisMetadata,
    Metadata1D,
    Metadata2D,
    Metadata3D,
    MetadataND,
    StackedMetadata,
)

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
        tuple(FundamentalBasis(basis.metadata()[i]) for i in range(basis.n_dim)),
        basis.metadata().extra,
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


class TupleBasis[
    M: BasisMetadata,
    E,
    DT: np.generic,
    _InnerM: StackedMetadata[Any, Any] = StackedMetadata[M, E],
](
    Basis[_InnerM, DT],
):
    """Represents a Tuple of independent basis."""

    def __init__(
        self: Self, children: tuple[Basis[M, Any], ...], extra_metadata: E
    ) -> None:
        self._children = children
        self._metadata = StackedMetadata(
            tuple(i.metadata() for i in children), extra_metadata
        )

    @override
    def conjugate_basis(self) -> TupleBasis[M, E, Any, _InnerM]:
        return self

    @property
    def children(self) -> tuple[Basis[M, DT], ...]:
        """Children basis."""
        return self._children

    @property
    @override
    def size(self) -> int:
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

    @override
    def __convert_vector_into__[
        DT1: np.generic
    ](  # [DT1: DT, B1: Basis[TupleMetadata[M, E], DT]]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        basis: Basis[BasisMetadata, Never],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        assert self.metadata() == basis.metadata()

        if self == basis:
            return vectors

        if not isinstance(basis, TupleBasis):
            return super().__convert_vector_into__(vectors, basis, axis)

        # We overload __convert_vector_into__, more likely to get the 'happy path'
        return _convert_tuple_basis_vector(vectors, self, basis, axis)  # type: ignore unknown

    @override
    def __eq__(self, value: object) -> bool:
        if isinstance(value, TupleBasis):
            return value.children == self.children  # type: ignore unknown
        return False

    @override
    def __hash__(self) -> int:
        return hash((self.metadata().extra, self.children))

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if all("SIMPLE_ADD" in c.features for c in self.children):
            out.add("ADD")
            out.add("SIMPLE_ADD")
        if all("SIMPLE_MUL" in c.features for c in self.children):
            out.add("MUL")
            out.add("SIMPLE_MUL")
        if all("SIMPLE_SUB" in c.features for c in self.children):
            out.add("SUB")
            out.add("SIMPLE_SUB")
        if all("INDEX" in c.features for c in self.children):
            out.add("INDEX")
        return out

    @override
    def add_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_ADD" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self: Self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: float
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_MUL" not in self.features:
            msg = "mul_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs * rhs).astype(lhs.dtype)

    @override
    def sub_data[DT1: np.number[Any]](
        self: Self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: np.ndarray[Any, np.dtype[DT1]],
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "SIMPLE_SUB" not in self.features:
            msg = "sub_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs - rhs).astype(lhs.dtype)

    @property
    @override
    def points(self: Self) -> np.ndarray[Any, np.dtype[np.int_]]:
        if "INDEX" not in self.features:
            msg = "points not implemented for this basis"
            raise NotImplementedError(msg)
        return self.__from_fundamental__(np.arange(self.size))


class TupleBasisND[
    DT: np.generic,
    E,
    _InnerM: MetadataND[*tuple[BasisMetadata, ...], Any] = MetadataND[
        *tuple[BasisMetadata, ...], E
    ],
](
    TupleBasis[Any, E, DT, _InnerM],
):
    """A variadic alternative to tuple basis.

    Note all sub basis must have the same datatype (DT), but it is not
    currently possible to add this information to the type system.
    """

    def __init__(
        self: Self, children: tuple[Basis[BasisMetadata, DT], ...], extra_metadata: E
    ) -> None:
        msg = "TupleBasisND is only a placeholder type, and should not be directly created."
        raise NotImplementedError(msg)
        super().__init__(
            cast(tuple[Basis[BasisMetadata, DT], ...], children), extra_metadata
        )

    @override
    def metadata(self) -> _InnerM:
        return super().metadata()

    @override
    def conjugate_basis(self) -> TupleBasisND[DT, E, _InnerM]:
        return self

    @property
    def children(self) -> tuple[Basis[BasisMetadata, DT], ...]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast(tuple[Basis[BasisMetadata, DT], ...], super().children)

    @override
    def __getitem__(self: Self, index: int) -> Basis[BasisMetadata, DT]:
        return super().__getitem__(index)


class TupleBasis1D[
    DT: np.generic,
    B0: Basis[BasisMetadata, Any],
    E,
](
    TupleBasisND[DT, E, Metadata1D[Any, E]],
):
    """A variadic alternative to tuple basis.

    Note all sub basis must have the same datatype (DT), but it is not
    currently possible to add this information to the type system.
    """

    def __init__[
        _DT: np.generic,
        _B0: Basis[BasisMetadata, Any],
        _E,
    ](
        self: TupleBasis1D[_DT, _B0, _E],
        children: tuple[_B0],
        extra_metadata: _E,
    ) -> None:
        msg = "TupleBasis1D is only a placeholder type, and should not be directly created."
        raise NotImplementedError(msg)
        super().__init__(
            cast(tuple[Basis[BasisMetadata, _DT]], children), extra_metadata
        )

    @override
    def metadata[_M0: BasisMetadata](
        self: TupleBasis1D[Any, Basis[_M0, Any], E],
    ) -> Metadata1D[_M0, E]:
        return cast(Metadata1D[_M0, E], super().metadata())

    @override
    def conjugate_basis(self) -> TupleBasis1D[DT, B0, E]:
        return cast(TupleBasis1D[DT, B0, E], super().conjugate_basis())

    @property
    def children(self) -> tuple[B0]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast(tuple[B0], super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> B0: ...

    @overload
    def __getitem__(self: Self, index: int) -> B0: ...

    def __getitem__(self: Self, index: int) -> B0:  # type: ignore inconsistent type
        return cast(B0, super().__getitem__(index))


class TupleBasis2D[
    DT: np.generic,
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    E,
](
    TupleBasisND[DT, E, Metadata2D[Any, Any, E]],
):
    """A variadic alternative to tuple basis.

    Note all sub basis must have the same datatype (DT), but it is not
    currently possible to add this information to the type system.
    """

    def __init__[
        _DT: np.generic,
        _B0: Basis[BasisMetadata, Any],
        _B1: Basis[BasisMetadata, Any],
        _E,
    ](
        self: TupleBasis2D[_DT, _B0, _B1, _E],
        children: tuple[_B0, _B1],
        extra_metadata: _E,
    ) -> None:
        msg = "TupleBasis2D is only a placeholder type, and should not be directly created."
        raise NotImplementedError(msg)
        super().__init__(children, extra_metadata)

    @override
    def metadata[_M0: BasisMetadata, _M1: BasisMetadata](
        self: TupleBasis2D[Any, Basis[_M0, Any], Basis[_M1, Any], E],
    ) -> Metadata2D[_M0, _M1, E]:
        return cast(Metadata2D[_M0, _M1, E], super().metadata())

    @override
    def conjugate_basis(self) -> TupleBasis2D[DT, B0, B1, E]:
        return cast(TupleBasis2D[DT, B0, B1, E], super().conjugate_basis())

    @property
    def children(self) -> tuple[B0, B1]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast(tuple[B0, B1], super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> B0: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> B1: ...

    @overload
    def __getitem__(self: Self, index: int) -> B0 | B1: ...

    def __getitem__(self: Self, index: int) -> B0 | B1:  # type: ignore overload
        return cast(B0 | B1, super().__getitem__(index))


class TupleBasis3D[
    DT: np.generic,
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    E,
](
    TupleBasisND[DT, E, Metadata3D[Any, Any, Any, E]],
):
    """A variadic alternative to tuple basis.

    Note all sub basis must have the same datatype (DT), but it is not
    currently possible to add this information to the type system.
    """

    def __init__[
        _DT: np.generic,
        _B0: Basis[BasisMetadata, Any],
        _B1: Basis[BasisMetadata, Any],
        _B2: Basis[BasisMetadata, Any],
        _E,
    ](
        self: TupleBasis3D[_DT, _B0, _B1, _B2, _E],
        children: tuple[_B0, _B1, _B2],
        extra_metadata: _E,
    ) -> None:
        msg = "TupleBasis3D is only a placeholder type, and should not be directly created."
        raise NotImplementedError(msg)
        super().__init__(children, extra_metadata)

    @override
    def metadata[_M0: BasisMetadata, _M1: BasisMetadata, _M2: BasisMetadata](
        self: TupleBasis3D[Any, Basis[_M0, Any], Basis[_M1, Any], Basis[_M2, Any], E],
    ) -> Metadata3D[_M0, _M1, _M2, E]:
        return cast(Metadata3D[_M0, _M1, _M2, E], super().metadata())

    @override
    def conjugate_basis(self) -> TupleBasis3D[DT, B0, B1, B2, E]:
        return cast(TupleBasis3D[DT, B0, B1, B2, E], super().conjugate_basis())

    @property
    def children(self) -> tuple[B0, B1, B2]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast(tuple[B0, B1, B2], super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> B0: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> B1: ...

    @overload
    def __getitem__(self, index: Literal[2]) -> B2: ...

    @overload
    def __getitem__(self: Self, index: int) -> B0 | B1 | B2: ...

    def __getitem__(self: Self, index: int) -> B0 | B1 | B2:  # type: ignore overload
        return cast(B0 | B1 | B2, super().__getitem__(index))


@overload
def tuple_basis[
    *TS,
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    E,
](children: tuple[B0], extra_metadata: None = None) -> TupleBasis1D[Any, B0, None]: ...
@overload
def tuple_basis[
    *TS,
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    E,
](children: tuple[B0], extra_metadata: E) -> TupleBasis1D[Any, B0, E]: ...


@overload
def tuple_basis[
    *TS,
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    E,
](
    children: tuple[B0, B1], extra_metadata: None = None
) -> TupleBasis2D[Any, B0, B1, None]: ...


@overload
def tuple_basis[
    *TS,
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    E,
](children: tuple[B0, B1], extra_metadata: E) -> TupleBasis2D[Any, B0, B1, E]: ...


@overload
def tuple_basis[
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    *TS,
    E,
](
    children: tuple[B0, B1, B2], extra_metadata: None = None
) -> TupleBasis3D[Any, B0, B1, B2, None]: ...


@overload
def tuple_basis[
    B0: Basis[BasisMetadata, Any],
    B1: Basis[BasisMetadata, Any],
    B2: Basis[BasisMetadata, Any],
    *TS,
    E,
](
    children: tuple[B0, B1, B2], extra_metadata: E
) -> TupleBasis3D[Any, B0, B1, B2, E]: ...


@overload
def tuple_basis[B0, B1, B2, *TS, E](
    children: tuple[*TS], extra_metadata: None = None
) -> TupleBasisND[Any, None]: ...


@overload
def tuple_basis[B0, B1, B2, *TS, E](
    children: tuple[*TS], extra_metadata: E
) -> TupleBasisND[Any, E]: ...


def tuple_basis[B0, B1, B2, *TS, E](
    children: Any, extra_metadata: Any | None = None
) -> Any:
    """Build a VariadicTupleBasis from a tuple."""
    return cast(Any, TupleBasis(children, extra_metadata))


def tuple_basis_is_variadic[M: BasisMetadata, E, DT: np.generic](
    _basis: TupleBasis[M, E, DT],
) -> TypeGuard[TupleBasisND[Any, E]]:
    """Cast a TupleBasis as a VariadicTupleBasis."""
    return True


def tuple_basis_with_modified_children[
    M: BasisMetadata,
    E,
    DT: np.generic,
    DT1: np.generic,
](
    basis: TupleBasis[M, E, DT1], wrapper: Callable[[int, Basis[M, DT1]], Basis[M, DT]]
) -> TupleBasis[M, E, DT]:
    """Get the basis with modified children."""
    return TupleBasis[M, E, DT](
        tuple(starmap(wrapper, enumerate(basis.children))), basis.metadata().extra
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


@overload
def fundamental_tuple_basis_from_metadata[M0: BasisMetadata, E](
    metadata: Metadata1D[M0, E],
) -> TupleBasis1D[np.generic, FundamentalBasis[M0], E]: ...


@overload
def fundamental_tuple_basis_from_metadata[M0: BasisMetadata, M1: BasisMetadata, E](
    metadata: Metadata2D[M0, M1, E],
) -> TupleBasis2D[np.generic, FundamentalBasis[M0], FundamentalBasis[M1], E]: ...


@overload
def fundamental_tuple_basis_from_metadata[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
](
    metadata: Metadata3D[M0, M1, M2, E],
) -> TupleBasis3D[
    np.generic,
    FundamentalBasis[M0],
    FundamentalBasis[M1],
    FundamentalBasis[M2],
    E,
]: ...


@overload
def fundamental_tuple_basis_from_metadata[M: BasisMetadata, E](
    metadata: StackedMetadata[M, E],
) -> TupleBasis[M, E, np.generic]: ...


def fundamental_tuple_basis_from_metadata[M: BasisMetadata, E](
    metadata: StackedMetadata[M, E],
) -> TupleBasis[M, E, np.generic]:
    """Get a basis with the basis at idx set to inner."""
    children = tuple(FundamentalBasis(c) for c in metadata.children)
    return TupleBasis(children, metadata.extra)


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: None = None
) -> TupleBasis1D[np.generic, Basis[SimpleMetadata, np.generic], None]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int], *, extra: E
) -> TupleBasis1D[np.generic, Basis[SimpleMetadata, np.generic], E]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: None = None
) -> TupleBasis2D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    None,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int], *, extra: E
) -> TupleBasis2D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    E,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: None = None
) -> TupleBasis3D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    None,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, int, int], *, extra: E
) -> TupleBasis3D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    E,
]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: None = None
) -> TupleBasis[BasisMetadata, None, np.generic]: ...


@overload
def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: E
) -> TupleBasis[BasisMetadata, E, np.generic]: ...


def fundamental_tuple_basis_from_shape[E](
    shape: tuple[int, ...], *, extra: Any | None = None
) -> Any:
    """Get a basis with the basis at idx set to inner."""
    return fundamental_tuple_basis_from_metadata(
        StackedMetadata.from_shape(shape, extra=extra)
    )


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[Metadata1D[M0, E], DT],
) -> TupleBasis1D[np.generic, Basis[M0, Any], E]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[Metadata2D[M0, M1, E], DT],
) -> TupleBasis2D[np.generic, Basis[M0, Any], Basis[M1, Any], E]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[Metadata3D[M0, M1, M2, E], DT],
) -> TupleBasis3D[np.generic, Basis[M0, Any], Basis[M1, Any], Basis[M1, Any], E]: ...


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[StackedMetadata[M0, E], DT],
) -> TupleBasis[M0, E, np.generic]: ...


def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Any,
) -> Any:
    """Get the closest basis to basis that is a TupleBasis.

    - For a wrapped TupleBasis, this will return the unwrapped basis
    - For a basis which is not a wrapped tuple_basis, this returns the fundamental stacked basis

    This is useful for functions which require a TupleBasis, and ensures that
    only the minimal basis conversion is required.
    """
    super_inner = get_wrapped_basis_super_inner(basis)  # type: ignore unknown
    if isinstance(super_inner, TupleBasis):
        return cast(TupleBasis[M0, E, np.generic], super_inner)

    return fundamental_tuple_basis_from_metadata(
        cast(StackedMetadata[Any, Any], basis.metadata())
    )
