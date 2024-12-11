from __future__ import annotations

from copy import copy
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Never,
    Self,
    cast,
    overload,
    override,
)

import numpy as np

from slate.basis._basis import Basis, BasisFeature, NestedBool, NestedBoolOrNone
from slate.basis._fundamental import FundamentalBasis
from slate.basis.wrapped import (
    WrappedBasis,
    get_wrapped_basis_super_inner,
    wrapped_basis_iter_inner,
)
from slate.metadata import (
    AnyMetadata,
    BasisMetadata,
    Metadata1D,
    Metadata2D,
    Metadata3D,
    MetadataND,
    StackedMetadata,
)
from slate.metadata._metadata import SimpleMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator


type StackedBasis[M: BasisMetadata, E, DT: np.generic] = Basis[
    StackedMetadata[M, E], DT
]


def _convert_tuple_basis_axes[M: BasisMetadata, E, DT: np.generic](
    vectors: np.ndarray[Any, np.dtype[DT]],
    initial_basis: TupleBasis[M, E, DT],
    final_basis: TupleBasis[M, E, DT],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    swapped = vectors.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(zip(initial_basis, final_basis, strict=True)):
        stacked = initial.__convert_vector_into__(stacked, final, axis=ax)

    return stacked.reshape(-1, *swapped.shape[1:]).swapaxes(axis, 0)


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
    if initial_basis.is_dual != final_basis.is_dual:
        # The conversion of a dual to a non-dual vector must happen in an index basis
        initial_fundamental = as_tuple_basis(as_index_basis(initial_basis))
        final_fundamental = as_tuple_basis(as_index_basis(final_basis))
        converted_0 = _convert_tuple_basis_axes(
            vectors, initial_basis, initial_fundamental, axis
        )
        converted_1 = _convert_tuple_basis_axes(
            converted_0, initial_fundamental, final_fundamental, axis
        )
        return _convert_tuple_basis_axes(
            converted_1, final_fundamental, final_basis, axis
        )

    return _convert_tuple_basis_axes(vectors, initial_basis, final_basis, axis)


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
        self,
        children: tuple[Basis[M, Any], ...],
        extra_metadata: E,
    ) -> None:
        self._children = children

        super().__init__(
            cast(
                "_InnerM",
                StackedMetadata(tuple(i.metadata() for i in children), extra_metadata),
            ),
        )

    @override
    def dual_basis(self) -> Self:
        copied = copy(self)
        copied._children = tuple(c.dual_basis() for c in self.children)  # noqa: SLF001
        return copied

    @property
    @override
    def is_dual(self) -> tuple[NestedBool, ...]:
        return tuple(c.is_dual for c in self.children)

    @property
    def children(self) -> tuple[Basis[M, Any], ...]:
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
        fundamental = from_metadata(self.metadata())
        return _convert_tuple_basis_vector(
            vectors,
            cast("TupleBasis[M, E, DT1]", self),
            cast("TupleBasis[M, E, DT1]", fundamental),
            axis,
        )

    @override
    def __from_fundamental__[DT1: np.generic](  # [DT1: DT]
        self,
        vectors: np.ndarray[Any, np.dtype[DT1]],
        axis: int = -1,
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        basis = from_metadata(self.metadata())
        return _convert_tuple_basis_vector(
            vectors,
            cast("TupleBasis[M, E, DT1]", basis),
            cast("TupleBasis[M, E, DT1]", self),
            axis,
        )

    def __iter__(self) -> Iterator[Basis[M, DT]]:
        return self.children.__iter__()

    def __getitem__(self, index: int) -> Basis[M, DT]:
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

        *inner, super_inner = wrapped_basis_iter_inner(basis)  # type: ignore unknown
        if isinstance(super_inner, TupleBasis):
            basis_as_tuple = cast("TupleBasis[M, E, np.generic]", super_inner)
        else:
            return super().__convert_vector_into__(vectors, basis, axis)

        converted = _convert_tuple_basis_vector(vectors, self, basis_as_tuple, axis)  # type: ignore unknown
        for wrapped_basis in reversed(inner):
            converted = cast("WrappedBasis[Any, Any]", wrapped_basis).__from_inner__(
                converted, axis
            )
        return converted

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TupleBasis):
            return other.children == self.children  # type: ignore unknown
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
        self, children: tuple[Basis[BasisMetadata, DT], ...], extra_metadata: E
    ) -> None:
        msg = "TupleBasisND is only a placeholder type, and should not be directly created."
        raise NotImplementedError(msg)
        super().__init__(
            cast("tuple[Basis[BasisMetadata, DT], ...]", children), extra_metadata
        )

    @override
    def metadata(self) -> _InnerM:
        return super().metadata()

    @property
    def children(self) -> tuple[Basis[BasisMetadata, DT], ...]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast("tuple[Basis[BasisMetadata, DT], ...]", super().children)

    @override
    def __getitem__(self, index: int) -> Basis[BasisMetadata, DT]:
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
            cast("tuple[Basis[BasisMetadata, _DT]]", children), extra_metadata
        )

    @override
    def metadata[_M0: BasisMetadata](
        self: TupleBasis1D[Any, Basis[_M0, Any], E],
    ) -> Metadata1D[_M0, E]:
        return cast("Metadata1D[_M0, E]", super().metadata())

    @property
    def children(self) -> tuple[B0]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast("tuple[B0]", super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> B0: ...

    @overload
    def __getitem__(self, index: int) -> B0: ...

    def __getitem__(self, index: int) -> B0:  # type: ignore inconsistent type
        return cast("B0", super().__getitem__(index))


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
        return cast("Metadata2D[_M0, _M1, E]", super().metadata())

    @property
    def children(self) -> tuple[B0, B1]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast("tuple[B0, B1]", super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> B0: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> B1: ...

    @overload
    def __getitem__(self, index: int) -> B0 | B1: ...

    def __getitem__(self, index: int) -> B0 | B1:  # type: ignore overload
        return cast("B0 | B1", super().__getitem__(index))


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
        return cast("Metadata3D[_M0, _M1, _M2, E]", super().metadata())

    @property
    def children(self) -> tuple[B0, B1, B2]:  # type: ignore inconsistent type
        """Inner basis."""
        return cast("tuple[B0, B1, B2]", super().children)

    @overload
    def __getitem__(self, index: Literal[0]) -> B0: ...

    @overload
    def __getitem__(self, index: Literal[1]) -> B1: ...

    @overload
    def __getitem__(self, index: Literal[2]) -> B2: ...

    @overload
    def __getitem__(self, index: int) -> B0 | B1 | B2: ...

    def __getitem__(self, index: int) -> B0 | B1 | B2:  # type: ignore overload
        return cast("B0 | B1 | B2", super().__getitem__(index))


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
    return cast("Any", TupleBasis(children, extra_metadata))


@overload
def from_metadata[M0: SimpleMetadata, E](
    metadata: Metadata1D[M0, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis1D[np.generic, FundamentalBasis[M0], E]: ...


@overload
def from_metadata[M0: SimpleMetadata, M1: SimpleMetadata, E](
    metadata: Metadata2D[M0, M1, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[np.generic, FundamentalBasis[M0], FundamentalBasis[M1], E]: ...


@overload
def from_metadata[M0: BasisMetadata, M1: BasisMetadata, E](
    metadata: Metadata2D[M0, M1, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis2D[np.generic, Basis[M0, np.generic], Basis[M1, np.generic], E]: ...


@overload
def from_metadata[
    M0: SimpleMetadata,
    M1: SimpleMetadata,
    M2: SimpleMetadata,
    E,
](
    metadata: Metadata3D[M0, M1, M2, E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis3D[
    np.generic,
    FundamentalBasis[M0],
    FundamentalBasis[M1],
    FundamentalBasis[M2],
    E,
]: ...


@overload
def from_metadata[M: AnyMetadata, E](
    metadata: StackedMetadata[M, E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[M, E, np.generic]: ...


@overload
def from_metadata[M: SimpleMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> FundamentalBasis[M]: ...


@overload
def from_metadata[M: AnyMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> Basis[M, np.generic]: ...


def from_metadata(
    metadata: AnyMetadata, *, is_dual: NestedBoolOrNone = None
) -> Basis[Any, np.generic]:
    """Get a basis with the basis at idx set to inner."""
    if isinstance(metadata, SimpleMetadata):
        is_dual = False if is_dual is None else is_dual
        assert isinstance(is_dual, bool)
        return FundamentalBasis(metadata, is_dual=is_dual)

    metadata = cast("StackedMetadata[AnyMetadata, Any]", metadata)
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(is_dual for _ in metadata.children)
    )

    children = tuple(
        from_metadata(c, is_dual=dual) for (c, dual) in zip(metadata.children, is_dual)
    )
    return TupleBasis(children, metadata.extra)


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


@overload
def as_tuple_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    DT: np.generic,
](
    basis: Basis[BasisMetadata, DT],
) -> TupleBasis[BasisMetadata, Any, np.generic]: ...


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
        return cast("TupleBasis[M0, E, np.generic]", super_inner)

    return from_metadata(cast("StackedMetadata[Any, Any]", basis.metadata()))


def as_feature_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT], features: set[BasisFeature]
) -> Basis[M, DT]:
    """Get the closest basis that supports the feature set."""
    inner_basis = from_metadata(basis.metadata(), is_dual=basis.is_dual)
    for inner_basis in wrapped_basis_iter_inner(basis):
        if features <= inner_basis.features:
            return inner_basis
    if isinstance(inner_basis, TupleBasis):
        inner_basis = cast("TupleBasis[Any, Any, Any]", inner_basis)
        return cast(
            "Basis[M, DT]",
            tuple_basis(
                tuple(as_feature_basis(b, features) for b in inner_basis),
                inner_basis.metadata().extra,
            ),
        )
    return inner_basis


def as_index_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports INDEX.

    If the basis is already an INDEX basis, return it.
    If it wraps a INDEX basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"INDEX"})
