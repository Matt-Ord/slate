from __future__ import annotations

from copy import copy
from typing import (
    Any,
    Literal,
    Never,
    TypeGuard,
    cast,
    overload,
    override,
)

import numpy as np

from slate_core.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    Ctype,
    NestedBool,
    NestedBoolOrNone,
    UnionCtype,
)
from slate_core.basis._fundamental import FundamentalBasis
from slate_core.basis._wrapped import (
    AsUpcast,
    get_wrapped_basis_super_inner,
    is_wrapped,
    wrapped_basis_iter_inner,
)
from slate_core.metadata import (
    AnyMetadata,
    BasisMetadata,
    SimpleMetadata,
    TupleMetadata,
    is_tuple_metadata,
)


def _convert_tuple_basis_axes[
    E,
    DT: np.generic,
](
    vectors: np.ndarray[Any, np.dtype[DT]],
    initial_basis: TupleBasis[
        tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], E, Ctype[np.generic]
    ],
    final_basis: TupleBasis[
        tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], E, Ctype[np.generic]
    ],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    swapped = vectors.swapaxes(axis, 0)
    stacked = swapped.reshape(*initial_basis.shape, *swapped.shape[1:])
    for ax, (initial, final) in enumerate(
        zip(initial_basis.children, final_basis.children, strict=True)
    ):
        stacked = initial.__convert_vector_into__(stacked, final, axis=ax).ok()

    return stacked.reshape(-1, *swapped.shape[1:]).swapaxes(axis, 0)


def _convert_tuple_basis_vector_unsafe[
    E,
    DT: np.generic,
](
    vectors: np.ndarray[Any, np.dtype[DT]],
    initial_basis: TupleBasis[
        tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], E, Ctype[np.generic]
    ],
    final_basis: TupleBasis[
        tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], E, Ctype[np.generic]
    ],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    """Convert a vector, expressed in terms of the given basis from_config in the basis to_config."""
    if initial_basis.is_dual != final_basis.is_dual:
        # The conversion of a dual to a non-dual vector must happen in an index basis
        initial_fundamental = as_tuple(as_index(initial_basis))
        final_fundamental = as_tuple(as_index(final_basis))
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


def _convert_vectors_unsafe[DT2: np.generic](
    initial: TupleBasis[Any, Any, Ctype[np.generic]],
    vectors: np.ndarray[Any, np.dtype[DT2]],
    final: Basis[Any, Ctype[np.generic]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT2]]:
    assert initial.metadata() == final.metadata()

    if initial == final:
        return vectors

    *inner, super_inner = wrapped_basis_iter_inner(final)
    if is_tuple(super_inner):
        basis_as_tuple = super_inner
    else:
        return super_inner.__convert_vector_into__(vectors, final, axis).ok()

    converted = _convert_tuple_basis_vector_unsafe(
        vectors, initial, basis_as_tuple, axis
    )
    for wrapped_basis in reversed(inner):
        assert is_wrapped(wrapped_basis)
        converted = wrapped_basis.__from_inner__(converted, axis).ok()
    return converted


class TupleBasis[
    C: tuple[Basis, ...],
    E,
    CT: Ctype[Never] = Ctype[Never],
](Basis[TupleMetadata[tuple[BasisMetadata, ...], E], CT]):
    """Represents a Tuple of independent basis."""

    @overload
    def __init__[C_: tuple[Basis, ...], E_](
        self: TupleBasis[C_, E_, Ctype[Never]],
        children: C_,
        extra: E_,
    ) -> None: ...

    @overload
    def __init__[C_: tuple[Basis, ...]](
        self: TupleBasis[C_, None, Ctype[Never]],
        children: C_,
        extra: None = None,
    ) -> None: ...

    def __init__[C_: tuple[Basis, ...], E_](
        self: TupleBasis[C_, E_, Ctype[Never]],
        children: C_,
        extra: E_ | None = None,
    ) -> None:
        self._children = cast("C", children)
        self._extra = cast("E", extra)

    @property
    @override
    def ctype(self) -> CT:
        return cast(
            "CT",
            UnionCtype(tuple(c.ctype for c in self.children)),
        )

    def resolve_ctype[DT_: Ctype[Never]](
        self: TupleBasis[tuple[Basis[BasisMetadata, DT_], ...], Any, Any],
    ) -> TupleBasis[C, E, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("TupleBasis[C,E, DT_]", self)

    @overload
    def upcast[M0: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M0]], E_, Any],
    ) -> AsUpcast[TupleBasis[C, E, CT], TupleMetadata[tuple[M0], E], CT]: ...
    @overload
    def upcast[M0: BasisMetadata, M1: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M0], Basis[M1]], E_, Any],
    ) -> AsUpcast[TupleBasis[C, E, CT], TupleMetadata[tuple[M0, M1], E], CT]: ...
    @overload
    def upcast[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M0], Basis[M1], Basis[M2]], E_, Any],
    ) -> AsUpcast[TupleBasis[C, E, CT], TupleMetadata[tuple[M0, M1, M2], E], CT]: ...
    @overload
    def upcast[M_: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M_], ...], E_, Any],
    ) -> AsUpcast[TupleBasis[C, E, CT], TupleMetadata[tuple[M_, ...], E], CT]: ...

    def upcast(
        self,
    ) -> AsUpcast[TupleBasis[C, E, CT], TupleMetadata[tuple[Any, ...], Any], CT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

    @override
    def dual_basis(
        self,
    ) -> TupleBasis[C, E, CT]:
        copied = copy(self)
        copied._children = tuple(c.dual_basis() for c in self.children)  # noqa: SLF001
        return cast("TupleBasis[C, E, CT]", copied)

    @property
    @override
    def is_dual(self) -> tuple[NestedBool, ...]:
        return tuple(c.is_dual for c in self.children)

    @property
    def children(self) -> C:
        """Children basis."""
        return self._children

    @overload
    def metadata[M0: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M0]], E_, Any],
    ) -> TupleMetadata[tuple[M0], E]: ...
    @overload
    def metadata[M0: BasisMetadata, M1: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M0], Basis[M1]], E_, Any],
    ) -> TupleMetadata[tuple[M0, M1], E]: ...
    @overload
    def metadata[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M0], Basis[M1], Basis[M2]], E_, Any],
    ) -> TupleMetadata[tuple[M0, M1, M2], E]: ...
    @overload
    def metadata[M: BasisMetadata, E_](
        self: TupleBasis[tuple[Basis[M], ...], E_, Any],
    ) -> TupleMetadata[tuple[M, ...], E]: ...

    @override
    def metadata(self) -> TupleMetadata[tuple[BasisMetadata, ...], E]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return TupleMetadata(tuple(c.metadata() for c in self.children), self._extra)

    @property
    @override
    def size(self) -> int:
        return np.prod(self.shape).item()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the children data."""
        return tuple(x.size for x in self.children)

    @override
    def __into_fundamental__[T1: np.generic, T2: np.generic](
        self: TupleBasis[tuple[Basis[BasisMetadata, Ctype[T1]], ...], Any, Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, np.generic]:
        fundamental = from_metadata(self.metadata())
        return BasisConversion[T1, T2, np.generic](
            lambda: _convert_tuple_basis_vector_unsafe(vectors, self, fundamental, axis)  # type: ignore BasisConversion makes this safe
        )

    @override
    def __from_fundamental__[T2: np.generic, T3: np.generic](
        self: TupleBasis[tuple[Basis[BasisMetadata, Ctype[T3]], ...], Any, Ctype[T3]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, T2, T3]:
        fundamental = from_metadata(self.metadata())
        return BasisConversion[np.generic, T2, T3](
            lambda: _convert_tuple_basis_vector_unsafe(vectors, fundamental, self, axis)  # type: ignore BasisConversion makes this safe
        )

    @override
    def __convert_vector_into__[
        M_: BasisMetadata,
        T1: np.generic,
        T2: np.generic,
        T3: np.generic,
    ](
        self: Basis[M_, Ctype[T1]],
        vectors: np.ndarray[Any, np.dtype[T2]],
        basis: Basis[M_, Ctype[T3]],
        axis: int = -1,
    ) -> BasisConversion[T1, T2, T3]:
        return BasisConversion[T1, T2, T3](
            lambda: _convert_vectors_unsafe(self, vectors, basis, axis)  # type: ignore BasisConversion makes this safe
        )

    @override
    def __eq__(self, other: object) -> bool:
        if is_tuple(other):
            return other.children == self.children
        if is_wrapped(other):
            # Check if the inner basis is equal
            return other == self  # type: ignore this is simply not true?
        return False

    @override
    def __hash__(self) -> int:
        return hash((self._extra, self.children))

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if all("LINEAR_MAP" in c.features for c in self.children):
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        if all("INDEX" in c.features for c in self.children):
            out.add("INDEX")
        return out

    @override
    def add_data[T: np.number](
        self, lhs: np.ndarray[Any, np.dtype[T]], rhs: np.ndarray[Any, np.dtype[T]]
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
        self, lhs: np.ndarray[Any, np.dtype[T]], rhs: np.ndarray[Any, np.dtype[T]]
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
            cast("TupleBasis[Any, Any, Ctype[np.int_]]", self)
            .__from_fundamental__(np.arange(self.size))
            .ok()
        )


@overload
def from_metadata[M0: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M0], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[tuple[Basis[M0, Ctype[np.generic]]], E, Ctype[np.generic]]: ...


@overload
def from_metadata[M0: BasisMetadata, M1: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M0, M1], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.generic]],
        Basis[M1, Ctype[np.generic]],
    ],
    E,
    Ctype[np.generic],
]: ...


@overload
def from_metadata[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M0, M1, M2], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[
        Basis[M0, Ctype[np.generic]],
        Basis[M1, Ctype[np.generic]],
        Basis[M2, Ctype[np.generic]],
    ],
    E,
    Ctype[np.generic],
]: ...


@overload
def from_metadata[M: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M, ...], E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[tuple[Basis[M, Ctype[np.generic]], ...], E, Ctype[np.generic]]: ...


@overload
def from_metadata[M: SimpleMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> FundamentalBasis[M]: ...


@overload
def from_metadata[M: AnyMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> Basis[M, Ctype[np.generic]]: ...


def from_metadata(
    metadata: AnyMetadata, *, is_dual: NestedBoolOrNone = None
) -> Basis[Any, Ctype[np.generic]]:
    """Get a basis with the basis at idx set to inner."""
    if isinstance(metadata, SimpleMetadata):
        is_dual = False if is_dual is None else is_dual
        assert isinstance(is_dual, bool)
        return FundamentalBasis(metadata, is_dual=is_dual)

    metadata = cast("TupleMetadata[tuple[AnyMetadata, ...], Any]", metadata)
    is_dual = (
        is_dual
        if isinstance(is_dual, tuple)
        else tuple(is_dual for _ in metadata.children)
    )

    children = tuple(
        from_metadata(c, is_dual=dual)
        for (c, dual) in zip(metadata.children, is_dual, strict=False)
    )
    return TupleBasis(children, metadata.extra).resolve_ctype()


@overload
def is_tuple[M0: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M0], E, CT],
) -> TypeGuard[TupleBasis[tuple[Basis[M0, CT]], E, CT]]: ...
@overload
def is_tuple[M0: BasisMetadata, M1: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M0, M1], E, CT],
) -> TypeGuard[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E, CT]]: ...
@overload
def is_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    CT: Ctype[Never],
](
    basis: TupleBasisLike[tuple[M0, M1, M2], E, CT],
) -> TypeGuard[
    TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT], Basis[M2, CT]], E, CT]
]: ...
@overload
def is_tuple[M: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M, ...], E, CT],
) -> TypeGuard[TupleBasis[tuple[Basis[M, CT], ...], E, CT]]: ...
@overload
def is_tuple[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> TypeGuard[TupleBasis[tuple[Basis[BasisMetadata, CT], ...], Any, CT]]: ...
@overload
def is_tuple(
    basis: object,
) -> TypeGuard[TupleBasis[tuple[Basis, ...], Any]]: ...


def is_tuple(
    basis: object,
) -> TypeGuard[TupleBasis[tuple[Basis, ...], Any]]:
    return isinstance(basis, TupleBasis)


type TupleBasis1D[
    C: tuple[Basis] = tuple[Basis],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = AsUpcast[TupleBasis[C, E, CT], TupleMetadata[tuple[BasisMetadata], E], CT]
type TupleBasis2D[
    C: tuple[Basis, Basis] = tuple[Basis, Basis],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = AsUpcast[
    TupleBasis[C, E, CT], TupleMetadata[tuple[BasisMetadata, BasisMetadata], E], CT
]
type TupleBasis3D[
    C: tuple[Basis, Basis, Basis] = tuple[Basis, Basis, Basis],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = AsUpcast[
    TupleBasis[C, E, CT],
    TupleMetadata[tuple[BasisMetadata, BasisMetadata, BasisMetadata], E],
    CT,
]

type TupleBasisLike[
    M: tuple[BasisMetadata, ...] = tuple[BasisMetadata, ...],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = Basis[TupleMetadata[M, E], CT]

type TupleBasisLike1D[
    M: tuple[BasisMetadata] = tuple[BasisMetadata],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = TupleBasisLike[M, E, CT]
type TupleBasisLike2D[
    M: tuple[BasisMetadata, BasisMetadata] = tuple[BasisMetadata, BasisMetadata],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = TupleBasisLike[M, E, CT]
type TupleBasisLike3D[
    M: tuple[BasisMetadata, BasisMetadata, BasisMetadata] = tuple[
        BasisMetadata, BasisMetadata, BasisMetadata
    ],
    E = Any,
    CT: Ctype[Never] = Ctype[Any],
] = TupleBasisLike[M, E, CT]


@overload
def is_tuple_basis_like[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT], *, n_dim: Literal[1]
) -> TypeGuard[TupleBasisLike[tuple[BasisMetadata], Never, CT]]: ...
@overload
def is_tuple_basis_like[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT], *, n_dim: Literal[2]
) -> TypeGuard[TupleBasisLike[tuple[BasisMetadata, BasisMetadata], Never, CT]]: ...
@overload
def is_tuple_basis_like[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT], *, n_dim: Literal[3]
) -> TypeGuard[
    TupleBasisLike[tuple[BasisMetadata, BasisMetadata, BasisMetadata], Never, CT]
]: ...
@overload
def is_tuple_basis_like[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT], *, n_dim: int | None = None
) -> TypeGuard[TupleBasisLike[tuple[BasisMetadata, ...], Never, CT]]: ...


def is_tuple_basis_like[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT], *, n_dim: int | None = None
) -> TypeGuard[TupleBasisLike[tuple[BasisMetadata, ...], Never, CT]]:
    return is_tuple_metadata(basis.metadata(), n_dim=n_dim)


@overload
def as_tuple[M0: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M0], E, CT],
) -> TupleBasis[tuple[Basis[M0, CT]], E, CT]: ...


@overload
def as_tuple[M0: BasisMetadata, M1: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M0, M1], E, CT],
) -> TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E, CT]: ...


@overload
def as_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    M2: BasisMetadata,
    E,
    CT: Ctype[Never],
](
    basis: TupleBasisLike[tuple[M0, M1, M2], E, CT],
) -> TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT], Basis[M2, CT]], E, CT]: ...


@overload
def as_tuple[M: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[M, ...], E, CT],
) -> TupleBasis[tuple[Basis[M, CT], ...], E, CT]: ...


@overload
def as_tuple[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT],
) -> TupleBasis[tuple[Basis[BasisMetadata, CT], ...], Any, CT]: ...


def as_tuple[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT],
) -> TupleBasis[tuple[Basis[BasisMetadata, CT], ...], Any, CT]:
    """Get the closest basis to basis that is a TupleBasis.

    - For a wrapped TupleBasis, this will return the unwrapped basis
    - For a basis which is not a wrapped tuple_basis, this returns the fundamental stacked basis

    This is useful for functions which require a TupleBasis, and ensures that
    only the minimal basis conversion is required.

    Raises
    ------
        ValueError: if the basis is not a TupleBasis
    """
    super_inner = get_wrapped_basis_super_inner(basis)
    if is_tuple(super_inner):
        return super_inner

    return from_metadata(basis.metadata())  # type: ignore This is ok, since np.generic is a subtype of DT


def as_feature[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT], features: set[BasisFeature]
) -> Basis[M, CT]:
    """Get the closest basis that supports the feature set."""
    inner_basis = from_metadata(basis.metadata(), is_dual=basis.is_dual)
    for inner_basis in wrapped_basis_iter_inner(basis):
        if features <= inner_basis.features:
            return inner_basis

    if is_tuple(inner_basis):
        return cast(
            "Any",
            TupleBasis(
                tuple(as_feature(b, features) for b in inner_basis.children),
                inner_basis.metadata().extra,
            ),
        )
    return from_metadata(basis.metadata(), is_dual=basis.is_dual)  # type: ignore does not understand everything is generic


def as_index[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> Basis[M, CT]:
    """Get the closest basis that supports INDEX.

    If the basis is already an INDEX basis, return it.
    If it wraps a INDEX basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature(basis, {"INDEX"})


def as_supports_type[M: BasisMetadata, T: np.generic](
    basis: Basis[M], ty: type[T]
) -> Basis[M, Ctype[T]]:
    """Get the closest basis that supports the type ty.

    Note that the type displayed will be incorrect if ty is porided as a
    union - this should only be called by a literal value for ty.
    """
    inner_basis = from_metadata(basis.metadata(), is_dual=basis.is_dual)
    for inner_basis in wrapped_basis_iter_inner(basis):
        if inner_basis.ctype.supports_type(ty):
            return cast("Basis[M, Ctype[T]]", inner_basis)
    if is_tuple(inner_basis):
        return cast(
            "Basis[M, Ctype[T]]",
            TupleBasis(
                tuple(as_supports_type(b, ty) for b in inner_basis.children),
                inner_basis.metadata().extra,
            ),
        )
    return from_metadata(basis.metadata(), is_dual=basis.is_dual)
