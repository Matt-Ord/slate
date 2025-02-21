from __future__ import annotations

from copy import copy
from typing import (
    Any,
    Never,
    TypeGuard,
    cast,
    overload,
    override,
)

import numpy as np

from slate.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    NestedBool,
    NestedBoolOrNone,
    ctype,
)
from slate.basis._fundamental import FundamentalBasis
from slate.basis.wrapped import (
    get_wrapped_basis_super_inner,
    is_wrapped_basis,
    wrapped_basis_iter_inner,
)
from slate.metadata import (
    AnyMetadata,
    BasisMetadata,
    TupleMetadata,
)
from slate.metadata._metadata import SimpleMetadata


def _convert_tuple_basis_axes[
    C: tuple[Basis[BasisMetadata, ctype[np.generic]], ...],
    E,
    DT: np.generic,
](
    vectors: np.ndarray[Any, np.dtype[DT]],
    initial_basis: TupleBasis[C, E, ctype[np.generic]],
    final_basis: TupleBasis[C, E, ctype[np.generic]],
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
    C: tuple[Basis[BasisMetadata, ctype[np.generic]], ...],
    E,
    DT: np.generic,
](
    vectors: np.ndarray[Any, np.dtype[DT]],
    initial_basis: TupleBasis[C, E, ctype[np.generic]],
    final_basis: TupleBasis[C, E, ctype[np.generic]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT]]:
    """Convert a vector, expressed in terms of the given basis from_config in the basis to_config."""
    if initial_basis.is_dual != final_basis.is_dual:
        # The conversion of a dual to a non-dual vector must happen in an index basis
        initial_fundamental = as_tuple_basis(as_index_basis(initial_basis))
        final_fundamental = as_tuple_basis(as_index_basis(final_basis))
        if initial_fundamental is None or final_fundamental is None:
            msg = "Cannot convert between a basis which is not a TupleBasis"
            raise ValueError(msg)
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
    initial: TupleBasis[Any, Any, ctype[np.generic]],
    vectors: np.ndarray[Any, np.dtype[DT2]],
    final: Basis[Any, ctype[np.generic]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT2]]:
    assert initial.metadata() == final.metadata()

    if initial == final:
        return vectors

    *inner, super_inner = wrapped_basis_iter_inner(final)
    if is_tuple_basis(super_inner):
        basis_as_tuple = super_inner
    else:
        return initial.__convert_vector_into__(vectors, final, axis).ok()

    converted = _convert_tuple_basis_vector_unsafe(
        vectors, initial, basis_as_tuple, axis
    )
    for wrapped_basis in reversed(inner):
        assert is_wrapped_basis(wrapped_basis)
        converted = wrapped_basis.__from_inner__(converted, axis).ok()
    return converted


class TupleBasisMetadata[
    C: tuple[Basis[BasisMetadata, ctype[Never]], ...] = tuple[
        Basis[BasisMetadata, ctype[Never]], ...
    ],
    E = Never,
](TupleMetadata[tuple[BasisMetadata, ...], E]):
    def __init__(self, children: C, extra: E) -> None:
        super().__init__(tuple(i.metadata() for i in children), extra)


class TupleBasis[
    C: tuple[Basis[BasisMetadata, ctype[Never]], ...],
    E,
    DT: ctype[Never],
](Basis[TupleBasisMetadata[C, E], DT]):
    """Represents a Tuple of independent basis."""

    def __init__[C_: tuple[Basis[BasisMetadata, ctype[Never]], ...], E_](
        self: TupleBasis[C_, E_, ctype[Never]],
        children: C_,
        extra: E_ | None = None,
    ) -> None:
        self._children = cast("C", children)
        self._extra = cast("E", extra)

    def upcast[DT_: ctype[Never]](
        self: TupleBasis[tuple[Basis[BasisMetadata, DT_], ...], Any, Any],
    ) -> TupleBasis[C, E, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("TupleBasis[C,E, DT_]", self)

    @override
    def dual_basis(
        self,
    ) -> TupleBasis[C, E, DT]:
        copied = copy(self)
        copied._children = tuple(c.dual_basis() for c in self.children)  # noqa: SLF001
        return cast("TupleBasis[C, E, DT]", copied)

    @property
    @override
    def is_dual(self) -> tuple[NestedBool, ...]:
        return tuple(c.is_dual for c in self.children)

    @property
    def children(self) -> C:
        """Children basis."""
        return self._children

    @override
    def metadata(self) -> TupleBasisMetadata[C, E]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return TupleBasisMetadata(self.children, self._extra)

    @property
    @override
    def size(self) -> int:
        return np.prod(self.shape).item()

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the children data."""
        return tuple(x.size for x in self.children)

    @override
    def __into_fundamental__[DT1: np.generic, DT2: np.generic](
        self: TupleBasis[tuple[Basis[BasisMetadata, ctype[DT1]], ...], Any, ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, np.generic]:
        fundamental = from_metadata(self.metadata())
        return BasisConversion[DT1, DT2, np.generic](
            lambda: _convert_tuple_basis_vector_unsafe(vectors, self, fundamental, axis)  # type: ignore BasisConversion makes this safe
        )

    @override
    def __from_fundamental__[DT2: np.generic, DT3: np.generic](
        self: TupleBasis[tuple[Basis[BasisMetadata, ctype[DT3]], ...], Any, ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, DT2, DT3]:
        fundamental = from_metadata(self.metadata())
        return BasisConversion[np.generic, DT2, DT3](
            lambda: _convert_tuple_basis_vector_unsafe(vectors, fundamental, self, axis)  # type: ignore BasisConversion makes this safe
        )

    @override
    def __convert_vector_into__[
        M_: TupleMetadata[Any, Any],
        DT1: np.generic,
        DT2: np.generic,
        DT3: np.generic,
    ](
        self: Basis[M_, ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        basis: Basis[M_, ctype[DT3]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion[DT1, DT2, DT3](
            lambda: _convert_vectors_unsafe(self, vectors, basis, axis)  # type: ignore BasisConversion makes this safe
        )

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TupleBasis):
            return other.children == self.children  # type: ignore unknown
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
    def add_data[DT1: np.number[Any]](
        self, lhs: np.ndarray[Any, np.dtype[DT1]], rhs: np.ndarray[Any, np.dtype[DT1]]
    ) -> np.ndarray[Any, np.dtype[DT1]]:
        if "LINEAR_MAP" not in self.features:
            msg = "add_data not implemented for this basis"
            raise NotImplementedError(msg)
        return (lhs + rhs).astype(lhs.dtype)

    @override
    def mul_data[DT1: np.number[Any]](
        self,
        lhs: np.ndarray[Any, np.dtype[DT1]],
        rhs: float,
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
            cast("TupleBasis[Any, Any, ctype[np.int_]]", self)
            .__from_fundamental__(np.arange(self.size))
            .ok()
        )


@overload
def from_metadata[M0: BasisMetadata, E](
    metadata: TupleBasisMetadata[tuple[Basis[M0, ctype[Never]]], E],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[tuple[Basis[M0, ctype[np.generic]]], E, ctype[np.generic]]: ...


@overload
def from_metadata[M0: BasisMetadata, M1: BasisMetadata, E](
    metadata: TupleBasisMetadata[
        tuple[
            Basis[M0, ctype[Never]],
            Basis[M1, ctype[Never]],
        ],
        E,
    ],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[
        Basis[M0, ctype[np.generic]],
        Basis[M1, ctype[np.generic]],
    ],
    E,
    ctype[np.generic],
]: ...


@overload
def from_metadata[M0: BasisMetadata, M1: BasisMetadata, M2: BasisMetadata, E](
    metadata: TupleBasisMetadata[
        tuple[
            Basis[M0, ctype[Never]],
            Basis[M1, ctype[Never]],
            Basis[M2, ctype[Never]],
        ],
        E,
    ],
    *,
    is_dual: NestedBoolOrNone = None,
) -> TupleBasis[
    tuple[
        Basis[M0, ctype[np.generic]],
        Basis[M1, ctype[np.generic]],
        Basis[M2, ctype[np.generic]],
    ],
    E,
    ctype[np.generic],
]: ...


@overload
def from_metadata[M: BasisMetadata, E](
    metadata: TupleMetadata[tuple[M, ...], E], *, is_dual: NestedBoolOrNone = None
) -> TupleBasis[tuple[Basis[M, ctype[np.generic]], ...], E, ctype[np.generic]]: ...


@overload
def from_metadata[M: SimpleMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> FundamentalBasis[M]: ...


@overload
def from_metadata[M: AnyMetadata](
    metadata: M, *, is_dual: NestedBoolOrNone = None
) -> Basis[M, ctype[np.generic]]: ...


def from_metadata(
    metadata: AnyMetadata, *, is_dual: NestedBoolOrNone = None
) -> Basis[Any, ctype[np.generic]]:
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
    return TupleBasis(children, metadata.extra).upcast()


def is_tuple_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> TypeGuard[TupleBasis[tuple[Basis[BasisMetadata, ctype[Never]], ...], Never, DT]]:
    return isinstance(basis, TupleBasis)


def as_tuple_basis[
    C: tuple[Basis[BasisMetadata, ctype[Never]], ...],
    E,
    DT: ctype[Never],
](
    basis: Basis[TupleBasisMetadata[C, E], DT],
) -> TupleBasis[C, E, DT] | None:
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
    if is_tuple_basis(super_inner):
        return cast("Any", super_inner)

    return None


def as_feature_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT], features: set[BasisFeature]
) -> Basis[M, DT]:
    """Get the closest basis that supports the feature set."""
    inner_basis = from_metadata(basis.metadata(), is_dual=basis.is_dual)
    for inner_basis in wrapped_basis_iter_inner(basis):
        if features <= inner_basis.features:
            return inner_basis

    if is_tuple_basis(inner_basis):
        return cast(
            "Any",
            TupleBasis(
                tuple(as_feature_basis(b, features) for b in inner_basis.children),
                inner_basis.metadata().extra,
            ),
        )
    return from_metadata(basis.metadata(), is_dual=basis.is_dual)  # type: ignore does not understand everything is generic


def as_index_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports INDEX.

    If the basis is already an INDEX basis, return it.
    If it wraps a INDEX basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"INDEX"})
