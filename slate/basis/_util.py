from __future__ import annotations

from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeGuard,
    cast,
    overload,
)

import numpy as np

from slate.basis._basis import Basis, NestedBool
from slate.basis._tuple import (
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    TupleBasisND,
    as_feature_basis,
    as_tuple_basis,
    from_metadata,
    tuple_basis,
)
from slate.basis.wrapped import (
    wrapped_basis_iter_inner,
)
from slate.metadata import (
    AnyMetadata,
    BasisMetadata,
    Metadata1D,
    NestedLength,
    SimpleMetadata,
    StackedMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@overload
def tuple_basis_is_variadic[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], *, n_dim: Literal[1]
) -> TypeGuard[TupleBasis1D[DT, Basis[M, DT], E]]: ...
@overload
def tuple_basis_is_variadic[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], *, n_dim: Literal[2]
) -> TypeGuard[TupleBasis2D[DT, Basis[M, DT], Basis[M, DT], E]]: ...
@overload
def tuple_basis_is_variadic[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], *, n_dim: Literal[3]
) -> TypeGuard[TupleBasis3D[DT, Basis[M, DT], Basis[M, DT], Basis[M, DT], E]]: ...
@overload
def tuple_basis_is_variadic[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], *, n_dim: int | None = None
) -> TypeGuard[TupleBasisND[Any, E]]: ...


def tuple_basis_is_variadic(
    basis: TupleBasis[Any, Any, Any], *, n_dim: int | None = None
) -> TypeGuard[Any]:
    """Cast a TupleBasis as a VariadicTupleBasis."""
    return n_dim is None or len(basis.shape) == n_dim


def with_modified_children[
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


def with_modified_child[
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
    return with_modified_children(
        basis, lambda i, b: cast("Basis[Any, Any]", b if i != idx else wrapper(b))
    )


def with_child[M: BasisMetadata, E, DT: np.generic](
    basis: TupleBasis[M, E, DT], inner: Basis[M, DT], idx: int
) -> TupleBasis[M, E, DT]:
    """Get a basis with the basis at idx set to inner."""
    return with_modified_child(basis, lambda _: inner, idx)


def as_fundamental[M: AnyMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, np.generic]:
    return from_metadata(basis.metadata(), is_dual=basis.is_dual)


@overload
def from_shape[E](
    shape: tuple[int], *, extra: None = None, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis1D[np.generic, Basis[SimpleMetadata, np.generic], None]: ...


@overload
def from_shape[E](
    shape: tuple[int], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis1D[np.generic, Basis[SimpleMetadata, np.generic], E]: ...


@overload
def from_shape[E](
    shape: tuple[int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis2D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    None,
]: ...


@overload
def from_shape[E](
    shape: tuple[int, int], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis2D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    E,
]: ...


@overload
def from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis3D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    None,
]: ...


@overload
def from_shape[E](
    shape: tuple[int, int, int], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis3D[
    np.generic,
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    Basis[SimpleMetadata, np.generic],
    E,
]: ...


@overload
def from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[BasisMetadata, None, np.generic]: ...


@overload
def from_shape[E](
    shape: tuple[int, ...], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[BasisMetadata, E, np.generic]: ...


@overload
def from_shape[E](
    shape: NestedLength, *, extra: None = None, is_dual: tuple[bool, ...] | None = None
) -> Basis[BasisMetadata, np.generic]: ...


def from_shape[E](
    shape: NestedLength,
    *,
    extra: Any | None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> Any:
    """Get a basis with the basis at idx set to inner."""
    return from_metadata(
        StackedMetadata.from_shape(shape, extra=extra), is_dual=is_dual
    )


def get_common_basis[M: BasisMetadata, E, DT: np.generic](
    lhs: Basis[M, DT],
    rhs: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest common basis of two bases."""
    assert rhs.metadata() == lhs.metadata()
    lhs_rev = list(wrapped_basis_iter_inner(lhs))
    rhs_rev = list(wrapped_basis_iter_inner(rhs))

    if (
        isinstance(lhs_rev[-1], TupleBasis)
        and isinstance(rhs_rev[-1], TupleBasis)
        and lhs_rev != rhs_rev
    ):
        # For a TupleBasis, we can do a bit better
        # By finding the common basis of the children
        lhs_children = cast("tuple[Basis[Any, Any], ...]", lhs_rev[-1].children)  # type: ignore unknown
        rhs_children = cast("tuple[Basis[Any, Any], ...]", rhs_rev[-1].children)  # type: ignore unknown

        basis = tuple_basis(
            tuple(
                starmap(get_common_basis, zip(lhs_children, rhs_children, strict=False))
            ),
            cast("StackedMetadata[Any, Any]", rhs.metadata()).extra,
        )
        return cast("Basis[M, DT]", basis)

    last_common = from_metadata(rhs.metadata(), is_dual=rhs.is_dual)
    for a, b in zip(reversed(lhs_rev), reversed(rhs_rev), strict=False):
        if a != b:
            return last_common
        last_common = a
    return last_common


@overload
def flatten[B: Basis[Any, Any], DT: np.generic](
    basis: TupleBasis1D[Any, B, Any],
) -> B: ...


@overload
def flatten[M: BasisMetadata, DT: np.generic](
    basis: Basis[Metadata1D[M, Any], DT],
) -> Basis[M, DT]: ...


@overload
def flatten[M: BasisMetadata, DT: np.generic](
    basis: Basis[StackedMetadata[StackedMetadata[M, Any], Any], DT],
) -> TupleBasis[M, None, DT]: ...


def flatten(
    basis: Basis[BasisMetadata, Any],
) -> Basis[Any, Any]:
    as_tuple = as_tuple_basis(basis)
    if len(as_tuple.shape) == 1:
        return as_tuple.children[0]
    children = tuple(c for b in as_tuple.children for c in as_tuple_basis(b).children)
    return tuple_basis(children)


def as_add_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports addition.

    If the basis is already an ADD basis, return it.
    If it wraps an ADD basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"ADD"})


def as_sub_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports subtraction.

    If the basis is already a SUB basis, return it.
    If it wraps a SUB basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"SUB"})


def as_mul_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports MUL.

    If the basis is already a MUL basis, return it.
    If it wraps a MUL basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"MUL"})


def as_linear_map_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports LINEAR_MAP.

    If the basis is already a LINEAR_MAP basis, return it.
    If it wraps a LINEAR_MAP basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"LINEAR_MAP"})


def as_is_dual_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT], is_dual: NestedBool
) -> Basis[M, DT]:
    """Get the closest basis that supports IS_DUAL.

    If the basis is already an IS_DUAL basis, return it.
    If it wraps an IS_DUAL basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    if is_dual == basis.is_dual:
        return basis
    if isinstance(is_dual, bool):
        assert isinstance(basis.is_dual, bool)
        return basis.dual_basis()
    assert isinstance(basis.is_dual, tuple)

    basis_as_tuple = as_tuple_basis(
        cast("Basis[StackedMetadata[Any, Any], Any]", basis)
    )
    return cast(
        "Basis[M, DT]",
        tuple_basis(
            tuple(
                starmap(
                    as_is_dual_basis,
                    zip(basis_as_tuple.children, is_dual, strict=False),
                )
            ),
            basis_as_tuple.metadata().extra,
        ),
    )
