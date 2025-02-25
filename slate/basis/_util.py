from __future__ import annotations

from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    cast,
    overload,
)

from slate.basis._basis import Basis, NestedBool, ctype
from slate.basis._tuple import (
    TupleBasis,
    TupleBasisLike,
    as_feature_basis,
    from_metadata,
    is_tuple_basis_like,
)
from slate.basis.wrapped import (
    wrapped_basis_iter_inner,
)
from slate.metadata import (
    AnyMetadata,
    BasisMetadata,
    NestedLength,
    TupleMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from slate.basis._fundamental import FundamentalBasis


def with_modified_children[
    M: BasisMetadata,
    E,
    DT0: ctype[Never],
    DT1: ctype[Never],
](
    basis: TupleBasis[tuple[Basis[M, DT0], ...], E, DT1],
    wrapper: Callable[[int, Basis[M, DT0]], Basis[M, DT0]],
) -> TupleBasis[tuple[Basis[M, DT0], ...], E, DT0]:
    """Get the basis with modified children."""
    return TupleBasis[tuple[Basis[M, DT0], ...], E, DT0](
        tuple(starmap(wrapper, enumerate(basis.children))), basis.metadata().extra
    ).upcast()


def with_modified_child[
    M: BasisMetadata,
    E,
    DT0: ctype[Never],
    DT1: ctype[Never],
](
    basis: TupleBasis[tuple[Basis[M, DT0], ...], E, DT1],
    wrapper: Callable[[Basis[M, DT0]], Basis[M, DT0]],
    idx: int,
) -> TupleBasis[tuple[Basis[M, DT0], ...], E, DT0]:
    """Get the basis with modified child."""
    return with_modified_children(
        basis, lambda i, b: cast("Basis", b if i != idx else wrapper(b))
    )


def with_child[M: BasisMetadata, E, DT: ctype[Never]](
    basis: TupleBasis[tuple[Basis[M, DT], ...], E, DT], inner: Basis[M, DT], idx: int
) -> TupleBasis[tuple[Basis[M, DT], ...], E, DT]:
    """Get a basis with the basis at idx set to inner."""
    return with_modified_child(basis, lambda _: inner, idx)


def as_fundamental[M: AnyMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> Basis[M, ctype[np.generic]]:
    return from_metadata(basis.metadata(), is_dual=basis.is_dual)


@overload
def from_shape[E](
    shape: tuple[int], *, extra: None = None, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[tuple[FundamentalBasis], None, ctype[np.generic]]: ...
@overload
def from_shape[E](
    shape: tuple[int], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[tuple[FundamentalBasis], E, ctype[np.generic]]: ...


@overload
def from_shape[E](
    shape: tuple[int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, FundamentalBasis], None, ctype[np.generic]]: ...
@overload
def from_shape[E](
    shape: tuple[int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, FundamentalBasis], E, ctype[np.generic]]: ...


@overload
def from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis], None, ctype[np.generic]
]: ...
@overload
def from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis], E, ctype[np.generic]
]: ...


@overload
def from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, ...], None, ctype[np.generic]]: ...
@overload
def from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, ...], E, ctype[np.generic]]: ...


@overload
def from_shape[E](
    shape: tuple[NestedLength, ...],
    *,
    extra: None = None,
    is_dual: tuple[NestedBool, ...] | None = None,
) -> TupleBasis[
    tuple[Basis[BasisMetadata, ctype[np.generic]], ...], None, ctype[np.generic]
]: ...
@overload
def from_shape[E](
    shape: tuple[NestedLength, ...],
    *,
    extra: E,
    is_dual: tuple[NestedBool, ...] | None = None,
) -> TupleBasis[
    tuple[Basis[BasisMetadata, ctype[np.generic]], ...], E, ctype[np.generic]
]: ...


@overload
def from_shape[E](
    shape: int,
    *,
    extra: None = None,
    is_dual: bool | None = None,
) -> TupleBasis[
    tuple[Basis[BasisMetadata, ctype[np.generic]], ...], None, ctype[np.generic]
]: ...


def from_shape[E](
    shape: NestedLength,
    *,
    extra: Any | None = None,
    is_dual: NestedBool | None = None,
) -> Basis[Any, ctype[np.generic]]:
    """Get a basis from the shape provided."""
    return from_metadata(TupleMetadata.from_shape(shape, extra=extra), is_dual=is_dual)


def get_common_basis[M: BasisMetadata, DT: ctype[Never]](
    lhs: Basis[M, DT],
    rhs: Basis[M, DT],
) -> Basis[M, DT | ctype[np.generic]]:
    """Get the closest common basis of two bases."""
    assert rhs.metadata() == lhs.metadata()
    lhs_rev = list(wrapped_basis_iter_inner(lhs))
    rhs_rev = list(wrapped_basis_iter_inner(rhs))

    if (
        is_tuple_basis(lhs_rev[-1])
        and is_tuple_basis(rhs_rev[-1])
        and lhs_rev != rhs_rev
    ):
        # For a TupleBasis, we can do a bit better
        # By finding the common basis of the children
        lhs_children = lhs_rev[-1].children
        rhs_children = rhs_rev[-1].children
        children = zip(lhs_children, rhs_children, strict=True)

        basis = TupleBasis(
            tuple(starmap(get_common_basis, children)),
            rhs_rev[-1].metadata().extra,
        )
        return cast("Basis[M, DT]", basis)

    last_common = from_metadata(rhs.metadata(), is_dual=rhs.is_dual)
    for a, b in zip(reversed(lhs_rev), reversed(rhs_rev), strict=False):
        if a != b:
            return last_common
        last_common = a
    return last_common


@overload
def flatten[B: Basis](basis: TupleBasis[tuple[B], Any, Any]) -> B: ...


@overload
def flatten[M: BasisMetadata, DT: ctype[Never]](
    basis: TupleBasisLike[tuple[TupleMetadata[tuple[M, ...]], ...], Never, DT],
) -> TupleBasis[tuple[Basis[M, DT], ...], None, DT]: ...


@overload
def flatten[DT: ctype[Never]](
    basis: Basis[TupleMetadata, DT],
) -> TupleBasis[tuple[Basis[BasisMetadata, DT], ...], None, DT]: ...


def flatten(
    basis: Basis[TupleMetadata, Any],
) -> Basis:
    as_tuple = as_tuple_basis(basis)
    if as_tuple is None:
        msg = "Cannot flatten a non-tuple basis."
        raise ValueError(msg)
    if len(as_tuple.shape) == 1:
        return as_tuple.children[0]

    children = tuple[Basis]()
    for b in as_tuple.children:
        if is_tuple_basis_like(b):
            b_as_tuple = as_tuple_basis(b)
            children = (*children, *b_as_tuple.children)
        else:
            children = (*children, b)
    b = as_tuple.metadata().extra
    return TupleBasis(children, as_tuple.metadata().extra)


def as_add_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports addition.

    If the basis is already an ADD basis, return it.
    If it wraps an ADD basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"ADD"})


def as_sub_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports subtraction.

    If the basis is already a SUB basis, return it.
    If it wraps a SUB basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"SUB"})


def as_mul_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports MUL.

    If the basis is already a MUL basis, return it.
    If it wraps a MUL basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"MUL"})


def as_linear_map_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports LINEAR_MAP.

    If the basis is already a LINEAR_MAP basis, return it.
    If it wraps a LINEAR_MAP basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"LINEAR_MAP"})


def as_is_dual_basis[M: BasisMetadata, DT: ctype[Never]](
    basis: Basis[M, DT], is_dual: NestedBool
) -> Basis[M, DT | ctype[np.generic]]:
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

    basis_as_tuple = as_tuple_basis(cast("Basis[TupleMetadata[Any,Any], Any]", basis))
    if basis_as_tuple is None:
        return from_metadata(basis.metadata(), is_dual=is_dual)

    return cast(
        "Basis[M, DT]",
        TupleBasis(
            tuple(
                starmap(
                    as_is_dual_basis,
                    zip(basis_as_tuple.children, is_dual, strict=False),
                )
            ),
            basis_as_tuple.metadata().extra,
        ),
    )
