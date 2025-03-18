from __future__ import annotations

from itertools import starmap
from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    cast,
    overload,
)

from slate_core.basis._basis import Basis, Ctype, NestedBool
from slate_core.basis._tuple import (
    TupleBasis,
    TupleBasisLike,
    as_feature,
    from_metadata,
    is_tuple,
    is_tuple_basis_like,
)
from slate_core.basis._tuple import (
    as_tuple as as_tuple_basis,
)
from slate_core.basis._wrapped import (
    wrapped_basis_iter_inner,
)
from slate_core.metadata import (
    AnyMetadata,
    BasisMetadata,
    NestedLength,
    TupleMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from slate_core.basis._fundamental import FundamentalBasis


def with_modified_children[
    M: BasisMetadata,
    E,
    DT0: Ctype[Never],
    DT1: Ctype[Never],
](
    basis: TupleBasis[tuple[Basis[M, DT0], ...], E, DT1],
    wrapper: Callable[[int, Basis[M, DT0]], Basis[M, DT0]],
) -> TupleBasis[tuple[Basis[M, DT0], ...], E, DT0]:
    """Get the basis with modified children."""
    return TupleBasis[tuple[Basis[M, DT0], ...], E, DT0](
        tuple(starmap(wrapper, enumerate(basis.children))), basis.metadata().extra
    ).resolve_ctype()


def with_modified_child[
    M: BasisMetadata,
    E,
    DT0: Ctype[Never],
    DT1: Ctype[Never],
](
    basis: TupleBasis[tuple[Basis[M, DT0], ...], E, DT1],
    wrapper: Callable[[Basis[M, DT0]], Basis[M, DT0]],
    idx: int,
) -> TupleBasis[tuple[Basis[M, DT0], ...], E, DT0]:
    """Get the basis with modified child."""
    return with_modified_children(basis, lambda i, b: b if i != idx else wrapper(b))


def with_child[M: BasisMetadata, E, CT: Ctype[Never]](
    basis: TupleBasis[tuple[Basis[M, CT], ...], E, CT], inner: Basis[M, CT], idx: int
) -> TupleBasis[tuple[Basis[M, CT], ...], E, CT]:
    """Get a basis with the basis at idx set to inner."""
    return with_modified_child(basis, lambda _: inner, idx)


def as_fundamental[M: AnyMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> Basis[M, Ctype[np.generic]]:
    return from_metadata(basis.metadata(), is_dual=basis.is_dual)


@overload
def from_shape[E](
    shape: tuple[int], *, extra: None = None, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[tuple[FundamentalBasis], None, Ctype[np.generic]]: ...
@overload
def from_shape[E](
    shape: tuple[int], *, extra: E, is_dual: tuple[bool, ...] | None = None
) -> TupleBasis[tuple[FundamentalBasis], E, Ctype[np.generic]]: ...


@overload
def from_shape[E](
    shape: tuple[int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, FundamentalBasis], None, Ctype[np.generic]]: ...
@overload
def from_shape[E](
    shape: tuple[int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, FundamentalBasis], E, Ctype[np.generic]]: ...


@overload
def from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis], None, Ctype[np.generic]
]: ...
@overload
def from_shape[E](
    shape: tuple[int, int, int],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[
    tuple[FundamentalBasis, FundamentalBasis, FundamentalBasis], E, Ctype[np.generic]
]: ...


@overload
def from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: None = None,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, ...], None, Ctype[np.generic]]: ...
@overload
def from_shape[E](
    shape: tuple[int, ...],
    *,
    extra: E,
    is_dual: tuple[bool, ...] | None = None,
) -> TupleBasis[tuple[FundamentalBasis, ...], E, Ctype[np.generic]]: ...


@overload
def from_shape[E](
    shape: tuple[NestedLength, ...],
    *,
    extra: None = None,
    is_dual: tuple[NestedBool, ...] | None = None,
) -> TupleBasis[
    tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], None, Ctype[np.generic]
]: ...
@overload
def from_shape[E](
    shape: tuple[NestedLength, ...],
    *,
    extra: E,
    is_dual: tuple[NestedBool, ...] | None = None,
) -> TupleBasis[
    tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], E, Ctype[np.generic]
]: ...


@overload
def from_shape[E](
    shape: int,
    *,
    extra: None = None,
    is_dual: bool | None = None,
) -> TupleBasis[
    tuple[Basis[BasisMetadata, Ctype[np.generic]], ...], None, Ctype[np.generic]
]: ...


def from_shape[E](
    shape: NestedLength,
    *,
    extra: Any | None = None,
    is_dual: NestedBool | None = None,
) -> Basis[Any, Ctype[np.generic]]:
    """Get a basis from the shape provided."""
    return from_metadata(TupleMetadata.from_shape(shape, extra=extra), is_dual=is_dual)


def get_common[M: BasisMetadata, CT: Ctype[Never]](
    lhs: Basis[M, CT],
    rhs: Basis[M, CT],
) -> Basis[M, CT | Ctype[np.generic]]:
    """Get the closest common basis of two bases."""
    assert rhs.metadata() == lhs.metadata()
    lhs_rev = list(wrapped_basis_iter_inner(lhs))
    rhs_rev = list(wrapped_basis_iter_inner(rhs))

    if is_tuple(lhs_rev[-1]) and is_tuple(rhs_rev[-1]) and lhs_rev != rhs_rev:
        # For a TupleBasis, we can do a bit better
        # By finding the common basis of the children
        lhs_children = lhs_rev[-1].children
        rhs_children = rhs_rev[-1].children
        children = zip(lhs_children, rhs_children, strict=True)

        basis = TupleBasis(
            tuple(starmap(get_common, children)),
            rhs_rev[-1].metadata().extra,
        )
        return cast("Basis[M, CT]", basis)

    last_common = from_metadata(rhs.metadata(), is_dual=rhs.is_dual)
    for a, b in zip(reversed(lhs_rev), reversed(rhs_rev), strict=False):
        if a != b:
            return last_common
        last_common = a
    return last_common


@overload
def flatten[B: Basis](basis: TupleBasis[tuple[B], Any, Any]) -> B: ...


@overload
def flatten[M: BasisMetadata, CT: Ctype[Never]](
    basis: TupleBasisLike[tuple[TupleMetadata[tuple[M, ...]], ...], Never, CT],
) -> TupleBasis[tuple[Basis[M, CT], ...], None, CT]: ...


@overload
def flatten[CT: Ctype[Never]](
    basis: Basis[TupleMetadata, CT],
) -> TupleBasis[tuple[Basis[BasisMetadata, CT], ...], None, CT]: ...


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


def as_add[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> Basis[M, CT]:
    """Get the closest basis that supports addition.

    If the basis is already an ADD basis, return it.
    If it wraps an ADD basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature(basis, {"ADD"})


def as_sub[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> Basis[M, CT]:
    """Get the closest basis that supports subtraction.

    If the basis is already a SUB basis, return it.
    If it wraps a SUB basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature(basis, {"SUB"})


def as_mul[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> Basis[M, CT]:
    """Get the closest basis that supports MUL.

    If the basis is already a MUL basis, return it.
    If it wraps a MUL basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature(basis, {"MUL"})


def as_linear_map[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT],
) -> Basis[M, CT]:
    """Get the closest basis that supports LINEAR_MAP.

    If the basis is already a LINEAR_MAP basis, return it.
    If it wraps a LINEAR_MAP basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature(basis, {"LINEAR_MAP"})


def as_is_dual[M: BasisMetadata, CT: Ctype[Never]](
    basis: Basis[M, CT], is_dual: NestedBool
) -> Basis[M, CT | Ctype[np.generic]]:
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
        "Basis[M, CT]",
        TupleBasis(
            tuple(
                starmap(
                    as_is_dual,
                    zip(basis_as_tuple.children, is_dual, strict=False),
                )
            ),
            basis_as_tuple.metadata().extra,
        ),
    )
