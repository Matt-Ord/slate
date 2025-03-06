from __future__ import annotations

from abc import abstractmethod
from copy import copy
from typing import TYPE_CHECKING, Any, Never, Self, TypeGuard, cast, override

import numpy as np

from slate_core.basis._basis import Basis, BasisConversion, NestedBool, ctype
from slate_core.metadata import BasisMetadata

if TYPE_CHECKING:
    from collections.abc import Iterator


def _convert_vectors_unsafe[DT2: np.generic](
    initial: WrappedBasis[Basis[Any, ctype[np.generic]], ctype[np.generic]],
    vectors: np.ndarray[Any, np.dtype[DT2]],
    final: Basis[Any, ctype[np.generic]],
    axis: int = -1,
) -> np.ndarray[Any, np.dtype[DT2]]:
    assert initial.metadata() == final.metadata()

    if initial == final:
        return vectors

    as_inner = initial.__into_inner__(vectors, axis).ok()
    if is_wrapped_basis(final) and initial.inner == final.inner:
        return final.__from_inner__(as_inner, axis).ok()
    return initial.inner.__convert_vector_into__(as_inner, final, axis=axis).ok()


class WrappedBasis[
    B: Basis = Basis,
    # Invariant: DT must also be supported by all basis in B.
    # It is not possible to specify this constraint in the type system, so instead
    # we enforce it at __init__ time.
    DT: ctype[Never] = ctype[Never],
](Basis[BasisMetadata, DT]):
    """A wrapped basis, represents some transformation over an underlying 'inner' basis."""

    def __init__[B_: Basis](self: WrappedBasis[B_, ctype[Never]], inner: B_) -> None:
        self._inner = cast("B", inner)

    def upcast[DT_: ctype[Never]](
        self: WrappedBasis[Basis[Any, DT_], Any],
    ) -> WrappedBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("WrappedBasis[B, DT_]", self)

    def downcast_metadata[M: BasisMetadata](
        self: WrappedBasis[Basis[M, Any], Any],
    ) -> Basis[M, DT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", self)

    @override
    def metadata[M: BasisMetadata](self: WrappedBasis[Basis[M, Any], Any]) -> M:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return self._inner.metadata()

    @property
    @override
    def is_dual(self) -> NestedBool:
        return self.inner.is_dual

    @override
    def dual_basis(self) -> Self:
        copied = copy(self)
        copied._inner = self._inner.dual_basis()  # noqa: SLF001
        return copied

    @property
    def inner(self) -> B:
        """Inner basis."""
        return self._inner

    @abstractmethod
    def __into_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: WrappedBasis[Basis[Any, ctype[DT3]], ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]: ...

    @abstractmethod
    def __from_inner__[DT1: np.generic, DT2: np.generic, DT3: np.generic](
        self: WrappedBasis[Basis[Any, ctype[DT1]], ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]: ...

    @override
    def __into_fundamental__[DT1: np.generic, DT2: np.generic](
        self: WrappedBasis[Basis[Any, ctype[DT1]], ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, np.generic]:
        transformed = self.__into_inner__(vectors, axis)
        return BasisConversion[DT1, DT2, np.generic](
            lambda: self._inner.__into_fundamental__(transformed.ok(), axis=axis).ok()  # type: ignore BasisConversion makes this safe
        )

    @override
    def __from_fundamental__[DT2: np.generic, DT3: np.generic](
        self: WrappedBasis[Basis[Any, ctype[DT3]], ctype[DT3]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        axis: int = -1,
    ) -> BasisConversion[np.generic, DT2, DT3]:
        transformed = self._inner.__from_fundamental__(vectors, axis=axis)
        return BasisConversion[np.generic, DT2, DT3](
            lambda: self.__from_inner__(transformed.ok(), axis).ok()  # type: ignore BasisConversion makes this safe
        )

    @override
    def __convert_vector_into__[
        M_: BasisMetadata,
        DT1: np.generic,
        DT2: np.generic,
        DT3: np.generic,
    ](
        self: WrappedBasis[Basis[M_, ctype[DT1]], ctype[DT1]],
        vectors: np.ndarray[Any, np.dtype[DT2]],
        basis: Basis[M_, ctype[DT3]],
        axis: int = -1,
    ) -> BasisConversion[DT1, DT2, DT3]:
        return BasisConversion[DT1, DT2, DT3](
            lambda: _convert_vectors_unsafe(self, vectors, basis, axis)  # type: ignore BasisConversion makes this safe
        )


def is_wrapped_basis[
    M: BasisMetadata,
    DT: ctype[Never],
](basis: Basis[M, DT]) -> TypeGuard[WrappedBasis[Basis[M, DT], DT]]:
    """Check if a basis is a wrapped basis."""
    return isinstance(basis, WrappedBasis)


def wrapped_basis_iter_inner[
    M: BasisMetadata,
    DT: ctype[Never],
](basis: Basis[M, DT]) -> Iterator[Basis[M, DT]]:
    """Return successive calls to basis.inner until the basis is not a WrappedBasis."""
    yield basis
    if is_wrapped_basis(basis):
        yield from wrapped_basis_iter_inner(basis.inner)


def get_wrapped_basis_super_inner[
    M: BasisMetadata,
    DT: ctype[Never],
](basis: Basis[M, DT]) -> Basis[M, DT]:
    """Get the `super inner` of a wrapped basis.

    If the inner is itself a wrapped basis, return the super inner of that basis.
    """
    *_, last = wrapped_basis_iter_inner(basis)
    return last
