from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Never,
    TypeGuard,
    cast,
    overload,
    override,
)

import numpy as np

from slate_core.basis._basis import Basis, BasisFeature, Ctype
from slate_core.basis._contracted import ContractedBasis
from slate_core.basis._tuple import TupleBasis
from slate_core.basis._wrapped import AsUpcast, WrappedBasis, wrapped_basis_iter_inner
from slate_core.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate_core.metadata import TupleMetadata


class DiagonalBasis[
    B: TupleBasis[
        tuple[Basis[BasisMetadata, Any], Basis[BasisMetadata, Any]], Any
    ] = TupleBasis[tuple[Basis, Basis], Any],
    CT: Ctype[Never] = Ctype[Never],
](
    ContractedBasis[B, CT],
):
    """Represents a diagonal basis."""

    def __init__[
        B_: TupleBasis[tuple[Basis[BasisMetadata, Any], Basis[BasisMetadata, Any]], Any]
    ](self: DiagonalBasis[B_, Ctype[Never]], inner: B_) -> None:
        super().__init__(cast("B", inner), (0, 0))

    @property
    @override
    def ctype(self) -> CT:
        return cast("CT", self.inner.ctype)

    @override
    def resolve_ctype[DT_: Ctype[Never]](
        self: DiagonalBasis[TupleBasis[tuple[Basis, Basis], Any, DT_], Any],
    ) -> DiagonalBasis[B, DT_]:
        """Upcast the wrapped basis to a more specific type."""
        return cast("DiagonalBasis[B, DT_]", self)

    @override
    def metadata[M0: BasisMetadata, M1: BasisMetadata, E](
        self: DiagonalBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> TupleMetadata[tuple[M0, M1], E]:
        return self.inner.metadata()

    @overload
    def upcast[M0: BasisMetadata, M1: BasisMetadata, E](
        self: DiagonalBasis[TupleBasis[tuple[Basis[M0], Basis[M1]], E], Any],
    ) -> AsUpcast[DiagonalBasis[B, CT], TupleMetadata[tuple[M0, M1], E], CT]: ...
    @overload
    def upcast(self) -> AsUpcast[DiagonalBasis[B, CT], BasisMetadata, CT]: ...

    @override
    def upcast(self) -> AsUpcast[DiagonalBasis[B, CT], BasisMetadata, CT]:
        """Metadata associated with the basis.

        Note: this should be a property, but this would ruin variance.
        """
        return cast("Any", AsUpcast(self, self.metadata()))

    @property
    @override
    def size(self) -> int:
        return self.inner.children[0].size

    @property
    @override
    def features(self) -> set[BasisFeature]:
        out = set[BasisFeature]()
        if "LINEAR_MAP" in self.inner.features:
            out.add("ADD")
            out.add("MUL")
            out.add("SUB")
            out.add("LINEAR_MAP")
        if "INDEX" in self.inner.features:
            out.add("INDEX")
        return out

    @override
    def add_data[T: np.number](
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
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
        self,
        lhs: np.ndarray[Any, np.dtype[T]],
        rhs: np.ndarray[Any, np.dtype[T]],
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
            cast("WrappedBasis[Any, Ctype[np.int_]]", self)
            .__from_inner__(self.inner.points)
            .ok()
        )


@overload
def is_diagonal[M1: BasisMetadata, M2: BasisMetadata, E, CT: Ctype[Never]](  # type: ignore not overlapping
    basis: Basis[TupleMetadata[tuple[M1, M2], E], CT],
) -> TypeGuard[
    DiagonalBasis[TupleBasis[tuple[Basis[M1, CT], Basis[M2, CT]], E], CT]
]: ...
@overload
def is_diagonal[M1: BasisMetadata, M2: BasisMetadata, E, CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT],
) -> TypeGuard[
    DiagonalBasis[
        TupleBasis[tuple[Basis[BasisMetadata, CT], Basis[BasisMetadata, CT]], Never], CT
    ]
]: ...
@overload
def is_diagonal(basis: object) -> TypeGuard[DiagonalBasis]: ...


def is_diagonal(basis: object) -> TypeGuard[DiagonalBasis]:
    return isinstance(basis, DiagonalBasis)


@overload
def as_diagonal[M1: BasisMetadata, M2: BasisMetadata, E, CT: Ctype[Never]](  # type: ignore not overlapping
    basis: Basis[TupleMetadata[tuple[M1, M2], E], CT],
) -> DiagonalBasis[TupleBasis[tuple[Basis[M1, CT], Basis[M2, CT]], E], CT] | None: ...
@overload
def as_diagonal[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT],
) -> (
    DiagonalBasis[
        TupleBasis[tuple[Basis[BasisMetadata, CT], Basis[BasisMetadata, CT]], Never], CT
    ]
    | None
): ...


def as_diagonal[CT: Ctype[Never]](
    basis: Basis[BasisMetadata, CT],
) -> DiagonalBasis[Any, CT] | None:
    """Get the closest basis that supports the feature set."""
    shape = basis.metadata().fundamental_shape
    if isinstance(shape, int) or len(shape) != 2:  # noqa: PLR2004
        return None
    return next((b for b in wrapped_basis_iter_inner(basis) if is_diagonal(b)), None)
