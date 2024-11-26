from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from slate.basis._tuple import fundamental_basis_from_metadata
from slate.basis.wrapped import wrapped_basis_iter_inner
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis._basis import Basis, BasisFeature


def as_feature_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT], features: set[BasisFeature]
) -> Basis[M, DT]:
    """Get the closest basis that supports the feature set."""
    return next(
        (b for b in wrapped_basis_iter_inner(basis) if features <= b.features),
        fundamental_basis_from_metadata(basis.metadata()),
    )


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


def as_index_basis[M: BasisMetadata, DT: np.generic](
    basis: Basis[M, DT],
) -> Basis[M, DT]:
    """Get the closest basis that supports INDEX.

    If the basis is already an INDEX basis, return it.
    If it wraps a INDEX basis, return the inner basis.
    Otherwise, return the fundamental.
    """
    return as_feature_basis(basis, {"INDEX"})
