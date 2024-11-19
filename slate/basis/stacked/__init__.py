"""A Stacked basis is a basis with a shape made from a tuple of individual basis."""

from __future__ import annotations

from ._diagonal import DiagonalBasis, diagonal_basis
from ._isotropic import IsotropicBasis, isotropic_basis
from ._tuple import (
    StackedBasis,
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    TupleBasisND,
    as_tuple_basis,
    fundamental_tuple_basis_from_metadata,
    fundamental_tuple_basis_from_shape,
    stacked_basis_as_fundamental,
    tuple_basis,
    tuple_basis_with_child,
)

__all__ = [
    "DiagonalBasis",
    "IsotropicBasis",
    "StackedBasis",
    "TupleBasis",
    "TupleBasis1D",
    "TupleBasis2D",
    "TupleBasis3D",
    "TupleBasisND",
    "as_tuple_basis",
    "diagonal_basis",
    "fundamental_tuple_basis_from_metadata",
    "fundamental_tuple_basis_from_shape",
    "isotropic_basis",
    "stacked_basis_as_fundamental",
    "tuple_basis",
    "tuple_basis_with_child",
]
