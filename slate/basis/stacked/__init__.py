"""A Stacked basis is a basis with a shape made from a tuple of individual basis."""

from __future__ import annotations

from ._diagonal_basis import DiagonalBasis, diagonal_basis
from ._tuple_basis import (
    StackedBasis,
    TupleBasis,
    VariadicTupleBasis,
    as_tuple_basis,
    fundamental_tuple_basis_from_metadata,
    fundamental_tuple_basis_from_shape,
    stacked_basis_as_fundamental,
    tuple_basis,
)

__all__ = [
    "DiagonalBasis",
    "StackedBasis",
    "TupleBasis",
    "VariadicTupleBasis",
    "as_tuple_basis",
    "diagonal_basis",
    "fundamental_tuple_basis_from_metadata",
    "fundamental_tuple_basis_from_shape",
    "stacked_basis_as_fundamental",
    "tuple_basis",
]
