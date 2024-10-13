"""A Stacked basis is a basis with a shape made from a tuple of individual basis."""

from __future__ import annotations

from ._diagonal_basis import DiagonalBasis
from ._tuple_basis import (
    StackedBasis,
    TupleBasis,
    TupleMetadata,
    VariadicTupleBasis,
    stacked_basis_as_fundamental,
)

__all__ = [
    "DiagonalBasis",
    "StackedBasis",
    "TupleBasis",
    "TupleMetadata",
    "VariadicTupleBasis",
    "stacked_basis_as_fundamental",
]
