"""The basis type that powers Slate."""

from __future__ import annotations

from ._basis import Basis, FundamentalBasis
from .evenly_spaced import EvenlySpacedBasis
from .stacked import (
    DiagonalBasis,
    StackedBasis,
    TupleBasis,
    VariadicTupleBasis,
    tuple_basis,
)
from .transformed import TransformedBasis
from .truncated import TruncatedBasis
from .wrapped import WrappedBasis

__all__ = [
    "Basis",
    "DiagonalBasis",
    "EvenlySpacedBasis",
    "FundamentalBasis",
    "StackedBasis",
    "TransformedBasis",
    "TruncatedBasis",
    "TupleBasis",
    "VariadicTupleBasis",
    "WrappedBasis",
    "tuple_basis",
]
