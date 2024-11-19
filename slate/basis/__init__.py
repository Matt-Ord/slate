"""The basis type that powers Slate."""

from __future__ import annotations

from ._basis import (
    Basis,
    BasisFeatures,
)
from .coordinate import CoordinateBasis
from .cropped import CroppedBasis
from .fundamental import FundamentalBasis
from .recast import RecastBasis
from .split import SplitBasis, split_basis
from .stacked import (
    DiagonalBasis,
    StackedBasis,
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    TupleBasisND,
    as_tuple_basis,
    tuple_basis,
)
from .transformed import TransformedBasis
from .truncated import TruncatedBasis, Truncation
from .wrapped import WrappedBasis, as_add_basis, as_mul_basis, as_sub_basis

__all__ = [
    "Basis",
    "BasisFeatures",
    "CoordinateBasis",
    "CroppedBasis",
    "DiagonalBasis",
    "FundamentalBasis",
    "RecastBasis",
    "SplitBasis",
    "StackedBasis",
    "TransformedBasis",
    "TruncatedBasis",
    "Truncation",
    "TupleBasis",
    "TupleBasis1D",
    "TupleBasis2D",
    "TupleBasis3D",
    "TupleBasisND",
    "WrappedBasis",
    "as_add_basis",
    "as_mul_basis",
    "as_sub_basis",
    "as_tuple_basis",
    "split_basis",
    "tuple_basis",
]
