"""The basis type that powers Slate."""

from __future__ import annotations

from ._basis import (
    Basis,
    BasisFeature,
)
from .coordinate import CoordinateBasis
from .cropped import CroppedBasis
from .fundamental import FundamentalBasis
from .recast import RecastBasis
from .split import SplitBasis, split_basis
from .stacked import (
    DiagonalBasis,
    IsotropicBasis,
    StackedBasis,
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    TupleBasisND,
    as_tuple_basis,
    diagonal_basis,
    fundamental_tuple_basis_from_metadata,
    fundamental_tuple_basis_from_shape,
    isotropic_basis,
    stacked_basis_as_fundamental,
    tuple_basis,
    tuple_basis_is_variadic,
    tuple_basis_with_child,
    tuple_basis_with_modified_child,
    tuple_basis_with_modified_children,
)
from .transformed import (
    TransformedBasis,
    fundamental_transformed_tuple_basis_from_metadata,
    fundamental_transformed_tuple_basis_from_shape,
)
from .truncated import Padding, TruncatedBasis, Truncation
from .wrapped import WrappedBasis, as_add_basis, as_mul_basis, as_sub_basis

__all__ = [
    "Basis",
    "BasisFeature",
    "CoordinateBasis",
    "CroppedBasis",
    "DiagonalBasis",
    "FundamentalBasis",
    "IsotropicBasis",
    "Padding",
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
    "diagonal_basis",
    "fundamental_transformed_tuple_basis_from_metadata",
    "fundamental_transformed_tuple_basis_from_shape",
    "fundamental_tuple_basis_from_metadata",
    "fundamental_tuple_basis_from_shape",
    "isotropic_basis",
    "split_basis",
    "stacked_basis_as_fundamental",
    "tuple_basis",
    "tuple_basis_is_variadic",
    "tuple_basis_with_child",
    "tuple_basis_with_modified_child",
    "tuple_basis_with_modified_children",
]
