"""The basis type that powers Slate."""

from __future__ import annotations

from slate.basis._basis import (
    Basis,
    BasisFeature,
)
from slate.basis._diagonal import DiagonalBasis, as_diagonal_basis, diagonal_basis
from slate.basis._isotropic import IsotropicBasis, isotropic_basis
from slate.basis._tuple import (
    StackedBasis,
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    TupleBasisND,
    as_tuple_basis,
    fundamental_tuple_basis_from_metadata,
    fundamental_tuple_basis_from_shape,
    tuple_basis,
    tuple_basis_is_variadic,
    tuple_basis_with_child,
    tuple_basis_with_modified_child,
    tuple_basis_with_modified_children,
)
from slate.basis.coordinate import CoordinateBasis
from slate.basis.cropped import CroppedBasis
from slate.basis.fundamental import FundamentalBasis
from slate.basis.recast import RecastBasis
from slate.basis.split import SplitBasis
from slate.basis.transformed import (
    TransformedBasis,
    fundamental_transformed_tuple_basis_from_metadata,
    fundamental_transformed_tuple_basis_from_shape,
)
from slate.basis.truncated import Padding, TruncatedBasis, Truncation
from slate.basis.wrapped import (
    WrappedBasis,
    as_add_basis,
    as_feature_basis,
    as_index_basis,
    as_mul_basis,
    as_sub_basis,
    get_common_basis,
    get_wrapped_basis_super_inner,
)

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
    "as_diagonal_basis",
    "as_feature_basis",
    "as_index_basis",
    "as_mul_basis",
    "as_sub_basis",
    "as_tuple_basis",
    "diagonal_basis",
    "fundamental_transformed_tuple_basis_from_metadata",
    "fundamental_transformed_tuple_basis_from_shape",
    "fundamental_tuple_basis_from_metadata",
    "fundamental_tuple_basis_from_shape",
    "get_common_basis",
    "get_wrapped_basis_super_inner",
    "isotropic_basis",
    "tuple_basis",
    "tuple_basis_is_variadic",
    "tuple_basis_with_child",
    "tuple_basis_with_modified_child",
    "tuple_basis_with_modified_children",
]
