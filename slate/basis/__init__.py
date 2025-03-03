"""The basis type that powers Slate."""

from __future__ import annotations

from slate.basis._basis import (
    Basis,
    BasisFeature,
    NestedBool,
    NestedBoolOrNone,
    are_dual,
    are_dual_shapes,
    ctype,
)
from slate.basis._basis_state_metadata import (
    BasisStateMetadata,
    is_basis_state_metadata,
)
from slate.basis._block_diagonal import BlockDiagonalBasis, as_block_diagonal_basis
from slate.basis._coordinate import CoordinateBasis
from slate.basis._cropped import CroppedBasis, is_cropped_basis
from slate.basis._diagonal import DiagonalBasis, as_diagonal_basis, is_diagonal_basis
from slate.basis._fundamental import FundamentalBasis, as_state_list
from slate.basis._isotropic import IsotropicBasis
from slate.basis._recast import RecastBasis, is_recast_basis
from slate.basis._split import SplitBasis
from slate.basis._transformed import (
    TransformedBasis,
    as_transformed,
    transformed_from_metadata,
    transformed_from_shape,
)
from slate.basis._truncated import Padding, TruncatedBasis, Truncation
from slate.basis._tuple import (
    TupleBasis,
    TupleBasisLike,
    as_feature_basis,
    as_index_basis,
    as_tuple_basis,
    from_metadata,
    is_tuple_basis,
    is_tuple_basis_like,
)
from slate.basis._util import (
    as_add_basis,
    as_fundamental,
    as_is_dual_basis,
    as_linear_map_basis,
    as_mul_basis,
    as_sub_basis,
    flatten,
    from_shape,
    get_common_basis,
    with_child,
    with_modified_child,
    with_modified_children,
)
from slate.basis._wrapped import (
    WrappedBasis,
    get_wrapped_basis_super_inner,
)

__all__ = [
    "Basis",
    "BasisFeature",
    "BasisStateMetadata",
    "BlockDiagonalBasis",
    "CoordinateBasis",
    "CroppedBasis",
    "DiagonalBasis",
    "FundamentalBasis",
    "IsotropicBasis",
    "NestedBool",
    "NestedBoolOrNone",
    "Padding",
    "RecastBasis",
    "SplitBasis",
    "TransformedBasis",
    "TruncatedBasis",
    "Truncation",
    "TupleBasis",
    "TupleBasisLike",
    "WrappedBasis",
    "are_dual",
    "are_dual_shapes",
    "as_add_basis",
    "as_block_diagonal_basis",
    "as_diagonal_basis",
    "as_feature_basis",
    "as_fundamental",
    "as_index_basis",
    "as_is_dual_basis",
    "as_linear_map_basis",
    "as_mul_basis",
    "as_state_list",
    "as_sub_basis",
    "as_transformed",
    "as_tuple_basis",
    "ctype",
    "flatten",
    "from_metadata",
    "from_shape",
    "get_common_basis",
    "get_wrapped_basis_super_inner",
    "is_basis_state_metadata",
    "is_cropped_basis",
    "is_diagonal_basis",
    "is_recast_basis",
    "is_tuple_basis",
    "is_tuple_basis_like",
    "transformed_from_metadata",
    "transformed_from_shape",
    "with_child",
    "with_modified_child",
    "with_modified_children",
]
