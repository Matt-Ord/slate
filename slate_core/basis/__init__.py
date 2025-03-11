"""The basis type that powers Slate."""

from __future__ import annotations

from slate_core.basis._basis import (
    Basis,
    BasisConversion,
    BasisFeature,
    Ctype,
    NestedBool,
    NestedBoolOrNone,
    UnionCtype,
    are_dual,
    are_dual_shapes,
    supports_dtype,
)
from slate_core.basis._basis_state_metadata import (
    BasisStateMetadata,
    is_basis_state_metadata,
)
from slate_core.basis._block_diagonal import BlockDiagonalBasis, as_block_diagonal_basis
from slate_core.basis._coordinate import CoordinateBasis
from slate_core.basis._cropped import CroppedBasis, is_cropped_basis
from slate_core.basis._diagonal import (
    DiagonalBasis,
    as_diagonal_basis,
    is_diagonal_basis,
)
from slate_core.basis._fundamental import FundamentalBasis, as_state_list
from slate_core.basis._isotropic import IsotropicBasis
from slate_core.basis._recast import RecastBasis, is_recast_basis
from slate_core.basis._split import SplitBasis
from slate_core.basis._transformed import (
    TransformedBasis,
    as_transformed,
    transformed_from_metadata,
    transformed_from_shape,
)
from slate_core.basis._truncated import Padding, TruncatedBasis, Truncation
from slate_core.basis._tuple import (
    TupleBasis,
    TupleBasis1D,
    TupleBasis2D,
    TupleBasis3D,
    TupleBasisLike,
    TupleBasisLike1D,
    TupleBasisLike2D,
    TupleBasisLike3D,
    as_feature_basis,
    as_index_basis,
    as_tuple_basis,
    from_metadata,
    is_tuple_basis,
    is_tuple_basis_like,
)
from slate_core.basis._util import (
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
from slate_core.basis._wrapped import (
    AsUpcast,
    WrappedBasis,
    WrappedBasisWithMetadata,
    get_wrapped_basis_super_inner,
    is_wrapped_basis,
    wrapped_basis_iter_inner,
)

__all__ = [
    "AsUpcast",
    "Basis",
    "BasisConversion",
    "BasisFeature",
    "BasisStateMetadata",
    "BlockDiagonalBasis",
    "CoordinateBasis",
    "CroppedBasis",
    "Ctype",
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
    "TupleBasis1D",
    "TupleBasis2D",
    "TupleBasis3D",
    "TupleBasisLike",
    "TupleBasisLike1D",
    "TupleBasisLike2D",
    "TupleBasisLike3D",
    "UnionCtype",
    "WrappedBasis",
    "WrappedBasisWithMetadata",
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
    "is_wrapped_basis",
    "supports_dtype",
    "transformed_from_metadata",
    "transformed_from_shape",
    "with_child",
    "with_modified_child",
    "with_modified_children",
    "wrapped_basis_iter_inner",
]
