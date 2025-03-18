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
    supports_type,
)
from slate_core.basis._basis_state_metadata import (
    BasisStateMetadata,
    is_basis_state_metadata,
)
from slate_core.basis._block_diagonal import BlockDiagonalBasis, as_block_diagonal
from slate_core.basis._coordinate import CoordinateBasis
from slate_core.basis._cropped import CroppedBasis, is_cropped
from slate_core.basis._diagonal import (
    DiagonalBasis,
    as_diagonal,
    is_diagonal,
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
    as_feature,
    as_index,
    as_supports_type,
    as_tuple,
    from_metadata,
    is_tuple,
    is_tuple_basis_like,
)
from slate_core.basis._util import (
    as_add,
    as_fundamental,
    as_is_dual,
    as_linear_map,
    as_mul,
    as_sub,
    flatten,
    from_shape,
    get_common,
    with_child,
    with_modified_child,
    with_modified_children,
)
from slate_core.basis._wrapped import (
    AsUpcast,
    WrappedBasis,
    WrappedBasisWithMetadata,
    get_wrapped_basis_super_inner,
    is_wrapped,
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
    "as_add",
    "as_block_diagonal",
    "as_diagonal",
    "as_feature",
    "as_fundamental",
    "as_index",
    "as_is_dual",
    "as_linear_map",
    "as_mul",
    "as_state_list",
    "as_sub",
    "as_supports_type",
    "as_transformed",
    "as_tuple",
    "flatten",
    "from_metadata",
    "from_shape",
    "get_common",
    "get_wrapped_basis_super_inner",
    "is_basis_state_metadata",
    "is_cropped",
    "is_diagonal",
    "is_recast_basis",
    "is_tuple",
    "is_tuple_basis_like",
    "is_wrapped",
    "supports_dtype",
    "supports_type",
    "transformed_from_metadata",
    "transformed_from_shape",
    "with_child",
    "with_modified_child",
    "with_modified_children",
    "wrapped_basis_iter_inner",
]
