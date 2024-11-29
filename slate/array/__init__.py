"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import SlateArray
from slate.array._conversion import with_basis
from slate.array._transpose import transpose
from slate.array._util import (
    as_add_basis,
    as_diagonal_basis,
    as_feature_basis,
    as_flatten_basis,
    as_index_basis,
    as_mul_basis,
    as_outer_basis,
    as_sub_basis,
    as_tuple_basis,
    conjugate,
)

__all__ = [
    "SlateArray",
    "as_add_basis",
    "as_diagonal_basis",
    "as_feature_basis",
    "as_flatten_basis",
    "as_index_basis",
    "as_mul_basis",
    "as_outer_basis",
    "as_sub_basis",
    "as_tuple_basis",
    "conjugate",
    "transpose",
    "with_basis",
]
