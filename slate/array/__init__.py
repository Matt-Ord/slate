"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import Array
from slate.array._conversion import (
    as_add_basis,
    as_diagonal_array,
    as_diagonal_basis,
    as_feature_basis,
    as_index_basis,
    as_mul_basis,
    as_outer_array,
    as_sub_basis,
    as_tuple_basis,
    cast_basis,
    flatten,
    nest,
    with_basis,
)
from slate.array._misc import abs, angle, imag, real  # noqa: A004
from slate.array._stats import average, standard_deviation
from slate.array._transpose import conjugate, dagger, transpose

__all__ = [
    "Array",
    "abs",
    "angle",
    "as_add_basis",
    "as_diagonal_array",
    "as_diagonal_basis",
    "as_feature_basis",
    "as_index_basis",
    "as_mul_basis",
    "as_outer_array",
    "as_sub_basis",
    "as_tuple_basis",
    "average",
    "cast_basis",
    "conjugate",
    "dagger",
    "flatten",
    "imag",
    "nest",
    "real",
    "standard_deviation",
    "transpose",
    "with_basis",
]
