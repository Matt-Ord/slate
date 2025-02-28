"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import Array, NestedIndex
from slate.array._conversion import (
    as_add_basis,
    as_diagonal_array,
    as_diagonal_basis,
    as_feature_basis,
    as_fundamental_basis,
    as_index_basis,
    as_mul_basis,
    as_outer_array,
    as_raw_array,
    as_sub_basis,
    as_tuple_basis,
    cast_basis,
    flatten,
    nest,
)
from slate.array._misc import (
    abs,  # noqa: A004
    angle,
    arccos,
    arccosh,
    arcsin,
    arcsinh,
    arctan,
    arctanh,
    cos,
    cosh,
    exp,
    imag,
    is_tuple_basis_like,
    log,
    log2,
    log10,
    mod,
    real,
    sin,
    sinh,
    sqrt,
    square,
    tan,
    tanh,
    unwrap,
)
from slate.array._stats import average, max, min, standard_deviation  # noqa: A004
from slate.array._transpose import conjugate, dagger, get_data_in_axes, transpose

__all__ = [
    "Array",
    "NestedIndex",
    "abs",
    "angle",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "as_add_basis",
    "as_diagonal_array",
    "as_diagonal_basis",
    "as_feature_basis",
    "as_fundamental_basis",
    "as_index_basis",
    "as_mul_basis",
    "as_outer_array",
    "as_raw_array",
    "as_sub_basis",
    "as_tuple_basis",
    "average",
    "cast_basis",
    "conjugate",
    "cos",
    "cosh",
    "dagger",
    "exp",
    "flatten",
    "get_data_in_axes",
    "imag",
    "is_tuple_basis_like",
    "log",
    "log2",
    "log10",
    "max",
    "min",
    "mod",
    "nest",
    "real",
    "sin",
    "sinh",
    "sqrt",
    "square",
    "standard_deviation",
    "tan",
    "tanh",
    "transpose",
    "unwrap",
]
