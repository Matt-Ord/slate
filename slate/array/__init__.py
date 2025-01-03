"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import Array
from slate.array._conversion import (
    as_add_basis,
    as_diagonal_array,
    as_diagonal_basis,
    as_feature_basis,
    as_fundamental_basis,
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
    log,
    log2,
    log10,
    mod,
    real,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
    unwrap,
)
from slate.array._stats import average, standard_deviation
from slate.array._transpose import conjugate, dagger, transpose

__all__ = [
    "Array",
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
    "imag",
    "log",
    "log2",
    "log10",
    "mod",
    "nest",
    "real",
    "sin",
    "sinh",
    "sqrt",
    "standard_deviation",
    "tan",
    "tanh",
    "transpose",
    "unwrap",
    "with_basis",
]
