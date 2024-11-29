"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import SlateArray
from slate.array._conversion import convert_array
from slate.array._util import (
    array_as_flatten_basis,
    array_as_outer_basis,
    array_as_tuple_basis,
    conjugate,
)

__all__ = [
    "SlateArray",
    "array_as_flatten_basis",
    "array_as_outer_basis",
    "array_as_tuple_basis",
    "conjugate",
    "convert_array",
]
