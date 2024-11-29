"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import SlateArray
from slate.array._conversion import with_basis
from slate.array._transpose import transpose
from slate.array._util import (
    as_flatten_basis,
    as_outer_basis,
    as_tuple_basis,
    conjugate,
)

__all__ = [
    "SlateArray",
    "as_flatten_basis",
    "as_outer_basis",
    "as_tuple_basis",
    "conjugate",
    "transpose",
    "with_basis",
]
