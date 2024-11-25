"""The array type for Slate."""

from __future__ import annotations

from slate.array._array import SlateArray
from slate.array._conversion import convert_array
from slate.array._util import conjugate

__all__ = ["SlateArray", "conjugate", "convert_array"]
