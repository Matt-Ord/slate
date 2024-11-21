"""The array type for Slate."""

from __future__ import annotations

from ._array import SlateArray
from ._conversion import convert_array
from ._util import transpose

__all__ = ["SlateArray", "convert_array", "transpose"]
