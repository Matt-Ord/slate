"""Slate, a framework for automatic basis conversions."""

from __future__ import annotations

from slate import array, basis, explicit_basis, metadata, plot, util
from slate.array import SlateArray
from slate.basis import Basis, FundamentalBasis, TupleBasis, tuple_basis
from slate.metadata import BasisMetadata, SimpleMetadata, StackedMetadata

__all__ = [
    "Basis",
    "BasisMetadata",
    "FundamentalBasis",
    "SimpleMetadata",
    "SlateArray",
    "StackedMetadata",
    "TupleBasis",
    "array",
    "basis",
    "explicit_basis",
    "metadata",
    "plot",
    "tuple_basis",
    "util",
]
