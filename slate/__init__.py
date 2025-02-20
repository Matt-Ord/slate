"""Slate, a framework for automatic basis conversions."""

from __future__ import annotations

from slate import array, basis, explicit_basis, metadata, plot, util
from slate.array import Array
from slate.basis import Basis, FundamentalBasis, TupleBasis, tuple_basis
from slate.metadata import BasisMetadata, SimpleMetadata, TupleMetadata

__all__ = [
    "Array",
    "Basis",
    "BasisMetadata",
    "FundamentalBasis",
    "SimpleMetadata",
    "TupleBasis",
    "TupleMetadata",
    "array",
    "basis",
    "explicit_basis",
    "metadata",
    "plot",
    "tuple_basis",
    "util",
]
