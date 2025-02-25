"""Slate, a framework for automatic basis conversions."""

from __future__ import annotations

from slate import array, basis, explicit_basis, metadata, plot, util
from slate.array import Array
from slate.basis import (
    Basis,
    BasisFeature,
    FundamentalBasis,
    TransformedBasis,
    TupleBasis,
    ctype,
)
from slate.metadata import AnyMetadata, BasisMetadata, SimpleMetadata, TupleMetadata

__all__ = [
    "AnyMetadata",
    "Array",
    "Basis",
    "BasisFeature",
    "BasisMetadata",
    "FundamentalBasis",
    "SimpleMetadata",
    "TransformedBasis",
    "TupleBasis",
    "TupleMetadata",
    "array",
    "basis",
    "ctype",
    "explicit_basis",
    "metadata",
    "plot",
    "util",
]
