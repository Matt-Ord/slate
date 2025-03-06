"""Slate, a framework for automatic basis conversions."""

from __future__ import annotations

from slate_core import array, basis, explicit_basis, metadata, plot, util
from slate_core.array import Array
from slate_core.basis import (
    Basis,
    BasisFeature,
    FundamentalBasis,
    TransformedBasis,
    TupleBasis,
    ctype,
)
from slate_core.metadata import (
    AnyMetadata,
    BasisMetadata,
    SimpleMetadata,
    TupleMetadata,
)

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
