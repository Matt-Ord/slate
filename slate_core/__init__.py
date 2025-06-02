"""Slate, a framework for automatic basis conversions."""

from __future__ import annotations

from slate_core import array, basis, explicit_basis, metadata, plot, util
from slate_core.array import Array
from slate_core.basis import (
    AsUpcast,
    Basis,
    BasisFeature,
    Ctype,
    FundamentalBasis,
    TransformedBasis,
    TupleBasis,
    TupleBasisLike,
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
    "AsUpcast",
    "Basis",
    "BasisFeature",
    "BasisMetadata",
    "Ctype",
    "FundamentalBasis",
    "SimpleMetadata",
    "TransformedBasis",
    "TupleBasis",
    "TupleBasisLike",
    "TupleMetadata",
    "array",
    "basis",
    "explicit_basis",
    "metadata",
    "plot",
    "util",
]
