from __future__ import annotations

from typing import Any, cast

import numpy as np

from slate.array.array import SlateArray
from slate.basis import Basis
from slate.basis._basis import FundamentalBasis
from slate.basis.metadata import BasisMetadata, FundamentalBasisMetadata
from slate.basis.stacked._tuple_basis import TupleBasis


def basis_variance() -> None:
    a = cast(Basis[FundamentalBasisMetadata, np.floating[Any]], {})
    _b: Basis[BasisMetadata, np.generic] = a  # type: ignore should fail
    _c: Basis[BasisMetadata, np.float128] = a


def tuple_basis_variance() -> None:
    a = cast(
        TupleBasis[FundamentalBasisMetadata, np.floating[Any], np.floating[Any]], {}
    )
    _b: Basis[BasisMetadata, np.generic] = a  # type: ignore should fail
    _c: Basis[BasisMetadata, np.float128] = a
    _d: TupleBasis[BasisMetadata, np.generic, np.float128] = a  # type: ignore should fail
    _d: TupleBasis[BasisMetadata, np.float128, np.float128] = a


def fundamental_basis_variance() -> None:
    a = cast(FundamentalBasis[FundamentalBasisMetadata], {})
    _b: Basis[BasisMetadata, np.generic] = a
    _c: Basis[BasisMetadata, np.float128] = a


def slate_array_basis_variance() -> None:
    a = cast(
        SlateArray[np.float64, Basis[FundamentalBasisMetadata, np.float64]],
        {},
    )
    _b: SlateArray[np.floating[Any], Basis[BasisMetadata, np.generic]] = a  # type: ignore should fail
    _c: SlateArray[np.floating[Any], Basis[BasisMetadata, np.float64]] = a
    _d: SlateArray[np.float128, Basis[BasisMetadata, np.float64]] = a  # type: ignore should fail
    _e: SlateArray[np.generic, Basis[BasisMetadata, np.float64]] = a
