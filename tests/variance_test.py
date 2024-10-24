from __future__ import annotations

from typing import Any, cast

import numpy as np

from slate.array.array import SlateArray
from slate.basis import Basis
from slate.basis._basis import FundamentalBasis
from slate.basis.stacked._tuple_basis import TupleBasis
from slate.basis.wrapped import WrappedBasis
from slate.metadata import BasisMetadata, SimpleMetadata


def basis_variance() -> None:
    a = cast(Basis[SimpleMetadata, np.floating[Any]], {})
    _b: Basis[BasisMetadata, np.generic] = a  # type: ignore should fail
    _c: Basis[BasisMetadata, np.float128] = a


def tuple_basis_variance() -> None:
    a = cast(TupleBasis[SimpleMetadata, np.floating[Any], np.floating[Any]], {})
    _b: Basis[BasisMetadata, np.generic] = a  # type: ignore should fail
    _c: Basis[BasisMetadata, np.float128] = a
    _d: TupleBasis[BasisMetadata, np.generic, np.float128] = a  # type: ignore should fail
    _d: TupleBasis[BasisMetadata, np.float128, np.float128] = a


def fundamental_basis_variance() -> None:
    a = cast(FundamentalBasis[SimpleMetadata], {})
    _b: Basis[BasisMetadata, np.generic] = a
    _c: Basis[BasisMetadata, np.float128] = a


def slate_array_basis_variance() -> None:
    a = cast(
        SlateArray[np.float64, Basis[SimpleMetadata, np.float64]],
        {},
    )
    _b: SlateArray[np.floating[Any], Basis[BasisMetadata, np.generic]] = a  # type: ignore should fail
    _c: SlateArray[np.floating[Any], Basis[BasisMetadata, np.float64]] = a
    _d: SlateArray[np.float128, Basis[BasisMetadata, np.float64]] = a  # type: ignore should fail
    _e: SlateArray[np.generic, Basis[BasisMetadata, np.float64]] = a


def wrappeed_basis_variance() -> None:
    a = cast(
        WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.float64]],
        {},
    )
    _b: WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.generic]] = a  # type: ignore should fail
    _c: WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.float64]] = a
    _d: WrappedBasis[SimpleMetadata, np.float64, Basis[BasisMetadata, np.generic]] = a  # type: ignore should fail
    _e: WrappedBasis[SimpleMetadata, np.float64, Basis[BasisMetadata, np.float64]] = a
