from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate.array import Array
from slate.basis import (
    Basis,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
    TupleBasis2D,
    WrappedBasis,
)
from slate.metadata import (
    BasisMetadata,
    LengthMetadata,
    Metadata2D,
    MetadataND,
    SimpleMetadata,
    VolumeMetadata,
)

if TYPE_CHECKING:
    from slate.basis import TransformedBasis
    from slate.metadata import StackedMetadata


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
    a1 = cast(
        TupleBasis[VolumeMetadata, None, np.generic],
        {},
    )
    _b1: TupleBasis[VolumeMetadata, None, np.complex128] = a1


def fundamental_basis_variance() -> None:
    a = cast(FundamentalBasis[SimpleMetadata], {})
    _b: Basis[BasisMetadata, np.generic] = a
    _c: Basis[BasisMetadata, np.float128] = a


def slate_array_basis_variance() -> None:
    a = cast(
        Array[SimpleMetadata, np.float64, FundamentalBasis[SimpleMetadata]],
        {},
    )
    _b: Array[BasisMetadata, np.generic] = a
    _c: Array[BasisMetadata, np.floating[Any]] = a
    _d: Array[BasisMetadata, np.float128] = a  # type: ignore should fail
    _d: Array[BasisMetadata, np.float64, TransformedBasis[SimpleMetadata]] = a  # type: ignore should fail
    _e: Array[BasisMetadata, np.float64] = a
    _f: Array[LengthMetadata, np.float64] = a  # type: ignore should fail


def wrappeed_basis_variance() -> None:
    a = cast(
        WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.float64]],
        {},
    )
    _b: WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.generic]] = a  # type: ignore should fail
    _c: WrappedBasis[SimpleMetadata, np.float64, Basis[SimpleMetadata, np.float64]] = a
    _d: WrappedBasis[SimpleMetadata, np.float64, Basis[BasisMetadata, np.generic]] = a  # type: ignore should fail
    _e: WrappedBasis[SimpleMetadata, np.float64, Basis[BasisMetadata, np.float64]] = a


def stacked_metadata_variance() -> None:
    a = cast(Metadata2D[SimpleMetadata, SimpleMetadata, SimpleMetadata], {})
    _b: Metadata2D[BasisMetadata, BasisMetadata, BasisMetadata] = a
    _c: Metadata2D[BasisMetadata, BasisMetadata, LengthMetadata] = a  # type: ignore should fail
    _d: Metadata2D[BasisMetadata, LengthMetadata, BasisMetadata] = a  # type: ignore should fail
    _e: Metadata2D[LengthMetadata, BasisMetadata, BasisMetadata] = a  # type: ignore should fail

    a1 = cast(MetadataND[SimpleMetadata, SimpleMetadata, SimpleMetadata], {})
    _b1: MetadataND[BasisMetadata, BasisMetadata, BasisMetadata] = a1  # type: ignore i disagree with type checker


def tuple_basis_2d_variance() -> None:
    a1 = cast(
        TupleBasis2D[
            np.floating[Any],
            Basis[LengthMetadata, np.float64],
            Basis[SimpleMetadata, np.float64],
            None,
        ],
        {},
    )
    _c1: TupleBasis2D[
        np.float64,
        Basis[SimpleMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a1
    _e1: TupleBasis2D[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a1
    _e2: TupleBasis2D[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a1
    _e5: TupleBasis[
        BasisMetadata,
        None,
        np.float64,
    ] = a1
    _e4: Metadata2D[LengthMetadata, SimpleMetadata, None] = a1.metadata()
    _e3: TupleBasis2D[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[LengthMetadata, np.float64],
        None,
    ] = a1  # type: ignore should fail

    a2 = cast(
        TupleBasis2D[
            np.generic, Basis[VolumeMetadata, Any], Basis[VolumeMetadata, Any], None
        ],
        {},
    )
    _b2: TupleBasis[Any, None, np.complex128, StackedMetadata[Any, None]] = a2


def diagonal_basis_variance() -> None:
    a = cast(
        DiagonalBasis[
            np.floating[Any],
            Basis[SimpleMetadata, np.float64],
            Basis[SimpleMetadata, np.float64],
            None,
        ],
        {},
    )
    _b: DiagonalBasis[
        np.generic,
        Basis[SimpleMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a  # type: ignore should fail
    _c: DiagonalBasis[
        np.float64,
        Basis[SimpleMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a
    _d: DiagonalBasis[
        np.float64,
        Basis[StackedMetadata[Any, Any], np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a  # type: ignore should fail
    _e: DiagonalBasis[
        np.float64,
        Basis[BasisMetadata, np.float64],
        Basis[SimpleMetadata, np.float64],
        None,
    ] = a
