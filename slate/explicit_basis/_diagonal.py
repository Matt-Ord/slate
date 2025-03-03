from __future__ import annotations

import uuid
from typing import Any, Never, cast

import numpy as np

from slate.array._array import Array, build
from slate.basis import (
    Basis,
    BasisStateMetadata,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
)
from slate.basis._basis import ctype
from slate.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate.metadata._metadata import SimpleMetadata
from slate.metadata._stacked import TupleMetadata


class TrivialExplicitBasis[
    Transform: Array[
        Basis[TupleMetadata[tuple[SimpleMetadata, BasisStateMetadata[Basis]], None]],
        Any,
    ],
    DT: ctype[Never] = ctype[Never],
](ExplicitUnitaryBasis[Transform, DT]):
    def __init__[B1: Basis](
        self: TrivialExplicitBasis[
            Array[
                DiagonalBasis[
                    tuple[
                        FundamentalBasis[BasisStateMetadata[B1]],
                        FundamentalBasis[BasisStateMetadata[B1]],
                    ],
                    None,
                ],
                np.dtype[np.number],
            ],
            ctype[Never],
        ],
        inner: B1,
    ) -> None:
        matrix = build(
            DiagonalBasis(
                TupleBasis(
                    (
                        FundamentalBasis(BasisStateMetadata(inner)),
                        FundamentalBasis(BasisStateMetadata(inner)),
                    )
                ).upcast()
            ),
            np.ones(inner.size),
        ).ok()
        super().__init__(
            cast("Any", matrix),
            data_id=uuid.UUID(int=0),
        )
