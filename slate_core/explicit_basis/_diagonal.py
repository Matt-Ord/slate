from __future__ import annotations

import uuid
from typing import Any, Never, cast, override

import numpy as np

from slate_core.array._array import Array, build
from slate_core.basis import (
    Basis,
    BasisStateMetadata,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
    TupleBasisLike,
    ctype,
)
from slate_core.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate_core.metadata import BasisMetadata, SimpleMetadata


class TrivialExplicitBasis[
    B1: Basis,
    DT: ctype[Never] = ctype[Never],
](
    ExplicitUnitaryBasis[
        Array[
            TupleBasisLike[
                tuple[SimpleMetadata, BasisStateMetadata[B1]],
                None,
                ctype[np.generic],
            ],
            np.dtype[np.float64],
        ],
        DT,
    ]
):
    def __init__[B1_: Basis](
        self: TrivialExplicitBasis[B1_, ctype[Never]],
        inner: B1_,
    ) -> None:
        matrix = build(
            DiagonalBasis(
                TupleBasis(
                    (
                        FundamentalBasis(BasisStateMetadata(inner)),
                        FundamentalBasis(BasisStateMetadata(inner)),
                    )
                ).upcast()
            ).upcast(),
            np.ones(inner.size),
        ).ok()
        super().__init__(cast("Any", matrix), data_id=uuid.UUID(int=0))

    @override
    def downcast_metadata[M: BasisMetadata](
        self: TrivialExplicitBasis[Basis[M]],
    ) -> Basis[M, DT]:
        return cast("Basis[M, DT]", self)
