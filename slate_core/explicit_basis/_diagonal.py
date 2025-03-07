from __future__ import annotations

import uuid
from typing import Any, Never, cast, override

import numpy as np

from slate_core.array._array import Array, build
from slate_core.basis import (
    AsUpcast,
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
    B: Basis,
    DT: ctype[Never] = ctype[Never],
](
    ExplicitUnitaryBasis[
        Array[
            TupleBasisLike[
                tuple[SimpleMetadata, BasisStateMetadata[B]],
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
                ).resolve_ctype()
            ).resolve_ctype(),
            np.ones(inner.size),
        ).ok()
        super().__init__(cast("Any", matrix), data_id=uuid.UUID(int=0))

    @override
    def upcast[M: BasisMetadata](
        self: TrivialExplicitBasis[Basis[M]],
    ) -> AsUpcast[TrivialExplicitBasis[B, DT], M, DT]:
        return cast("Any", AsUpcast(self, self.metadata()))
