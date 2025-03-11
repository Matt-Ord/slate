from __future__ import annotations

import uuid
from typing import Any, Never, cast, override

import numpy as np

from slate_core.array._array import Array, build
from slate_core.basis import (
    AsUpcast,
    Basis,
    BasisStateMetadata,
    Ctype,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
    TupleBasisLike,
)
from slate_core.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate_core.metadata import BasisMetadata, SimpleMetadata


class TrivialExplicitBasis[
    B: Basis,
    CT: Ctype[Never] = Ctype[Never],
](
    ExplicitUnitaryBasis[
        Array[
            TupleBasisLike[
                tuple[SimpleMetadata, BasisStateMetadata[B]],
                None,
                Ctype[np.generic],
            ],
            np.dtype[np.float64],
        ],
        CT,
    ]
):
    def __init__[B1_: Basis](
        self: TrivialExplicitBasis[B1_, Ctype[Never]],
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
    ) -> AsUpcast[TrivialExplicitBasis[B, CT], M, CT]:
        return cast("Any", AsUpcast(self, self.metadata()))
