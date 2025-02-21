from __future__ import annotations

import uuid
from typing import Any, cast

import numpy as np

from slate.array import Array
from slate.basis import (
    Basis,
    BasisStateMetadata,
    DiagonalBasis,
    FundamentalBasis,
    TupleBasis,
)
from slate.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata


class TrivialExplicitBasis[
    M: BasisMetadata,
    DT: np.dtype[np.generic],
    B: Basis[Any, Any],
](ExplicitUnitaryBasis[M, DT, B, Basis[Any, Any]]):
    def __init__[B1: Basis[Any, Any]](
        self: TrivialExplicitBasis[Any, Any, B1], inner: B1
    ) -> None:
        super().__init__(
            Array(
                DiagonalBasis(
                    TupleBasis(
                        (
                            FundamentalBasis(BasisStateMetadata(inner)),
                            FundamentalBasis(BasisStateMetadata(inner)),
                        )
                    )
                ),
                cast("np.ndarray[Any, DT]", np.ones(inner.size)),
            ),
            data_id=uuid.UUID(int=0),
        )
