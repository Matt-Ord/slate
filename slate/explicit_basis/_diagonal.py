from __future__ import annotations

import uuid
from typing import Any, cast

import numpy as np

from slate.array import Array
from slate.basis import Basis, BasisStateMetadata, FundamentalBasis, diagonal_basis
from slate.explicit_basis._explicit_basis import ExplicitUnitaryBasis
from slate.metadata import BasisMetadata


class TrivialExplicitBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any],
](ExplicitUnitaryBasis[M, DT, B, Basis[Any, Any]]):
    def __init__[B1: Basis[Any, Any]](
        self: TrivialExplicitBasis[Any, Any, B1], inner: B1
    ) -> None:
        super().__init__(
            Array(
                diagonal_basis(
                    (
                        FundamentalBasis(BasisStateMetadata(inner)),
                        FundamentalBasis(BasisStateMetadata(inner)),
                    )
                ),
                cast("np.ndarray[Any, np.dtype[DT]]", np.ones(inner.size)),
            ),
            data_id=uuid.UUID(int=0),
        )
