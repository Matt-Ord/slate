from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate.explicit_basis._explicit_basis import (
    Direction,
    ExplicitBasis,
    ExplicitUnitaryBasis,
)
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    import uuid

    from slate.array import Array
    from slate.basis import Basis, BasisStateMetadata, BlockDiagonalBasis, TupleBasis2D
    from slate.metadata import Metadata2D


class BlockDiagonalExplicitBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](ExplicitBasis[M, DT, B]):
    def __init__[DT1: np.generic, B1: Basis[Any, Any]](
        self: BlockDiagonalExplicitBasis[Any, DT1, B1],
        matrix: Array[
            Metadata2D[BasisMetadata, BasisStateMetadata[B1], Any],
            DT,
            BlockDiagonalBasis[
                DT,
                BasisMetadata,
                None,
                TupleBasis2D[
                    DT,
                    Basis[BasisMetadata, Any],
                    Basis[BasisStateMetadata[B1], Any],
                    None,
                ],
            ],
        ],
        *,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__(cast("Any", matrix), direction=direction, data_id=data_id)


class BlockDiagonalExplicitUnitaryBasis[
    M: BasisMetadata,
    DT: np.generic,
    B: Basis[Any, Any] = Basis[M, DT],
](ExplicitUnitaryBasis[M, DT, B]):
    def __init__[DT1: np.generic, B1: Basis[Any, Any]](
        self: BlockDiagonalExplicitUnitaryBasis[Any, DT1, B1],
        matrix: Array[
            Metadata2D[BasisMetadata, BasisStateMetadata[B], Any],
            DT,
            BlockDiagonalBasis[
                DT,
                BasisMetadata,
                None,
                TupleBasis2D[
                    DT,
                    Basis[BasisMetadata, Any],
                    Basis[BasisStateMetadata[B], Any],
                    None,
                ],
            ],
        ],
        *,
        direction: Direction = "forward",
        data_id: uuid.UUID | None = None,
        assert_unitary: bool = False,
    ) -> None:
        super().__init__(
            matrix, direction=direction, data_id=data_id, assert_unitary=assert_unitary
        )
