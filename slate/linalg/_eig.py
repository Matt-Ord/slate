from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis._basis import Basis, FundamentalBasis
from slate.basis.stacked._diagonal_basis import DiagonalBasis
from slate.basis.stacked._tuple_basis import (
    TupleBasis,
    fundamental_tuple_basis_from_metadata,
    tuple_basis,
)
from slate.explicit_basis._explicit_basis import ExplicitBasis, ExplicitUnitaryBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata.stacked import StackedMetadata


def eigvals[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[
        DT,
        Basis[StackedMetadata[M, E], np.generic],
    ],
) -> SlateArray[np.complex128, FundamentalBasis[BasisMetadata]]:
    a = np.linalg.eigvals(array.as_array())
    return SlateArray(FundamentalBasis.from_shape(a.shape), a)


def _eig_from_tuple[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[DT, TupleBasis[M, E, np.generic]],
) -> SlateArray[
    DT,
    DiagonalBasis[
        DT,
        ExplicitBasis[BasisMetadata, DT],
        ExplicitBasis[BasisMetadata, DT],
        E,
    ],
]:
    eig = np.linalg.eig(array.raw_data.reshape(array.basis.shape))

    states_basis = tuple_basis(
        (FundamentalBasis.from_shape((eig.eigenvalues.size,)), array.basis[1])
    )

    basis = ExplicitBasis(SlateArray(states_basis, eig.eigenvectors))
    return SlateArray(
        DiagonalBasis(tuple_basis((basis, basis), array.basis.metadata.extra)),
        eig.eigenvalues,
    )


def eig[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[
        DT,
        Basis[StackedMetadata[M, E], np.generic],
    ],
) -> SlateArray[
    DT,
    DiagonalBasis[
        DT,
        ExplicitBasis[BasisMetadata, DT],
        ExplicitBasis[BasisMetadata, DT],
        E,
    ],
]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    tuple_basis = fundamental_tuple_basis_from_metadata(array.basis.metadata)
    return _eig_from_tuple(convert_array(array, tuple_basis))


def _eigh_from_tuple[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[DT, TupleBasis[M, E, np.generic]],
) -> SlateArray[
    np.float64,
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[BasisMetadata, DT],
        ExplicitUnitaryBasis[BasisMetadata, DT],
        E,
    ],
]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    eig = np.linalg.eigh(array.raw_data.reshape(array.basis.shape))

    states_basis = tuple_basis((FundamentalBasis.from_shape((4,)), array.basis[1]))
    basis = ExplicitUnitaryBasis(SlateArray(states_basis, eig.eigenvectors))
    return SlateArray(
        DiagonalBasis(tuple_basis((basis, basis), array.basis.metadata.extra)),
        eig.eigenvalues,
    )


def eigh[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[
        DT,
        Basis[StackedMetadata[M, E], np.generic],
    ],
) -> SlateArray[
    np.float64,
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[BasisMetadata, DT],
        ExplicitUnitaryBasis[BasisMetadata, DT],
        E,
    ],
]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    tuple_basis = fundamental_tuple_basis_from_metadata(array.basis.metadata)
    return _eigh_from_tuple(convert_array(array, tuple_basis))
