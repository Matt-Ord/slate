from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.basis.fundamental import FundamentalBasis
from slate.basis.stacked import (
    DiagonalBasis,
    TupleBasis,
    as_tuple_basis,
    diagonal_basis,
    tuple_basis,
)
from slate.explicit_basis import ExplicitBasis, ExplicitUnitaryBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.metadata.stacked.stacked import StackedMetadata


def eig_vals[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[BasisMetadata, np.complex128, FundamentalBasis[BasisMetadata]]:
    """Get the eigenvalues of a matrix."""
    a = np.linalg.eigvals(array.as_array())
    return SlateArray(FundamentalBasis.from_shape(a.shape), a)


def _eig_from_tuple[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT, TupleBasis[M, E, np.generic]],
) -> SlateArray[
    StackedMetadata[M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitBasis[M, DT],
        ExplicitBasis[M, DT],
        E,
    ],
]:
    eig = np.linalg.eig(array.raw_data.reshape(array.basis.shape))
    states_basis = tuple_basis(
        (FundamentalBasis.from_shape((eig.eigenvalues.size,)), array.basis[1])
    )

    basis_0 = ExplicitBasis(SlateArray(states_basis, np.transpose(eig.eigenvectors)))
    basis_1 = basis_0.conjugate_basis()

    return SlateArray(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eig.eigenvalues,
    )


def eig[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[
    StackedMetadata[M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitBasis[M, DT],
        ExplicitBasis[M, DT],
        E,
    ],
]:
    """Get the diagonal form of a matrix.

    .. literalinclude:: ../../examples/linalg.py
        :language: python
        :lines: 9-
        :linenos:
        :lineno-start: 1
        :dedent: 4
    """
    assert array.basis.n_dim == 2  # noqa: PLR2004
    tuple_basis = as_tuple_basis(array.basis)
    return _eig_from_tuple(convert_array(array, tuple_basis))


def eigh_vals[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[BasisMetadata, np.float64, FundamentalBasis[BasisMetadata]]:
    a = np.linalg.eigvalsh(array.as_array())
    return SlateArray(FundamentalBasis.from_shape(a.shape), a)


def _eigh_from_tuple[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT, TupleBasis[M, E, np.generic]],
) -> SlateArray[
    StackedMetadata[M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[M, Any],
        ExplicitUnitaryBasis[M, Any],
        E,
    ],
]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    eig = np.linalg.eigh(array.raw_data.reshape(array.basis.shape))

    states_basis = tuple_basis(
        (FundamentalBasis.from_shape((eig.eigenvalues.size,)), array.basis[1])
    )
    basis_0 = ExplicitUnitaryBasis(
        SlateArray(states_basis, np.transpose(eig.eigenvectors))
    )
    basis_1 = basis_0.conjugate_basis()

    return SlateArray(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eig.eigenvalues.astype(np.complex128),
    )


def eigh[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[
    StackedMetadata[M, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[M, Any],
        ExplicitUnitaryBasis[M, Any],
        E,
    ],
]:
    assert array.basis.n_dim == 2  # noqa: PLR2004
    tuple_basis = as_tuple_basis(array.basis)
    return _eigh_from_tuple(convert_array(array, tuple_basis))
