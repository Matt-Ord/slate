from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array import SlateArray, convert_array
from slate.basis import (
    DiagonalBasis,
    FundamentalBasis,
    as_tuple_basis,
    diagonal_basis,
    tuple_basis,
)
from slate.basis._diagonal import as_diagonal_basis
from slate.explicit_basis import ExplicitBasis, ExplicitUnitaryBasis
from slate.explicit_basis._explicit_basis import TrivialExplicitBasis
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.basis._tuple import TupleBasis2D
    from slate.metadata import StackedMetadata
    from slate.metadata.stacked import Metadata2D


def _diagonal_basis_as_explicit[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.complexfloating[Any, Any],
](
    basis: DiagonalBasis[DT, Basis[M0, DT], Basis[M1, DT], E],
) -> DiagonalBasis[
    DT,
    ExplicitUnitaryBasis[M0, Any],
    ExplicitUnitaryBasis[M1, Any],
    E,
]:
    return diagonal_basis(
        (
            TrivialExplicitBasis(basis.inner[0]),
            TrivialExplicitBasis(basis.inner[1]),
        ),
        basis.metadata().extra,
    )


def get_eigenvalues[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[BasisMetadata, np.complex128, FundamentalBasis[BasisMetadata]]:
    """Get the eigenvalues of a matrix."""
    a = np.linalg.eigvals(array.as_array())
    return SlateArray(FundamentalBasis.from_shape(a.shape), a)


def _eig_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.complexfloating[Any, Any],
](
    array: SlateArray[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E],
    ],
) -> SlateArray[
    Metadata2D[M0, M1, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitBasis[M0, DT],
        ExplicitBasis[M1, DT],
        E,
    ],
]:
    eig = np.linalg.eig(array.raw_data.reshape(array.basis.shape))
    states_basis = tuple_basis(
        (FundamentalBasis.from_shape((eig.eigenvalues.size,)), array.basis[1])
    )

    basis_0 = ExplicitBasis(SlateArray(states_basis, np.transpose(eig.eigenvectors)))
    basis_1 = basis_0.inverse_basis()

    return SlateArray(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eig.eigenvalues,
    )


def into_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.complexfloating[Any, Any],
](
    array: SlateArray[Metadata2D[M0, M1, E], DT],
) -> SlateArray[
    Metadata2D[M0, M1, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitBasis[M0, DT],
        ExplicitBasis[M1, DT],
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
    diagonal = as_diagonal_basis(array.basis)
    if diagonal is not None:
        return array.with_basis(_diagonal_basis_as_explicit(diagonal))

    tuple_basis = as_tuple_basis(array.basis)
    return _eig_from_tuple(convert_array(array, tuple_basis))


def get_eigenvalues_hermitian[M: BasisMetadata, E, DT: np.complexfloating[Any, Any]](
    array: SlateArray[StackedMetadata[M, E], DT],
) -> SlateArray[BasisMetadata, np.float64, FundamentalBasis[BasisMetadata]]:
    a = np.linalg.eigvalsh(array.as_array())
    return SlateArray(FundamentalBasis.from_shape(a.shape), a)


def _eigh_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.complexfloating[Any, Any],
](
    array: SlateArray[
        Metadata2D[M0, M1, E], DT, TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E]
    ],
) -> SlateArray[
    Metadata2D[M0, M1, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[M0, Any],
        ExplicitUnitaryBasis[M1, Any],
        E,
    ],
]:
    eig = np.linalg.eigh(array.raw_data.reshape(array.basis.shape))

    states_basis = tuple_basis(
        (FundamentalBasis.from_shape((eig.eigenvalues.size,)), array.basis[1])
    )
    basis_0 = ExplicitUnitaryBasis(
        SlateArray(states_basis, np.transpose(eig.eigenvectors))
    )
    basis_1 = basis_0.inverse_basis()

    return SlateArray(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eig.eigenvalues.astype(np.complex128),
    )


def into_diagonal_hermitian[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.complexfloating[Any, Any],
](
    array: SlateArray[Metadata2D[M0, M1, E], DT],
) -> SlateArray[
    Metadata2D[M0, M1, E],
    np.complex128,
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[M0, Any],
        ExplicitUnitaryBasis[M1, Any],
        E,
    ],
]:
    diagonal = as_diagonal_basis(array.basis)
    if diagonal is not None:
        return array.with_basis(_diagonal_basis_as_explicit(diagonal))

    tuple_basis = as_tuple_basis(array.basis)
    return _eigh_from_tuple(convert_array(array, tuple_basis))
