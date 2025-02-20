from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from slate.array import Array, with_basis
from slate.basis import (
    DiagonalBasis,
    FundamentalBasis,
    as_tuple_basis,
    diagonal_basis,
    tuple_basis,
)
from slate.basis._basis_state_metadata import BasisStateMetadata
from slate.basis._block_diagonal import BlockDiagonalBasis, as_block_diagonal_basis
from slate.basis._diagonal import as_diagonal_basis
from slate.explicit_basis import (
    ExplicitBasis,
    ExplicitUnitaryBasis,
    TrivialExplicitBasis,
)
from slate.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.basis._tuple import TupleBasis2D
    from slate.metadata import TupleMetadata
    from slate.metadata._metadata import SimpleMetadata
    from slate.metadata.stacked import Metadata2D


def _diagonal_basis_as_explicit[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
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


def get_eigenvalues[M: BasisMetadata, E, DT: np.dtype[np.complexfloating[Any, Any]]](
    array: Array[TupleMetadata[M, E], DT],
) -> Array[
    BasisMetadata,
    np.dtype[np.complexfloating[Any, Any]],
    FundamentalBasis[SimpleMetadata],
]:
    """Get the eigenvalues of a matrix."""
    a = np.linalg.eigvals(array.as_array())
    return Array(FundamentalBasis.from_size(a.size), a)


def _eig_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E],
    ],
) -> Array[
    Metadata2D[M0, M1, E],
    np.dtype[np.complexfloating[Any, Any]],
    DiagonalBasis[
        DT,
        ExplicitBasis[M0, DT],
        ExplicitBasis[M1, DT],
        E,
    ],
]:
    eig = np.linalg.eig(array.raw_data.reshape(array.basis.shape))

    states_basis_0 = tuple_basis(
        (
            FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=False),
            FundamentalBasis(BasisStateMetadata(array.basis[0])),
        )
    )
    basis_0 = ExplicitBasis(
        Array(states_basis_0, np.transpose(eig.eigenvectors)),
    )
    states_basis_1 = tuple_basis(
        (
            FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=True),
            FundamentalBasis(BasisStateMetadata(array.basis[1])),
        )
    )
    basis_1 = ExplicitBasis(
        Array(states_basis_1, np.transpose(eig.eigenvectors)),
        data_id=basis_0.data_id,
        direction="backward",
    )
    return Array(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eig.eigenvalues,
    )


def _eig_from_block_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        BlockDiagonalBasis[
            DT, Any, E, TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E]
        ],
    ],
) -> Array[
    Metadata2D[M0, M1, E],
    np.dtype[np.complexfloating[Any, Any]],
    DiagonalBasis[
        DT,
        ExplicitBasis[M0, Any],
        ExplicitBasis[M1, Any],
        E,
    ],
]:
    assert array.basis.inner.shape[0] == array.basis.inner.shape[1]
    n_states = array.basis.inner.shape[0]
    n_repeats = array.basis.n_repeats
    n_block = array.basis.block_shape[0]

    eigenvalues = np.empty((n_repeats, n_block), dtype=array.dtype)
    eigenvectors = np.empty((n_repeats, n_block, n_block), dtype=array.dtype)
    raw_data = array.raw_data.reshape((n_repeats, n_block, n_block))
    for i in range(n_repeats):
        eig = np.linalg.eig(raw_data[i])
        eigenvalues[i] = eig.eigenvalues
        eigenvectors[i] = np.transpose(eig.eigenvectors)

    states_basis_0 = BlockDiagonalBasis(
        tuple_basis(
            (
                FundamentalBasis.from_size(n_states, is_dual=False),
                FundamentalBasis(BasisStateMetadata(array.basis.inner[0])),
            )
        ),
        array.basis.block_shape,
    )
    basis_0 = ExplicitBasis(Array(states_basis_0, eigenvectors))
    states_basis_1 = BlockDiagonalBasis(
        tuple_basis(
            (
                FundamentalBasis.from_size(n_states, is_dual=True),
                FundamentalBasis(BasisStateMetadata(array.basis.inner[1])),
            )
        ),
        array.basis.block_shape,
    )
    basis_1 = ExplicitBasis(
        Array(states_basis_1, eigenvectors),
        data_id=basis_0.data_id,
        direction="backward",
    )

    return Array(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eigenvalues,
    )


def into_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[Metadata2D[M0, M1, E], DT],
) -> Array[
    Metadata2D[M0, M1, E],
    np.dtype[np.complexfloating[Any, Any]],
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

    block_diagonal = as_block_diagonal_basis(array.basis)
    if block_diagonal is not None:
        return _eig_from_block_diagonal_basis(with_basis(array, block_diagonal))

    tuple_basis = as_tuple_basis(array.basis)
    return _eig_from_tuple(with_basis(array, tuple_basis))


def get_eigenvalues_hermitian[
    M: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[TupleMetadata[M, E], DT],
) -> Array[BasisMetadata, np.floating, FundamentalBasis[SimpleMetadata]]:
    a = np.linalg.eigvalsh(array.as_array())
    return Array(FundamentalBasis.from_size(a.size), a)


def _eigh_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[
        Metadata2D[M0, M1, E], DT, TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E]
    ],
) -> Array[
    Metadata2D[M0, M1, E],
    np.dtype[np.complexfloating[Any, Any]],
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[M0, Any],
        ExplicitUnitaryBasis[M1, Any],
        E,
    ],
]:
    eig = np.linalg.eigh(array.raw_data.reshape(array.basis.shape))

    states_basis_0 = tuple_basis(
        (
            FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=False),
            FundamentalBasis(BasisStateMetadata(array.basis[0])),
        )
    )
    basis_0 = ExplicitUnitaryBasis(
        Array(states_basis_0, np.transpose(eig.eigenvectors)),
        direction="forward",
        assert_unitary=False,
    )
    states_basis_1 = tuple_basis(
        (
            FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=True),
            FundamentalBasis(BasisStateMetadata(array.basis[1])),
        )
    )
    basis_1 = ExplicitUnitaryBasis(
        Array(states_basis_1, (np.transpose(eig.eigenvectors))),
        data_id=basis_0.data_id,
        direction="backward",
        assert_unitary=False,
    )

    return Array(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eig.eigenvalues.astype(np.complex128),
    )


def _eigh_from_block_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[
        Metadata2D[M0, M1, E],
        DT,
        BlockDiagonalBasis[
            DT, Any, E, TupleBasis2D[DT, Basis[M0, DT], Basis[M1, DT], E]
        ],
    ],
) -> Array[
    Metadata2D[M0, M1, E],
    np.dtype[np.complexfloating[Any, Any]],
    DiagonalBasis[
        DT,
        ExplicitUnitaryBasis[M0, Any],
        ExplicitUnitaryBasis[M1, Any],
        E,
    ],
]:
    assert array.basis.inner.shape[0] == array.basis.inner.shape[1]
    n_states = array.basis.inner.shape[0]
    n_repeats = array.basis.n_repeats
    n_block = array.basis.block_shape[0]

    eigenvalues = np.empty((n_repeats, n_block), dtype=array.dtype)
    eigenvectors = np.empty((n_repeats, n_block, n_block), dtype=array.dtype)
    raw_data = array.raw_data.reshape((n_repeats, n_block, n_block))
    for i in range(n_repeats):
        eig = np.linalg.eigh(raw_data[i])
        eigenvalues[i] = eig.eigenvalues
        eigenvectors[i] = np.transpose(eig.eigenvectors)

    states_basis_0 = BlockDiagonalBasis(
        tuple_basis(
            (
                FundamentalBasis.from_size(n_states, is_dual=False),
                FundamentalBasis(BasisStateMetadata(array.basis.inner[0])),
            )
        ),
        array.basis.block_shape,
    )
    basis_0 = ExplicitUnitaryBasis(
        Array(states_basis_0, eigenvectors),
    )
    states_basis_1 = BlockDiagonalBasis(
        tuple_basis(
            (
                FundamentalBasis.from_size(n_states, is_dual=True),
                FundamentalBasis(BasisStateMetadata(array.basis.inner[1])),
            )
        ),
        array.basis.block_shape,
    )
    basis_1 = ExplicitUnitaryBasis(
        Array(states_basis_1, eigenvectors),
        data_id=basis_0.data_id,
        direction="backward",
    )

    return Array(
        diagonal_basis((basis_0, basis_1), array.basis.metadata().extra),
        eigenvalues,
    )


def into_diagonal_hermitian[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating[Any, Any]],
](
    array: Array[Metadata2D[M0, M1, E], DT],
) -> Array[
    Metadata2D[M0, M1, E],
    np.dtype[np.complexfloating[Any, Any]],
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

    block_diagonal = as_block_diagonal_basis(array.basis)
    if block_diagonal is not None:
        return _eigh_from_block_diagonal_basis(with_basis(array, block_diagonal))

    tuple_basis = as_tuple_basis(array.basis)
    return _eigh_from_tuple(with_basis(array, tuple_basis))
