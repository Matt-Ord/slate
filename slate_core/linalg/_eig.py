from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate_core.array._array import Array
from slate_core.array._conversion import as_supports_type_basis, cast_basis
from slate_core.basis import (
    DiagonalBasis,
    FundamentalBasis,
)
from slate_core.basis._basis import Basis, Ctype
from slate_core.basis._basis_state_metadata import BasisStateMetadata
from slate_core.basis._block_diagonal import BlockDiagonalBasis, as_block_diagonal
from slate_core.basis._diagonal import as_diagonal
from slate_core.basis._tuple import TupleBasis, TupleBasisLike, as_tuple
from slate_core.explicit_basis import (
    ExplicitBasis,
    ExplicitUnitaryBasis,
    TrivialExplicitBasis,
)
from slate_core.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate_core.array import ArrayWithMetadata
    from slate_core.explicit_basis import (
        ExplicitDiagonalBasis,
    )
    from slate_core.metadata._metadata import SimpleMetadata
    from slate_core.metadata._tuple import TupleMetadata


def _diagonal_basis_as_explicit[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    CT: Ctype[np.complexfloating],
](
    basis: DiagonalBasis[TupleBasis[tuple[Basis[M0, CT], Basis[M1, CT]], E], CT],
) -> ExplicitDiagonalBasis[M0, M1, E, CT]:
    return DiagonalBasis(
        TupleBasis(
            (
                TrivialExplicitBasis(basis.inner.children[0]).resolve_ctype().upcast(),
                TrivialExplicitBasis(basis.inner.children[1]).resolve_ctype().upcast(),
            ),
            basis.metadata().extra,
        ).resolve_ctype()
    ).resolve_ctype()


def get_eigenvalues[M: BasisMetadata, E, DT: np.dtype[np.complexfloating]](
    array: ArrayWithMetadata[TupleMetadata[tuple[M, ...], E], DT],
) -> Array[
    FundamentalBasis[SimpleMetadata],
    np.dtype[np.complexfloating],
]:
    """Get the eigenvalues of a matrix."""
    a = np.linalg.eigvals(array.as_array())
    return Array(FundamentalBasis.from_size(a.size), a)


def _eig_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[
        TupleBasis[tuple[Basis[M0], Basis[M1]], E],
        DT,
    ],
) -> Array[
    ExplicitDiagonalBasis[M0, M1, E, Ctype[np.complexfloating]],
    np.dtype[np.complexfloating],
]:
    eig = np.linalg.eig(array.raw_data.reshape(array.basis.shape))
    matrix = cast(
        "np.ndarray[Any, np.dtype[np.complexfloating]]",
        np.transpose(eig.eigenvectors),
    )
    states_basis_0 = (
        TupleBasis(
            (
                FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=False),
                FundamentalBasis(BasisStateMetadata(array.basis.children[0])),
            )
        )
        .resolve_ctype()
        .upcast()
    )
    basis_0 = ExplicitBasis(Array(states_basis_0, matrix)).resolve_ctype().upcast()
    states_basis_1 = (
        TupleBasis(
            (
                FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=True),
                FundamentalBasis(BasisStateMetadata(array.basis.children[1])),
            )
        )
        .resolve_ctype()
        .upcast()
    )
    basis_1 = (
        ExplicitBasis(
            Array(states_basis_1, matrix),
            data_id=basis_0.inner.data_id,
            direction="backward",
        )
        .resolve_ctype()
        .upcast()
    )
    return Array(  # type: ignore unkown
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).resolve_ctype()
        ).resolve_ctype(),
        eig.eigenvalues,
    )


def _eig_from_block_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[
        BlockDiagonalBasis[
            TupleBasis[
                tuple[
                    Basis[M0, Ctype[np.complexfloating]],
                    Basis[M1, Ctype[np.complexfloating]],
                ],
                E,
            ]
        ],
        DT,
    ],
) -> Array[
    ExplicitDiagonalBasis[M0, M1, E, Ctype[np.complexfloating]],
    np.dtype[np.complexfloating],
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

    states_basis_0 = (
        BlockDiagonalBasis(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_states, is_dual=False),
                    FundamentalBasis(BasisStateMetadata(array.basis.inner.children[0])),
                )
            ).resolve_ctype(),
            array.basis.block_shape,
        )
        .resolve_ctype()
        .upcast()
    )
    basis_0 = (
        ExplicitBasis(Array(states_basis_0, eigenvectors)).resolve_ctype().upcast()
    )
    states_basis_1 = (
        BlockDiagonalBasis(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_states, is_dual=True),
                    FundamentalBasis(BasisStateMetadata(array.basis.inner.children[1])),
                )
            ).resolve_ctype(),
            array.basis.block_shape,
        )
        .resolve_ctype()
        .upcast()
    )
    basis_1 = (
        ExplicitBasis(
            Array(states_basis_1, eigenvectors),
            data_id=basis_0.inner.data_id,
            direction="backward",
        )
        .resolve_ctype()
        .upcast()
    )
    return Array(
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).resolve_ctype()
        ).resolve_ctype(),
        eigenvalues,
    )


def into_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[TupleBasisLike[tuple[M0, M1], E], DT],
) -> Array[
    ExplicitDiagonalBasis[M0, M1, E, Ctype[np.complexfloating]],
    np.dtype[np.complexfloating],
]:
    """Get the diagonal form of a matrix.

    .. literalinclude:: ../../examples/linalg.py
        :language: python
        :lines: 9-
        :linenos:
        :lineno-start: 1
        :dedent: 4
    """
    suporting = as_supports_type_basis(array, np.complexfloating)
    diagonal = as_diagonal(suporting.basis)
    if diagonal is not None:
        return suporting.with_basis(_diagonal_basis_as_explicit(diagonal))
    block_diagonal = as_block_diagonal(suporting.basis)
    if block_diagonal is not None:
        return _eig_from_block_diagonal_basis(suporting.with_basis(block_diagonal))
    tuple_basis = as_tuple(suporting.basis)
    converted = cast_basis(array.with_basis(tuple_basis.upcast()), tuple_basis)
    return _eig_from_tuple(converted)


def get_eigenvalues_hermitian[
    M: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: ArrayWithMetadata[TupleMetadata[tuple[M, ...], E], DT],
) -> Array[FundamentalBasis[SimpleMetadata], np.dtype[np.floating]]:
    """Get the eigenvalues of a Hermitian matrix."""
    a = np.linalg.eigvalsh(array.as_array())
    return Array(FundamentalBasis.from_size(a.size), a)


def _eigh_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[
        TupleBasis[
            tuple[
                Basis[M0, Ctype[np.complexfloating]],
                Basis[M1, Ctype[np.complexfloating]],
            ],
            E,
        ],
        DT,
    ],
) -> Array[
    ExplicitDiagonalBasis[M0, M1, E, Ctype[np.complexfloating]],
    np.dtype[np.complexfloating],
]:
    eig = np.linalg.eigh(array.raw_data.reshape(array.basis.shape))
    eigenvectors = cast(
        "np.ndarray[Any, np.dtype[np.complexfloating]]",
        np.transpose(eig.eigenvectors),
    )
    states_basis_0 = (
        TupleBasis(
            (
                FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=False),
                FundamentalBasis(BasisStateMetadata(array.basis.children[0])),
            )
        )
        .resolve_ctype()
        .upcast()
    )
    basis_0 = (
        ExplicitUnitaryBasis(
            Array(states_basis_0, eigenvectors),
            direction="forward",
            assert_unitary=False,
        )
        .resolve_ctype()
        .upcast()
    )
    states_basis_1 = (
        TupleBasis(
            (
                FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=True),
                FundamentalBasis(BasisStateMetadata(array.basis.children[1])),
            )
        )
        .resolve_ctype()
        .upcast()
    )
    basis_1 = (
        ExplicitUnitaryBasis(
            Array(states_basis_1, eigenvectors),
            data_id=basis_0.inner.data_id,
            direction="backward",
            assert_unitary=False,
        )
        .resolve_ctype()
        .upcast()
    )
    return Array(
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).resolve_ctype()
        ).resolve_ctype(),
        eig.eigenvalues.astype(np.complex128),
    )


def _eigh_from_block_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    array: Array[
        BlockDiagonalBasis[
            TupleBasis[
                tuple[
                    Basis[M0, Ctype[np.complexfloating]],
                    Basis[M1, Ctype[np.complexfloating]],
                ],
                E,
            ]
        ],
        np.dtype[np.complexfloating],
    ],
) -> Array[
    ExplicitDiagonalBasis[M0, M1, E, Ctype[np.complexfloating]],
    np.dtype[np.complexfloating],
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

    states_basis_0 = (
        BlockDiagonalBasis(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_states, is_dual=False),
                    FundamentalBasis(BasisStateMetadata(array.basis.inner.children[0])),
                )
            ).resolve_ctype(),
            array.basis.block_shape,
        )
        .resolve_ctype()
        .upcast()
    )
    basis_0 = (
        ExplicitUnitaryBasis(
            Array(states_basis_0, eigenvectors),
        )
        .resolve_ctype()
        .upcast()
    )
    states_basis_1 = (
        BlockDiagonalBasis(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_states, is_dual=True),
                    FundamentalBasis(BasisStateMetadata(array.basis.inner.children[1])),
                )
            ).resolve_ctype(),
            array.basis.block_shape,
        )
        .resolve_ctype()
        .upcast()
    )
    basis_1 = (
        ExplicitUnitaryBasis(
            Array(states_basis_1, eigenvectors),
            data_id=basis_0.inner.data_id,
            direction="backward",
        )
        .resolve_ctype()
        .upcast()
    )
    return Array(
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).resolve_ctype()
        ).resolve_ctype(),
        eigenvalues,
    )


def into_diagonal_hermitian[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: ArrayWithMetadata[TupleMetadata[tuple[M0, M1], E], DT],
) -> Array[
    ExplicitDiagonalBasis[M0, M1, E, Ctype[np.complexfloating]],
    np.dtype[np.complexfloating],
]:
    """Get the diagonal form of a Hermitian matrix."""
    suporting = as_supports_type_basis(array, np.complexfloating)
    diagonal = as_diagonal(suporting.basis)
    if diagonal is not None:
        return suporting.with_basis(_diagonal_basis_as_explicit(diagonal))
    block_diagonal = as_block_diagonal(suporting.basis)
    if block_diagonal is not None:
        return _eigh_from_block_diagonal_basis(suporting.with_basis(block_diagonal))
    tuple_basis = as_tuple(suporting.basis)
    converted = cast_basis(array.with_basis(tuple_basis.upcast()), tuple_basis)
    return _eigh_from_tuple(converted)
