from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slate_core.array import Array, build
from slate_core.array._conversion import cast_basis
from slate_core.basis import (
    DiagonalBasis,
    FundamentalBasis,
)
from slate_core.basis._basis import Basis, ctype
from slate_core.basis._basis_state_metadata import BasisStateMetadata
from slate_core.basis._block_diagonal import BlockDiagonalBasis, as_block_diagonal_basis
from slate_core.basis._diagonal import as_diagonal_basis
from slate_core.basis._tuple import TupleBasis, as_tuple_basis
from slate_core.explicit_basis import (
    ExplicitBasis,
    ExplicitUnitaryBasis,
    TrivialExplicitBasis,
)
from slate_core.metadata import BasisMetadata

if TYPE_CHECKING:
    from slate_core.basis._tuple import TupleBasisLike
    from slate_core.explicit_basis._explicit_basis import ExplicitBasisWithInner
    from slate_core.metadata._metadata import SimpleMetadata


def _diagonal_basis_as_explicit[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: ctype[np.complexfloating],
](
    basis: DiagonalBasis[TupleBasis[tuple[Basis[M0, DT], Basis[M1, DT]], E], DT],
) -> DiagonalBasis[
    TupleBasis[
        tuple[
            ExplicitBasisWithInner[Basis[M0, DT]], ExplicitBasisWithInner[Basis[M1, DT]]
        ],
        E,
        DT,
    ],
    DT,
]:
    return DiagonalBasis(  # type: ignore cant infer
        TupleBasis(
            (
                TrivialExplicitBasis(basis.inner.children[0]).upcast(),
                TrivialExplicitBasis(basis.inner.children[1]).upcast(),
            ),
            basis.metadata().extra,
        ).upcast()
    ).upcast()


def get_eigenvalues[M: BasisMetadata, E, DT: np.dtype[np.complexfloating]](
    array: Array[TupleBasisLike[tuple[M, ...], E], DT],
) -> Array[
    FundamentalBasis[SimpleMetadata],
    np.dtype[np.complexfloating],
]:
    """Get the eigenvalues of a matrix."""
    a = np.linalg.eigvals(array.as_array())
    return build(FundamentalBasis.from_size(a.size), a).ok()


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
    DiagonalBasis[
        TupleBasis[
            tuple[
                ExplicitBasisWithInner[Basis[M0, ctype[np.complexfloating]]],
                ExplicitBasisWithInner[Basis[M1, ctype[np.complexfloating]]],
            ],
            E,
            ctype[np.complexfloating],
        ],
        ctype[np.complexfloating],
    ],
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
        .upcast()
        .downcast_metadata()
    )
    basis_0 = ExplicitBasis(build(states_basis_0, matrix).ok()).upcast()
    states_basis_1 = (
        TupleBasis(
            (
                FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=True),
                FundamentalBasis(BasisStateMetadata(array.basis.children[1])),
            )
        )
        .upcast()
        .downcast_metadata()
    )
    basis_1 = ExplicitBasis(
        build(states_basis_1, matrix).ok(),
        data_id=basis_0.data_id,
        direction="backward",
    ).upcast()
    return build(  # type: ignore cant infer
        DiagonalBasis(TupleBasis((basis_0, basis_1), array.basis.metadata().extra)),
        eig.eigenvalues,
    ).ok()


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
                    Basis[M0, ctype[np.complexfloating]],
                    Basis[M1, ctype[np.complexfloating]],
                ],
                E,
            ]
        ],
        DT,
    ],
) -> Array[
    DiagonalBasis[
        TupleBasis[
            tuple[
                ExplicitBasisWithInner[Basis[M0, ctype[np.complexfloating]]],
                ExplicitBasisWithInner[Basis[M1, ctype[np.complexfloating]]],
            ],
            E,
            ctype[np.complexfloating],
        ],
        ctype[np.complexfloating],
    ],
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
            ).upcast(),
            array.basis.block_shape,
        )
        .upcast()
        .downcast_metadata()
    )

    basis_0 = ExplicitBasis(build(states_basis_0, eigenvectors).ok()).upcast()
    states_basis_1 = (
        BlockDiagonalBasis(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_states, is_dual=True),
                    FundamentalBasis(BasisStateMetadata(array.basis.inner.children[1])),
                )
            ).upcast(),
            array.basis.block_shape,
        )
        .upcast()
        .downcast_metadata()
    )
    basis_1 = ExplicitBasis(
        build(states_basis_1, eigenvectors).ok(),
        data_id=basis_0.data_id,
        direction="backward",
    ).upcast()

    return build(  # type: ignore cant infer
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).upcast()
        ).upcast(),
        eigenvalues,
    ).ok()


def into_diagonal[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[TupleBasisLike[tuple[M0, M1], E, ctype[np.complexfloating]], DT],
) -> Array[
    DiagonalBasis[
        TupleBasis[
            tuple[
                ExplicitBasisWithInner[Basis[M0, ctype[np.complexfloating]]],
                ExplicitBasisWithInner[Basis[M1, ctype[np.complexfloating]]],
            ],
            E,
            ctype[np.complexfloating],
        ],
        ctype[np.complexfloating],
    ],
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
    diagonal = as_diagonal_basis(array.basis)
    if diagonal is not None:
        return array.with_basis(_diagonal_basis_as_explicit(diagonal)).ok()

    block_diagonal = as_block_diagonal_basis(array.basis)
    if block_diagonal is not None:
        return _eig_from_block_diagonal_basis(array.with_basis(block_diagonal).ok())

    tuple_basis = as_tuple_basis(array.basis)
    converted = cast_basis(
        array.with_basis(tuple_basis.downcast_metadata()).ok(), tuple_basis
    ).ok()
    return _eig_from_tuple(converted)


def get_eigenvalues_hermitian[
    M: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[TupleBasisLike[tuple[M, ...], E], DT],
) -> Array[FundamentalBasis[SimpleMetadata], np.dtype[np.floating]]:
    a = np.linalg.eigvalsh(array.as_array())
    return build(FundamentalBasis.from_size(a.size), a).ok()


def _eigh_from_tuple[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[
        TupleBasis[
            tuple[
                Basis[M0, ctype[np.complexfloating]],
                Basis[M1, ctype[np.complexfloating]],
            ],
            E,
        ],
        DT,
    ],
) -> Array[
    DiagonalBasis[
        TupleBasis[
            tuple[
                ExplicitBasisWithInner[Basis[M0, ctype[np.complexfloating]]],
                ExplicitBasisWithInner[Basis[M1, ctype[np.complexfloating]]],
            ],
            E,
            ctype[np.complexfloating],
        ],
        ctype[np.complexfloating],
    ],
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
        .upcast()
        .downcast_metadata()
    )

    basis_0 = ExplicitUnitaryBasis(
        build(states_basis_0, eigenvectors).ok(),
        direction="forward",
        assert_unitary=False,
    ).upcast()
    states_basis_1 = (
        TupleBasis(
            (
                FundamentalBasis.from_size(eig.eigenvalues.size, is_dual=True),
                FundamentalBasis(BasisStateMetadata(array.basis.children[1])),
            )
        )
        .upcast()
        .downcast_metadata()
    )
    basis_1 = ExplicitUnitaryBasis(
        build(states_basis_1, eigenvectors).ok(),
        data_id=basis_0.data_id,
        direction="backward",
        assert_unitary=False,
    ).upcast()

    return build(  # type: ignore cant infer
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).upcast()
        ).upcast(),
        eig.eigenvalues.astype(np.complex128),
    ).ok()


def _eigh_from_block_diagonal_basis[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
](
    array: Array[
        BlockDiagonalBasis[
            TupleBasis[
                tuple[
                    Basis[M0, ctype[np.complexfloating]],
                    Basis[M1, ctype[np.complexfloating]],
                ],
                E,
            ]
        ],
        np.dtype[np.complexfloating],
    ],
) -> Array[
    DiagonalBasis[
        TupleBasis[
            tuple[
                ExplicitBasisWithInner[Basis[M0, ctype[np.complexfloating]]],
                ExplicitBasisWithInner[Basis[M1, ctype[np.complexfloating]]],
            ],
            E,
            ctype[np.complexfloating],
        ],
        ctype[np.complexfloating],
    ],
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
            ).upcast(),
            array.basis.block_shape,
        )
        .upcast()
        .downcast_metadata()
    )
    basis_0 = ExplicitUnitaryBasis(
        build(states_basis_0, eigenvectors).ok(),
    ).upcast()
    states_basis_1 = (
        BlockDiagonalBasis(
            TupleBasis(
                (
                    FundamentalBasis.from_size(n_states, is_dual=True),
                    FundamentalBasis(BasisStateMetadata(array.basis.inner.children[1])),
                )
            ).upcast(),
            array.basis.block_shape,
        )
        .upcast()
        .downcast_metadata()
    )
    basis_1 = ExplicitUnitaryBasis(
        build(states_basis_1, eigenvectors).ok(),
        data_id=basis_0.data_id,
        direction="backward",
    ).upcast()

    return build(  # type: ignore cant infer
        DiagonalBasis(
            TupleBasis((basis_0, basis_1), array.basis.metadata().extra).upcast()
        ).upcast(),
        eigenvalues,
    ).ok()


def into_diagonal_hermitian[
    M0: BasisMetadata,
    M1: BasisMetadata,
    E,
    DT: np.dtype[np.complexfloating],
](
    array: Array[TupleBasisLike[tuple[M0, M1], E, ctype[np.complexfloating]], DT],
) -> Array[
    DiagonalBasis[
        TupleBasis[
            tuple[
                ExplicitBasisWithInner[Basis[M0, ctype[np.complexfloating]]],
                ExplicitBasisWithInner[Basis[M1, ctype[np.complexfloating]]],
            ],
            E,
            ctype[np.complexfloating],
        ],
        ctype[np.complexfloating],
    ],
    np.dtype[np.complexfloating],
]:
    diagonal = as_diagonal_basis(array.basis)
    if diagonal is not None:
        return array.with_basis(_diagonal_basis_as_explicit(diagonal)).ok()

    block_diagonal = as_block_diagonal_basis(array.basis)
    if block_diagonal is not None:
        array.with_basis(block_diagonal).ok()
        return _eigh_from_block_diagonal_basis(array.with_basis(block_diagonal).ok())

    tuple_basis = as_tuple_basis(array.basis)
    converted = cast_basis(
        array.with_basis(tuple_basis.downcast_metadata()).ok(), tuple_basis
    ).ok()
    return _eigh_from_tuple(converted)
