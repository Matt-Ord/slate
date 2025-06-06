from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from slate_core import array as _array
from slate_core.array import Array
from slate_core.basis import (
    BlockDiagonalBasis,
    DiagonalBasis,
    TupleBasisLike,
    as_tuple,
    from_metadata,
    from_shape,
)
from slate_core.linalg import (
    get_eigenvalues_hermitian,
    into_diagonal,
    into_diagonal_hermitian,
)

if TYPE_CHECKING:
    from slate_core.basis import Basis, Ctype, TupleBasis
    from slate_core.basis._fundamental import FundamentalBasis
    from slate_core.metadata import BasisMetadata, SimpleMetadata
    from slate_core.metadata._tuple import TupleMetadata


@pytest.fixture
def slate_array_stacked() -> Array[
    TupleBasis[tuple[FundamentalBasis, ...], None, Ctype[np.generic]],
    np.dtype[np.complexfloating],
]:
    rng = np.random.default_rng()
    shape = (10, 10)
    data = rng.random(shape) + 1j * rng.random(shape)
    return Array.from_array(data)


def _test_into_diagonal(
    array: Array[
        TupleBasisLike[
            tuple[BasisMetadata, BasisMetadata], None, Ctype[np.complexfloating]
        ],
        np.dtype[np.complexfloating],
    ],
) -> None:
    diagonal = into_diagonal(array)

    original_as_tuple = as_tuple(array.basis)
    diagonal_basis = diagonal.basis.inner
    assert diagonal_basis.children[0].is_dual == original_as_tuple.children[0].is_dual
    assert diagonal_basis.children[1].is_dual == original_as_tuple.children[1].is_dual

    assert (
        as_tuple(diagonal_basis.children[0].inner.eigenvectors().basis)
        .children[1]
        .is_dual
        == original_as_tuple.children[0].is_dual
    )
    assert (
        as_tuple(diagonal_basis.children[1].inner.eigenvectors().basis)
        .children[1]
        .is_dual
        == original_as_tuple.children[1].is_dual
    )

    full_as_diagonal = array.with_basis(diagonal.basis.upcast())
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data, atol=1e-15)

    diagonal_as_full = diagonal.with_basis(array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data, atol=1e-15)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array(), atol=1e-15)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
        DiagonalBasis(from_shape((10, 10))),
        BlockDiagonalBasis(from_shape((10, 10)), (2, 2)),
    ],
)
def test_linalg_complex(
    basis: Basis[
        TupleMetadata[tuple[BasisMetadata, BasisMetadata], None], Ctype[np.generic]
    ],
) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.size) + 1j * rng.random(basis.size)

    array = Array(basis, data)
    _test_into_diagonal(array)


def _test_into_diagonal_hermitian(
    array: Array[
        TupleBasisLike[
            tuple[SimpleMetadata, SimpleMetadata], None, Ctype[np.complexfloating]
        ],
        np.dtype[np.complexfloating],
    ],
) -> None:
    diagonal = into_diagonal_hermitian(array)

    original_as_tuple = as_tuple(array.basis)
    diagonal_basis = diagonal.basis.inner
    assert diagonal_basis.children[0].is_dual == original_as_tuple.children[0].is_dual
    assert diagonal_basis.children[1].is_dual == original_as_tuple.children[1].is_dual

    assert (
        as_tuple(diagonal_basis.children[0].inner.eigenvectors().basis)
        .children[1]
        .is_dual
        == original_as_tuple.children[0].is_dual
    )
    assert (
        as_tuple(diagonal_basis.children[1].inner.eigenvectors().basis)
        .children[1]
        .is_dual
        == original_as_tuple.children[1].is_dual
    )

    eigenvalues = get_eigenvalues_hermitian(array)
    np.testing.assert_allclose(
        np.sort(eigenvalues.as_array()), np.sort(diagonal.raw_data)
    )

    np.testing.assert_allclose(diagonal.as_array(), array.as_array())
    full_as_diagonal = array.with_basis(diagonal.basis.upcast())
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data)

    diagonal_as_full = diagonal.with_basis(array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data, atol=1e-15)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
        DiagonalBasis(from_shape((10, 10))),
        BlockDiagonalBasis(from_shape((10, 10)), (2, 2)),
    ],
)
def test_linalg_diagonal(
    basis: Basis[
        TupleMetadata[tuple[SimpleMetadata, SimpleMetadata], None], Ctype[np.generic]
    ],
) -> None:
    rng = np.random.default_rng()
    fundamental = from_metadata(basis.metadata())
    data = np.diag(rng.random(fundamental.shape[0])).astype(np.complex128)
    array = Array(fundamental, data).with_basis(basis)

    _test_into_diagonal_hermitian(array)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
        DiagonalBasis(from_shape((3, 3))),
        BlockDiagonalBasis(from_shape((10, 10)), (2, 2)),
    ],
)
def test_linalg_complex_hermitian(
    basis: Basis[
        TupleMetadata[tuple[SimpleMetadata, SimpleMetadata], None],
        Ctype[np.generic],
    ],
) -> None:
    rng = np.random.default_rng()

    array = Array(basis, rng.random(basis.size) + 1j * rng.random(basis.size))
    array += _array.conjugate(_array.transpose(array))
    array = array.with_basis(basis)

    _test_into_diagonal_hermitian(array)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
        DiagonalBasis(from_shape((3, 3))),
        BlockDiagonalBasis(from_shape((10, 10)), (2, 2)),
    ],
)
def test_linalg_real_hermitian(
    basis: Basis[
        TupleMetadata[tuple[SimpleMetadata, SimpleMetadata], None], Ctype[np.generic]
    ],
) -> None:
    rng = np.random.default_rng()
    array = Array(basis, rng.random(basis.size).astype(np.complex128))
    array += _array.conjugate(_array.transpose(array))
    array = array.with_basis(basis)

    _test_into_diagonal_hermitian(array)
