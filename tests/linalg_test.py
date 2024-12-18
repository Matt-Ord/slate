from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array import Array, with_basis
from slate.basis import (
    TupleBasis2D,
    as_tuple_basis,
    from_shape,
)
from slate.linalg import into_diagonal
from slate.linalg._eig import (
    get_eigenvalues_hermitian,
    into_diagonal_hermitian,
)

if TYPE_CHECKING:
    from slate.basis import Basis, TupleBasis
    from slate.metadata import BasisMetadata, SimpleMetadata, StackedMetadata
    from slate.metadata.stacked import Metadata2D


@pytest.fixture
def slate_array_stacked() -> (
    Array[
        StackedMetadata[SimpleMetadata, None],
        np.complexfloating,
        TupleBasis[BasisMetadata, None, np.generic],
    ]
):
    rng = np.random.default_rng()
    shape = (10, 10)
    data = rng.random(shape) + 1j * rng.random(shape)
    return Array.from_array(data)


def _test_into_diagonal(
    array: Array[
        Metadata2D[BasisMetadata, BasisMetadata, None],
        np.complexfloating[Any, Any],
    ],
) -> None:
    diagonal = into_diagonal(array)

    original_as_tuple = as_tuple_basis(array.basis)
    diagonal_basis = diagonal.basis.inner
    assert diagonal_basis[0].is_dual == original_as_tuple[0].is_dual
    assert diagonal_basis[1].is_dual == original_as_tuple[1].is_dual

    assert (
        as_tuple_basis(diagonal_basis[0].eigenvectors.basis)[1].is_dual
        == original_as_tuple[0].is_dual
    )
    assert (
        as_tuple_basis(diagonal_basis[1].eigenvectors.basis)[1].is_dual
        == original_as_tuple[1].is_dual
    )

    full_as_diagonal = with_basis(array, diagonal.basis)
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data, atol=1e-15)

    diagonal_as_full = with_basis(diagonal, array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data, atol=1e-15)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array(), atol=1e-15)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
    ],
)
def test_linalg_complex(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.shape) + 1j * rng.random(basis.shape)

    array = Array(basis, data)
    _test_into_diagonal(array)


def _test_into_diagonal_hermitian(
    array: Array[
        Metadata2D[BasisMetadata, BasisMetadata, None],
        np.complexfloating[Any, Any],
    ],
) -> None:
    diagonal = into_diagonal_hermitian(array)

    original_as_tuple = as_tuple_basis(array.basis)
    diagonal_basis = diagonal.basis.inner
    assert diagonal_basis[0].is_dual == original_as_tuple[0].is_dual
    assert diagonal_basis[1].is_dual == original_as_tuple[1].is_dual

    assert (
        as_tuple_basis(diagonal_basis[0].eigenvectors.basis)[1].is_dual
        == original_as_tuple[0].is_dual
    )
    assert (
        as_tuple_basis(diagonal_basis[1].eigenvectors.basis)[1].is_dual
        == original_as_tuple[1].is_dual
    )

    eigenvalues = get_eigenvalues_hermitian(array)
    np.testing.assert_allclose(eigenvalues.as_array(), diagonal.raw_data)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array())

    full_as_diagonal = array.with_basis(diagonal.basis)
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
    ],
)
def test_linalg_diagonal(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    rng = np.random.default_rng()
    data = rng.random(10)
    array = Array(
        basis,
        np.diag(data).astype(np.complex128),
    )

    _test_into_diagonal_hermitian(array)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
    ],
)
def test_linalg_complex_hermitian(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.shape) + 1j * rng.random(basis.shape)
    data += np.conj(np.transpose(data))

    array = Array(basis, data)
    _test_into_diagonal_hermitian(array)


@pytest.mark.parametrize(
    "basis",
    [
        from_shape((10, 10)),
        from_shape((10, 10), is_dual=(False, True)),
        from_shape((10, 10), is_dual=(True, False)),
        from_shape((10, 10), is_dual=(True, True)),
    ],
)
def test_linalg_real_hermitian(
    basis: TupleBasis2D[
        np.generic,
        Basis[SimpleMetadata, np.generic],
        Basis[SimpleMetadata, np.generic],
        None,
    ],
) -> None:
    rng = np.random.default_rng()
    data = rng.random(basis.shape).astype(np.complex128)
    data += np.conj(np.transpose(data))
    array = Array(basis, data)

    _test_into_diagonal_hermitian(array)
