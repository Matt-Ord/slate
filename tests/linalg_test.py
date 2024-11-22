from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array import SlateArray, convert_array
from slate.basis._tuple import fundamental_tuple_basis_from_shape
from slate.linalg import into_diagonal
from slate.linalg._eig import (
    get_eigenvalues,
    get_eigenvalues_hermitian,
    into_diagonal_hermitian,
)

if TYPE_CHECKING:
    from slate.basis import TupleBasis
    from slate.metadata import BasisMetadata, SimpleMetadata, StackedMetadata
    from slate.metadata.stacked import Metadata2D


@pytest.fixture
def slate_array_stacked() -> (
    SlateArray[
        StackedMetadata[SimpleMetadata, None],
        np.complex128,
        TupleBasis[BasisMetadata, None, np.generic],
    ]
):
    rng = np.random.default_rng()
    shape = (10, 10)
    data = rng.random(shape) + 1j * rng.random(shape)
    return SlateArray.from_array(data)


def _test_into_diagonal(
    array: SlateArray[
        Metadata2D[BasisMetadata, BasisMetadata, None],
        np.complexfloating[Any, Any],
    ],
) -> None:
    diagonal = into_diagonal(array)

    eigenvalues = get_eigenvalues(array)
    np.testing.assert_allclose(eigenvalues.as_array(), diagonal.raw_data)

    full_as_diagonal = convert_array(array, diagonal.basis)
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data)

    diagonal_as_full = convert_array(diagonal, array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array())


def test_linalg_complex() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray(fundamental_tuple_basis_from_shape((10, 10)), data)

    _test_into_diagonal(array)


def _test_into_diagonal_hermitian(
    array: SlateArray[
        Metadata2D[BasisMetadata, BasisMetadata, None],
        np.complexfloating[Any, Any],
    ],
) -> None:
    diagonal = into_diagonal_hermitian(array)

    eigenvalues = get_eigenvalues_hermitian(array)
    np.testing.assert_allclose(eigenvalues.as_array(), diagonal.raw_data)

    full_as_diagonal = convert_array(array, diagonal.basis)
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data)

    diagonal_as_full = convert_array(diagonal, array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array())


def test_linalg_diagonal() -> None:
    rng = np.random.default_rng()
    data = rng.random(10)
    array = SlateArray(
        fundamental_tuple_basis_from_shape((10, 10)),
        np.diag(data).astype(np.complex128),
    )

    _test_into_diagonal_hermitian(array)


def test_linalg_complex_hermitian() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    data += np.conj(np.transpose(data))
    array = SlateArray(fundamental_tuple_basis_from_shape((10, 10)), data)

    _test_into_diagonal_hermitian(array)


def test_linalg_real_hermitian() -> None:
    data = np.array([[1, 1, 2], [-1, 1, 0], [0, 0, 2]])
    data += np.conj(np.transpose(data))
    array = SlateArray(fundamental_tuple_basis_from_shape((3, 3)), data)

    _test_into_diagonal_hermitian(array)
