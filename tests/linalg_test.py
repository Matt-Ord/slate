from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array.array import SlateArray
from slate.array.conversion import convert_array
from slate.linalg._eig import eig, eigh, eigvals, eigvalsh

if TYPE_CHECKING:
    from slate.basis._basis import Basis
    from slate.basis.stacked_basis import TupleBasis
    from slate.metadata import SimpleMetadata
    from slate.metadata.stacked import StackedMetadata


@pytest.fixture
def slate_array_stacked() -> (
    SlateArray[np.complex128, TupleBasis[SimpleMetadata, None, np.generic]]
):
    rng = np.random.default_rng()
    shape = (10, 10)
    data = rng.random(shape) + 1j * rng.random(shape)
    return SlateArray.from_array(data)


def _test_eig(
    array: SlateArray[
        np.complexfloating[Any, Any],
        Basis[StackedMetadata[SimpleMetadata, None], Any],
    ],
) -> None:
    diagonal = eig(array)

    eigenvalues = eigvals(array)
    np.testing.assert_allclose(eigenvalues.as_array(), diagonal.raw_data)

    full_as_diagonal = convert_array(array, diagonal.basis)
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data)

    diagonal_as_full = convert_array(diagonal, array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array())


def test_linalg_complex() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = SlateArray.from_array(data)

    _test_eig(array)


def _test_eigh(
    array: SlateArray[
        np.complexfloating[Any, Any],
        Basis[StackedMetadata[SimpleMetadata, None], Any],
    ],
) -> None:
    diagonal = eigh(array)

    eigenvalues = eigvalsh(array)
    np.testing.assert_allclose(eigenvalues.as_array(), diagonal.raw_data)

    full_as_diagonal = convert_array(array, diagonal.basis)
    np.testing.assert_allclose(full_as_diagonal.raw_data, diagonal.raw_data)

    diagonal_as_full = convert_array(diagonal, array.basis)
    np.testing.assert_allclose(diagonal_as_full.raw_data, array.raw_data)

    np.testing.assert_allclose(diagonal.as_array(), array.as_array())


def test_linalg_diagonal() -> None:
    rng = np.random.default_rng()
    data = rng.random(10)
    array = SlateArray.from_array(np.diag(data).astype(np.complex128))

    _test_eigh(array)


def test_linalg_complex_hermitian() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    data += np.conj(np.transpose(data))
    array = SlateArray.from_array(data)

    _test_eigh(array)


def test_linalg_real_hermitian() -> None:
    data = np.array([[1, 1, 2], [-1, 1, 0], [0, 0, 2]])
    data += np.conj(np.transpose(data))
    array = SlateArray.from_array(data)

    _test_eigh(array)
