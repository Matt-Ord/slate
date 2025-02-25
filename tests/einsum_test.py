from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate._einsum._einstein_index import (
    EinsteinIndex,
    NestedEinsteinIndex,
    parse_einsum_index,
)
from slate.array._array import ArrayBuilder
from slate.basis import (
    CroppedBasis,
    FundamentalBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    TupleBasis,
    with_child,
)
from slate.linalg._eig import into_diagonal, into_diagonal_hermitian

if TYPE_CHECKING:
    from slate.array import Array
    from slate.basis import Basis


def _test_einsum_in_basis(
    array: Array[Any, Any, TupleBasis[Any, None, Any]],
    vector: Array[Any, Any, Any],
    basis: Basis,
) -> None:
    transformed_array = array.with_basis(
        with_child(array.basis, basis.dual_basis(), 1),
    ).ok()
    transformed_vector = vector.with_basis(basis).ok()

    np.testing.assert_allclose(
        np.einsum(  # type: ignore libary
            "ij,j->i",
            array.raw_data.reshape(array.basis.shape),
            vector.raw_data.reshape(vector.basis.size),
        ),
        np.einsum(  # type: ignore libary
            "ij,j->i",
            transformed_array.raw_data.reshape(array.basis.shape),
            transformed_vector.raw_data.reshape(vector.basis.size),
        ),
        atol=1e-15,
    )


def test_einsum() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = ArrayBuilder(
        TupleBasis(
            (
                FundamentalBasis.from_size(10),
                FundamentalBasis.from_size(10).dual_basis(),
            )
        ).upcast(),
        data,
    ).ok()

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = ArrayBuilder(array.basis.children[0], data).ok()

    _test_einsum_in_basis(array, vector, TransformedBasis(vector.basis))
    _test_einsum_in_basis(array, vector, CroppedBasis(vector.basis.size, vector.basis))
    _test_einsum_in_basis(
        array,
        vector,
        TruncatedBasis(Truncation(vector.basis.size, 1, 0), vector.basis),
    )
    _test_einsum_in_basis(
        array,
        vector,
        TruncatedBasis(Truncation(vector.basis.size, 1, 10), vector.basis),
    )


def test_einsum_diagonal() -> None:
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = ArrayBuilder(
        TupleBasis(
            (
                FundamentalBasis.from_size(10),
                FundamentalBasis.from_size(10).dual_basis(),
            )
        ).upcast(),
        data,
    ).ok()

    data = rng.random((10,)) + 1j * rng.random((10,))
    vector = ArrayBuilder(array.basis.children[0], data).ok()
    diagonal_array = into_diagonal(array)

    _test_einsum_in_basis(array, vector, diagonal_array.basis.inner[0])

    data = array.raw_data.reshape(array.basis.shape)
    data += np.conj(data.T)
    array = ArrayBuilder(
        TupleBasis(
            (
                FundamentalBasis.from_size(10),
                FundamentalBasis.from_size(10).dual_basis(),
            )
        ).upcast(),
        data,
    )
    diagonal_array = into_diagonal_hermitian(array)
    _test_einsum_in_basis(array, vector, diagonal_array.basis.inner[0])


def test_einsum_specification_parse() -> None: ...


@pytest.mark.parametrize(
    "case",
    [
        (
            "(i   j)",
            (
                EinsteinIndex(label="i", is_dual=False),
                EinsteinIndex(label="j", is_dual=False),
            ),
        ),
        (
            "(m (i j))",
            (
                EinsteinIndex(label="m", is_dual=False),
                (
                    EinsteinIndex(label="i", is_dual=False),
                    EinsteinIndex(label="j", is_dual=False),
                ),
            ),
        ),
        (
            "(m (i j) k)",
            (
                EinsteinIndex(label="m", is_dual=False),
                (
                    EinsteinIndex(label="i", is_dual=False),
                    EinsteinIndex(label="j", is_dual=False),
                ),
                EinsteinIndex(label="k", is_dual=False),
            ),
        ),
        (
            "(m (i (k l)) (k l))",
            (
                EinsteinIndex(label="m", is_dual=False),
                (
                    EinsteinIndex(label="i", is_dual=False),
                    (
                        EinsteinIndex(label="k", is_dual=False),
                        EinsteinIndex(label="l", is_dual=False),
                    ),
                ),
                (
                    EinsteinIndex(label="k", is_dual=False),
                    EinsteinIndex(label="l", is_dual=False),
                ),
            ),
        ),
        (
            "(i   j')",
            (
                EinsteinIndex(label="i", is_dual=False),
                EinsteinIndex(label="j", is_dual=True),
            ),
        ),
        ("i", EinsteinIndex(label="i", is_dual=False)),
        ("i'", EinsteinIndex(label="i", is_dual=True)),
    ],
)
def test_einsum_nested_index_parse(case: tuple[str, NestedEinsteinIndex]) -> None:
    string, expected = case
    parsed = parse_einsum_index(string)
    assert parsed == expected
