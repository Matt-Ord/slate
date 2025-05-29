from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from slate_core.array import Array
from slate_core.basis import (
    CoordinateBasis,
    CroppedBasis,
    FundamentalBasis,
    TransformedBasis,
    TrigonometricTransformBasis,
    TruncatedBasis,
    Truncation,
)
from slate_core.metadata import SimpleMetadata

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_core.basis._basis import Basis

BUILD_SIMPLE_BASIS: list[
    Callable[[FundamentalBasis[SimpleMetadata]], Basis[SimpleMetadata]]
] = [
    lambda b: TransformedBasis(b, direction="forward").upcast(),
    lambda b: TransformedBasis(b, direction="backward").upcast(),
    lambda b: TrigonometricTransformBasis(b, ty="cos").upcast(),
    lambda b: TrigonometricTransformBasis(b, ty="sin").upcast(),
    lambda b: CoordinateBasis((0, 1, 2), b).upcast(),
    lambda b: TruncatedBasis(Truncation(3, 7, 0), b).upcast(),
    lambda b: TruncatedBasis(Truncation(3, 7, 1), b).upcast(),
    # TODO: currently step = -1 is broken  # noqa: FIX002
    # lambda b: TruncatedBasis(Truncation(3, -1, 1), b).upcast(),  # noqa: ERA001
    lambda b: CroppedBasis(0, b).upcast(),
    lambda b: CroppedBasis(1, b).upcast(),
]


@pytest.mark.parametrize(
    "build_basis",
    BUILD_SIMPLE_BASIS,
)
def test_simple_basis_round_trip(
    build_basis: Callable[[FundamentalBasis[SimpleMetadata]], Basis[SimpleMetadata]],
) -> None:
    inner_basis = FundamentalBasis.from_size(10)
    basis = build_basis(inner_basis)
    array = Array(basis, data=np.arange(basis.size, dtype=np.complex128))

    converted = array.with_basis(inner_basis).with_basis(basis)
    np.testing.assert_array_almost_equal(array.raw_data, converted.raw_data)


@pytest.mark.parametrize(
    "build_basis",
    BUILD_SIMPLE_BASIS,
)
def test_simple_basis_equality(
    build_basis: Callable[[FundamentalBasis[SimpleMetadata]], Basis[SimpleMetadata]],
) -> None:
    basis_0 = build_basis(FundamentalBasis(SimpleMetadata(10)))
    basis_1 = build_basis(FundamentalBasis(SimpleMetadata(10)))

    assert basis_0 == basis_1
