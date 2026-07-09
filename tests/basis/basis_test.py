from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from slate_core import basis
from slate_core.array import Array
from slate_core.basis import (
    AsUpcast,
    ContractedBasis,
    CoordinateBasis,
    CroppedBasis,
    DiagonalBasis,
    FundamentalBasis,
    IsotropicBasis,
    TransformedBasis,
    TrigonometricTransformBasis,
    TruncatedBasis,
    Truncation,
    TupleBasis,
    get_common_contraction_index,
)
from slate_core.metadata import (
    AxisDirections,
    Domain,
    EvenlySpacedLengthMetadata,
    LobattoSpacedLengthMetadata,
    SimpleMetadata,
    TupleMetadata,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from slate_core.basis._basis import Basis

BUILD_SIMPLE_BASIS: list[Callable[[SimpleMetadata], Basis[SimpleMetadata]]] = [
    lambda b: TransformedBasis(FundamentalBasis(b), direction="forward").upcast(),
    lambda b: TransformedBasis(FundamentalBasis(b), direction="backward").upcast(),
    lambda b: TrigonometricTransformBasis(FundamentalBasis(b), fn="cos").upcast(),
    lambda b: TrigonometricTransformBasis(FundamentalBasis(b), fn="sin").upcast(),
    lambda b: TrigonometricTransformBasis(
        FundamentalBasis(b), ty="type 2", fn="cos"
    ).upcast(),
    lambda b: TrigonometricTransformBasis(
        FundamentalBasis(b), ty="type 2", fn="sin"
    ).upcast(),
    lambda b: TrigonometricTransformBasis(
        FundamentalBasis(b), ty="type 3", fn="cos"
    ).upcast(),
    lambda b: TrigonometricTransformBasis(
        FundamentalBasis(b), ty="type 3", fn="sin"
    ).upcast(),
    lambda b: TrigonometricTransformBasis(
        FundamentalBasis(b), ty="type 4", fn="cos"
    ).upcast(),
    lambda b: TrigonometricTransformBasis(
        FundamentalBasis(b), ty="type 4", fn="sin"
    ).upcast(),
    lambda b: CoordinateBasis((0, 1, 2), FundamentalBasis(b)).upcast(),
    lambda b: TruncatedBasis(Truncation(3, 7, 0), FundamentalBasis(b)).upcast(),
    lambda b: TruncatedBasis(Truncation(3, 7, 1), FundamentalBasis(b)).upcast(),
    # TODO: currently step = -1 is broken  # noqa: FIX002
    # lambda b: TruncatedBasis(Truncation(3, -1, 1), b).upcast(),  # noqa: ERA001
    lambda b: CroppedBasis(0, FundamentalBasis(b)).upcast(),
    lambda b: CroppedBasis(1, FundamentalBasis(b)).upcast(),
]


@pytest.mark.parametrize(
    "build_basis",
    BUILD_SIMPLE_BASIS,
)
def test_simple_basis_round_trip(
    build_basis: Callable[[SimpleMetadata], Basis[SimpleMetadata]],
) -> None:
    inner_basis = FundamentalBasis.from_size(10)
    basis = build_basis(inner_basis.metadata())
    array = Array(basis, data=np.arange(basis.size, dtype=np.complex128))

    converted = array.with_basis(inner_basis).with_basis(basis)
    np.testing.assert_array_almost_equal(array.raw_data, converted.raw_data)


@pytest.mark.parametrize(
    "build_basis",
    BUILD_SIMPLE_BASIS,
)
def test_simple_basis_equality(
    build_basis: Callable[[SimpleMetadata], Basis[SimpleMetadata]],
) -> None:
    basis_0 = build_basis(SimpleMetadata(10))
    basis_1 = build_basis(SimpleMetadata(10))

    assert basis_0 == basis_1

    basis_2 = build_basis(SimpleMetadata(11))
    assert basis_0 != basis_2

    basis_3 = TransformedBasis(build_basis(SimpleMetadata(10)))
    assert basis_0 != basis_3


def _scatter_basis() -> Basis:
    state_metadata = TupleMetadata(
        (
            EvenlySpacedLengthMetadata(5, domain=Domain(delta=2.0)),
            EvenlySpacedLengthMetadata(7, domain=Domain(delta=2.0)),
            LobattoSpacedLengthMetadata(3, domain=Domain(delta=2.0)),
        ),
        AxisDirections(
            vectors=(np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]))
        ),
    )
    fundamental_state_basis = AsUpcast(
        basis.from_metadata(state_metadata),
        state_metadata,
    )

    return DiagonalBasis(
        TupleBasis((fundamental_state_basis, fundamental_state_basis), None),
    ).upcast()


BUILD_GENERIC_BASIS: list[Callable[[], Basis]] = [
    lambda: basis.from_shape(((2, 3), (2, 3))),
    lambda: basis.from_shape((5, 7)).upcast(),
    _scatter_basis,
    lambda: TransformedBasis(basis.from_shape((5, 7)), direction="forward"),
    lambda: DiagonalBasis(basis.from_shape((5, 5))),
    lambda: ContractedBasis(basis.from_shape((5, 5)), (0, 0)),
    lambda: ContractedBasis(basis.from_shape((5, 7)), (0, 1)),
    lambda: ContractedBasis(basis.from_shape((5, 7)), (1, 0)),
    lambda: ContractedBasis(basis.from_shape((5, 5)), (1, 1)),
    lambda: ContractedBasis(basis.from_shape((5, (5, 7))), (1, (1, 0))),
    lambda: ContractedBasis(basis.from_shape((5, 5, 7)), (1, 1, 0)),
    lambda: ContractedBasis(
        basis.from_shape(((5, 7, 3), (5, 7, 3))), ((0, 1, 2), (0, 1, 3))
    ),
    lambda: IsotropicBasis(basis.from_shape((3, 3))),
]


@pytest.mark.parametrize(
    "build_basis",
    BUILD_GENERIC_BASIS,
)
def test_general_basis_round_trip(
    build_basis: Callable[[], Basis],
) -> None:
    outer_basis = build_basis()
    inner_basis = basis.from_metadata(outer_basis.metadata())
    array = Array(outer_basis, data=np.arange(outer_basis.size, dtype=np.complex128))

    converted = array.with_basis(inner_basis).with_basis(outer_basis)
    np.testing.assert_array_almost_equal(array.raw_data, converted.raw_data)


@pytest.mark.parametrize(
    "build_basis",
    BUILD_GENERIC_BASIS,
)
def test_general_basis_equality(
    build_basis: Callable[[], Basis],
) -> None:
    basis_0 = build_basis()
    basis_1 = build_basis()

    assert basis_0 == basis_1


def test_common_contraction() -> None:
    index = get_common_contraction_index(
        basis.from_shape(((1, 2, 3), (1, 2, 3), 6)),
        ((0, 1, 2), (0, 1, 2), 3),
        ((0, 1, 2), (0, 1, 2), 3),
    )
    assert index == ((0, 1, 2), (0, 1, 2), 3)

    index = get_common_contraction_index(
        basis.from_shape(((1, 2, 3), (1, 2, 3), 6)),
        ((0, 1, 2), 3, 3),
        ((0, 1, 2), (0, 1, 2), 3),
    )
    assert index == ((0, 1, 2), (4, 5, 6), 3)

    index = get_common_contraction_index(
        basis.from_shape(((1, 2, 3), (1, 2, 3), 6)),
        (0, 0, 0),
        ((0, 1, 2), 3, 3),
    )
    assert index == ((4, 5, 6), 0, 0)

    index = get_common_contraction_index(
        basis.from_shape(((1, 2, 3), (1, 2, 3), 6)),
        0,
        1,
    )
    assert index == 0
