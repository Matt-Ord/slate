from __future__ import annotations

import numpy as np

from slate import array, basis
from slate.array import build
from slate.array._conversion import cast_basis
from slate.basis import (
    DiagonalBasis,
    TransformedBasis,
    TruncatedBasis,
    Truncation,
    from_shape,
)
from slate.basis._diagonal import DiagonalBasis
from slate.basis._fundamental import FundamentalBasis
from slate.basis._tuple import TupleBasis
from slate.linalg._eig import into_diagonal

rng = np.random.default_rng()
data_basis = DiagonalBasis(basis.from_shape((3, 3))).upcast().downcast_metadata()
data = rng.random(data_basis.size).astype(np.complex128)
arr = array.build(data_basis, data).ok()

transposed = array.transpose(arr)
np.testing.assert_allclose(transposed.as_array(), arr.as_array().transpose())
np.testing.assert_allclose(
    arr.as_array(), array.transpose(array.transpose(arr)).as_array()
)


half_basis = from_shape((105,))
full_basis = TupleBasis((half_basis, half_basis)).upcast()
spaced_basis = TruncatedBasis(
    Truncation(3, 5, 0), TransformedBasis(half_basis).upcast()
).upcast()

rng = np.random.default_rng()
data = rng.random((10, 10)) + 1j * rng.random((10, 10))
test_array = build(
    TupleBasis(
        (
            FundamentalBasis.from_size(10),
            FundamentalBasis.from_size(10).dual_basis(),
        )
    ).upcast(),
    data,
).ok()

data = rng.random((10,)) + 1j * rng.random((10,))
vector = build(test_array.basis.children[0], data).ok()
downcast_array = cast_basis(test_array, test_array.basis.downcast_metadata()).ok()
diagonal_array = into_diagonal(downcast_array)
