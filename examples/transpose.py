from __future__ import annotations

import numpy as np

from slate import array, basis
from slate.basis import (
    DiagonalBasis,
)

if __name__ == "__main__":
    rng = np.random.default_rng()
    data_basis = DiagonalBasis(basis.from_shape((3, 3))).upcast().downcast_metadata()
    data = rng.random(data_basis.size).astype(np.complex128)
    arr = array.build(data_basis, data).ok()

    transposed = array.transpose(arr)
    np.testing.assert_allclose(transposed.as_array(), arr.as_array().transpose())
