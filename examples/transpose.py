from __future__ import annotations

import numpy as np

from slate_core import Array, array, basis
from slate_core.basis import (
    DiagonalBasis,
)

if __name__ == "__main__":
    rng = np.random.default_rng()
    data_basis = DiagonalBasis(basis.from_shape((3, 3))).resolve_ctype().upcast()
    data = rng.random(data_basis.size).astype(np.complex128)
    arr = Array(data_basis, data)

    transposed = array.transpose(arr)
    np.testing.assert_allclose(transposed.as_array(), arr.as_array().transpose())
