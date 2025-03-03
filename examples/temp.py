from __future__ import annotations

import numpy as np

from slate import array
from slate.array import build
from slate.basis._diagonal import DiagonalBasis
from slate.basis._util import from_shape

rng = np.random.default_rng()
basis = DiagonalBasis(from_shape((3, 3))).upcast()
data = rng.random(basis.size).astype(np.complex128)
arr = build(basis, data).ok()

transposed = array.transpose(arr)
np.testing.assert_allclose(transposed.as_array(), arr.as_array().transpose())
np.testing.assert_allclose(
    arr.as_array(), array.transpose(array.transpose(arr)).as_array()
)
