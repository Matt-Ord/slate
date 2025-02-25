from __future__ import annotations

import numpy as np

from slate.array import Array
from slate.linalg import get_eigenvalues, into_diagonal

if __name__ == "__main__":
    rng = np.random.default_rng()
    data = rng.random((10, 10)) + 1j * rng.random((10, 10))
    array = Array.from_array(data)

    diagonal = into_diagonal(array)
    eigenvalues = get_eigenvalues(array)
    a = diagonal.raw_data

    # Diagonal is a matrix with the eigenvalues along the diagonal
    np.testing.assert_allclose(diagonal.raw_data, eigenvalues.as_array())
    # Fundamentally, diagonal and array represent the same matrix
    np.testing.assert_allclose(diagonal.as_array(), array.as_array())
