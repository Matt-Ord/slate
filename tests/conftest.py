from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array import Array

if TYPE_CHECKING:
    from slate.basis import TupleBasisLike
    from slate.metadata import BasisMetadata


@pytest.fixture
def shape() -> tuple[int, ...]:
    return (2, 3)


@pytest.fixture
def sample_data(shape: tuple[int, ...]) -> np.ndarray[Any, np.dtype[np.int64]]:
    rng = np.random.default_rng()
    return rng.integers(0, 10, shape)


@pytest.fixture
def slate_array_integer(
    sample_data: np.ndarray[Any, np.dtype[np.int64]],
) -> Array[TupleBasisLike[tuple[BasisMetadata, ...], None], np.dtype[np.int64]]:
    return Array.from_array(sample_data)


@pytest.fixture
def slate_array_complex(
    sample_data: np.ndarray[Any, np.dtype[np.int64]],
) -> Array[TupleBasisLike[tuple[BasisMetadata, ...], None], np.dtype[np.complex128]]:
    return Array.from_array(sample_data.astype(np.complex128))
