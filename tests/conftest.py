from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from slate.array.array import SlateArray

if TYPE_CHECKING:
    from slate.basis import FundamentalBasis
    from slate.metadata import SimpleMetadata


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
) -> SlateArray[np.int64, FundamentalBasis[SimpleMetadata]]:
    return SlateArray.from_array(sample_data)


@pytest.fixture
def slate_array_complex(
    sample_data: np.ndarray[Any, np.dtype[np.int64]],
) -> SlateArray[np.complex128, FundamentalBasis[SimpleMetadata]]:
    return SlateArray.from_array(sample_data.astype(np.complex128))
