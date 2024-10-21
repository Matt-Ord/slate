from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from slate.array.array import SlateArray
from slate.linalg._eig import eig, eigvals

if TYPE_CHECKING:
    from slate.basis._basis import FundamentalBasis
    from slate.metadata._metadata import SimpleMetadata
    from slate.metadata.stacked import StackedMetadata


@pytest.fixture
def slate_array_stacked() -> (
    SlateArray[np.complex128, FundamentalBasis[StackedMetadata[SimpleMetadata, None]]]
):
    rng = np.random.default_rng()
    shape = (10, 10)
    data = rng.random(shape) + 1j * rng.random(shape)
    return SlateArray.from_array(data)


def test_linalg_eig(
    slate_array_stacked: SlateArray[
        np.complex128, FundamentalBasis[StackedMetadata[SimpleMetadata, None]]
    ],
) -> None:
    diagonal = eig(slate_array_stacked)

    eigenvalues = eigvals(slate_array_stacked)
    np.testing.assert_allclose(eigenvalues.as_array(), diagonal.raw_data)

    np.testing.assert_allclose(diagonal.as_array(), slate_array_stacked.as_array())
