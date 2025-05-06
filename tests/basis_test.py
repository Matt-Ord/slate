from __future__ import annotations

import numpy as np

from slate_core import basis
from slate_core.metadata.volume import spaced_volume_metadata_from_stacked_delta_x


def test_transformed_basis_eq() -> None:
    metadata_0 = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (160,)
    )
    metadata_1 = spaced_volume_metadata_from_stacked_delta_x(
        (np.array([2 * np.pi]),), (160,)
    )
    assert metadata_0 == metadata_1

    basis_0 = basis.from_metadata(metadata_0)
    basis_1 = basis.from_metadata(metadata_1)
    assert basis_0 == basis_1

    upcast_0 = basis_0.upcast()
    upcast_1 = basis_1.upcast()
    assert upcast_0 == upcast_1
