from __future__ import annotations

import unittest

import numpy as np

from surface_potential_analysis.basis_config.build import (
    build_momentum_basis_config_from_resolution,
    build_position_basis_config_from_resolution,
)
from surface_potential_analysis.basis_config.conversion import (
    basis_config_as_fundamental_momentum_basis_config,
    basis_config_as_fundamental_position_basis_config,
    get_rotated_basis_config,
)
from surface_potential_analysis.basis_config.util import (
    BasisConfigUtil,
    _wrap_distance,
    calculate_cumulative_x_distances_along_path,
)
from surface_potential_analysis.util.util import slice_along_axis

rng = np.random.default_rng()


class TestBasisConfig(unittest.TestCase):
    def test_surface_volume_100(self) -> None:
        points = rng.random(3)
        basis = build_position_basis_config_from_resolution(
            (1, 1, 1),
            (
                np.array([points[0], 0, 0]),
                np.array([0, points[1], 0]),
                np.array([0, 0, points[2]]),
            ),
        )
        util = BasisConfigUtil(basis)

        np.testing.assert_almost_equal(util.volume, np.prod(points))
        np.testing.assert_almost_equal(
            util.reciprocal_volume, (2 * np.pi) ** 3 / np.prod(points)
        )

    def test_inverse_lattuice_points_100(self) -> None:
        delta_x = (
            np.array([1, 0, 0]),
            np.array([0, 2, 0]),
            np.array([0, 0, 1]),
        )
        basis = build_position_basis_config_from_resolution((1, 1, 1), delta_x)
        util = BasisConfigUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util.delta_x0)
        np.testing.assert_array_equal(delta_x[1], util.delta_x1)
        np.testing.assert_array_equal(delta_x[2], util.delta_x2)

        self.assertEqual(util.dk0[0], 2 * np.pi)
        self.assertEqual(util.dk0[1], 0)
        self.assertEqual(util.dk1[0], 0)
        self.assertEqual(util.dk1[1], np.pi)

    def test_inverse_lattuice_points_111(self) -> None:
        delta_x = (
            np.array([1, 0, 0]),
            np.array([0.5, np.sqrt(3) / 2, 0]),
            np.array([0, 0, 1]),
        )
        basis = build_position_basis_config_from_resolution((1, 1, 1), delta_x)
        util = BasisConfigUtil(basis)

        np.testing.assert_array_equal(delta_x[0], util.delta_x0)
        np.testing.assert_array_equal(delta_x[1], util.delta_x1)
        np.testing.assert_array_equal(delta_x[2], util.delta_x2)

        self.assertEqual(util.dk0[0], 2 * np.pi)
        self.assertEqual(util.dk0[1], -2 * np.pi / np.sqrt(3))
        self.assertEqual(util.dk1[0], 0)
        self.assertEqual(util.dk1[1], 4 * np.pi / np.sqrt(3))

    def test_reciprocal_lattuice(self) -> None:
        delta_x = (
            rng.random(3),
            rng.random(3),
            rng.random(3),
        )
        basis = build_position_basis_config_from_resolution((1, 1, 1), delta_x)
        util = BasisConfigUtil(basis)

        np.testing.assert_array_almost_equal(delta_x[0], util.delta_x0)
        np.testing.assert_array_almost_equal(delta_x[1], util.delta_x1)
        np.testing.assert_array_almost_equal(delta_x[2], util.delta_x2)

        reciprocal = basis_config_as_fundamental_momentum_basis_config(basis)
        reciprocal_util = BasisConfigUtil(reciprocal)

        np.testing.assert_array_almost_equal(reciprocal_util.delta_x0, util.delta_x0)
        np.testing.assert_array_almost_equal(reciprocal_util.delta_x1, util.delta_x1)
        np.testing.assert_array_almost_equal(reciprocal_util.delta_x2, util.delta_x2)

        np.testing.assert_array_almost_equal(reciprocal_util.dk0, util.dk0)
        np.testing.assert_array_almost_equal(reciprocal_util.dk1, util.dk1)
        np.testing.assert_array_almost_equal(reciprocal_util.dk2, util.dk2)

        np.testing.assert_array_almost_equal(reciprocal_util.volume, util.volume)
        np.testing.assert_array_almost_equal(
            reciprocal_util.reciprocal_volume, util.reciprocal_volume
        )

        reciprocal_2 = basis_config_as_fundamental_position_basis_config(reciprocal)
        reciprocal_2_util = BasisConfigUtil(reciprocal_2)

        np.testing.assert_array_almost_equal(reciprocal_2_util.delta_x0, util.delta_x0)
        np.testing.assert_array_almost_equal(reciprocal_2_util.delta_x1, util.delta_x1)
        np.testing.assert_array_almost_equal(reciprocal_2_util.delta_x2, util.delta_x2)

        np.testing.assert_array_almost_equal(reciprocal_2_util.dk0, util.dk0)
        np.testing.assert_array_almost_equal(reciprocal_2_util.dk1, util.dk1)
        np.testing.assert_array_almost_equal(reciprocal_2_util.dk2, util.dk2)

    def test_get_stacked_index(self) -> None:
        delta_x = (
            np.array([1, 0, 0]),
            np.array([0, 2, 0]),
            np.array([0, 0, 1]),
        )
        resolution = (
            rng.integers(1, 10),
            rng.integers(1, 10),
            rng.integers(1, 10),
        )
        basis = build_position_basis_config_from_resolution(resolution, delta_x)
        util = BasisConfigUtil(basis)
        for i in range(np.prod(resolution)):
            self.assertEqual(i, util.get_flat_index(util.get_stacked_index(i)))

    def test_rotated_basis_111(self) -> None:
        delta_x = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = build_position_basis_config_from_resolution((1, 1, 1), delta_x)

        rotated_0 = get_rotated_basis_config(basis, 0, delta_x[0])  # type: ignore[arg-type,var-annotated]
        np.testing.assert_array_almost_equal(rotated_0[0].delta_x, [1, 0, 0])
        np.testing.assert_array_almost_equal(rotated_0[1].delta_x, [0, 1, 0])
        np.testing.assert_array_almost_equal(rotated_0[2].delta_x, [0, 0, 1])

        rotated_1 = get_rotated_basis_config(basis, 0, delta_x[1])  # type: ignore[arg-type,var-annotated]
        np.testing.assert_array_almost_equal(rotated_1[0].delta_x, [0, 1, 0])
        np.testing.assert_array_almost_equal(rotated_1[1].delta_x, [-1, 0, 0])
        np.testing.assert_array_almost_equal(rotated_1[2].delta_x, [0, 0, 1])

        rotated_2 = get_rotated_basis_config(basis, 0, delta_x[2])  # type: ignore[arg-type,var-annotated]
        np.testing.assert_array_almost_equal(rotated_2[0].delta_x, [0, 0, 1])
        np.testing.assert_array_almost_equal(rotated_2[1].delta_x, [0, 1, 0])
        np.testing.assert_array_almost_equal(rotated_2[2].delta_x, [-1, 0, 0])

    def test_rotated_basis(self) -> None:
        delta_x = (
            rng.random(3),
            rng.random(3),
            rng.random(3),
        )
        basis = build_position_basis_config_from_resolution((1, 1, 1), delta_x)

        for i in (0, 1, 2):
            rotated = get_rotated_basis_config(basis, i)  # type: ignore[arg-type,var-annotated]
            np.testing.assert_array_almost_equal(
                rotated[i].delta_x, [0, 0, np.linalg.norm(delta_x[i])]
            )
            for j in (0, 1, 2):
                np.testing.assert_almost_equal(
                    np.linalg.norm(rotated[j].delta_x),
                    np.linalg.norm(basis[j].delta_x),
                )

            direction = rng.random(3)
            rotated = get_rotated_basis_config(basis, i, direction)  # type: ignore[arg-type]
            np.testing.assert_almost_equal(
                np.dot(rotated[i].delta_x, direction),
                np.linalg.norm(direction) * np.linalg.norm(rotated[i].delta_x),
            )
            for j in (0, 1, 2):
                np.testing.assert_almost_equal(
                    np.linalg.norm(rotated[j].delta_x),
                    np.linalg.norm(basis[j].delta_x),
                )

    def test_nx_points_simple(self) -> None:
        delta_x = (
            rng.random(3),
            rng.random(3),
            rng.random(3),
        )
        basis = build_position_basis_config_from_resolution((2, 2, 2), delta_x)
        util = BasisConfigUtil(basis)

        actual = util.fundamental_nx_points
        expected = [
            [0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1, 0, 1],
        ]
        np.testing.assert_array_equal(expected, actual)

        resolution = (
            rng.integers(1, 20),
            rng.integers(1, 20),
            rng.integers(1, 20),
        )
        basis = build_position_basis_config_from_resolution(resolution, delta_x)
        util = BasisConfigUtil(basis)
        actual = util.fundamental_nx_points

        for axis in range(3):
            basis_for_axis = actual[axis].reshape(*resolution)
            for j in range(resolution[axis]):
                slice_j = basis_for_axis[slice_along_axis(j, axis)]
                np.testing.assert_equal(slice_j, j)

    def test_nk_points_simple(self) -> None:
        delta_x = (
            rng.random(3),
            rng.random(3),
            rng.random(3),
        )
        basis = build_position_basis_config_from_resolution((2, 2, 2), delta_x)
        util = BasisConfigUtil(basis)

        actual = util.fundamental_nk_points
        expected = [
            [0, 0, 0, 0, -1, -1, -1, -1],
            [0, 0, -1, -1, 0, 0, -1, -1],
            [0, -1, 0, -1, 0, -1, 0, -1],
        ]
        np.testing.assert_array_equal(expected, actual)

        resolution = (
            rng.integers(1, 20),
            rng.integers(1, 20),
            rng.integers(1, 20),
        )
        basis = build_position_basis_config_from_resolution(resolution, delta_x)
        util = BasisConfigUtil(basis)
        actual = util.fundamental_nk_points

        for axis in range(3):
            basis_for_axis = actual[axis].reshape(*resolution)
            expected_for_axis = np.fft.fftfreq(resolution[axis], 1 / resolution[axis])

            for j, expected in enumerate(expected_for_axis):
                slice_j = basis_for_axis[slice_along_axis(j, axis)]
                np.testing.assert_equal(slice_j, expected)

    def test_x_points_100(self) -> None:
        delta_x = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = build_position_basis_config_from_resolution((3, 3, 3), delta_x)
        util = BasisConfigUtil(basis)

        actual = util.fundamental_x_points
        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / 3
        expected_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]) / 3
        expected_z = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0]) / 3
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        delta_x = (
            np.array([0, 1, 0], dtype=float),
            np.array([3, 0, 0], dtype=float),
            np.array([0, 0, 5], dtype=float),
        )
        basis = build_position_basis_config_from_resolution((3, 3, 3), delta_x)
        util = BasisConfigUtil(basis)
        actual = util.fundamental_x_points

        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0, 6.0, 6.0, 6.0]) / 3
        expected_y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]) / 3
        expected_z = np.array([0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0, 0.0, 5.0, 10.0]) / 3
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

    def test_k_points_100(self) -> None:
        dx = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = build_momentum_basis_config_from_resolution((3, 3, 3), dx)
        util = BasisConfigUtil(basis)

        actual = util.fundamental_nk_points
        # fmt: off
        expected_x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        expected_y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
        expected_z = np.array([0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0])
        # fmt: on

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        delta_x = (
            np.array([0, 1, 0], dtype=float),
            np.array([3, 0, 0], dtype=float),
            np.array([0, 0, 5], dtype=float),
        )
        basis = build_momentum_basis_config_from_resolution((3, 3, 3), delta_x)
        util = BasisConfigUtil(basis)
        actual = util.fundamental_nk_points

        np.testing.assert_array_equal(expected_x, actual[0])
        np.testing.assert_array_equal(expected_y, actual[1])
        np.testing.assert_array_equal(expected_z, actual[2])

        actual_k = util.fundamental_k_points
        expected_kx = 2 * np.pi * expected_y / 3.0
        expected_ky = 2 * np.pi * expected_x / 1.0
        expected_kz = 2 * np.pi * expected_z / 5.0

        np.testing.assert_array_almost_equal(expected_kx, actual_k[0])
        np.testing.assert_array_almost_equal(expected_ky, actual_k[1])
        np.testing.assert_array_almost_equal(expected_kz, actual_k[2])

    def test_wrap_distance(self) -> None:
        expected = [0, 1, -1, 0, 1, -1, 0]
        distances = np.array([-3, -2, -1, 0, 1, 2, 3])
        for e, d in zip(expected, distances, strict=True):
            np.testing.assert_equal(_wrap_distance(d, 3), e, f"d={d}, l=3")
        np.testing.assert_array_equal(_wrap_distance(distances, 3), expected)

        expected = [0, 1, -2, -1, 0, 1, -2, -1, 0]
        distances = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4])
        for e, d in zip(expected, distances, strict=True):
            np.testing.assert_equal(_wrap_distance(d, 4), e, f"d={d}, l=4")
        np.testing.assert_array_equal(_wrap_distance(distances, 4), expected)

    def test_calculate_cumulative_distances_along_path(self) -> None:
        dx = (
            np.array([1, 0, 0], dtype=float),
            np.array([0, 1, 0], dtype=float),
            np.array([0, 0, 1], dtype=float),
        )
        basis = build_position_basis_config_from_resolution((3, 3, 3), dx)

        distances = calculate_cumulative_x_distances_along_path(
            basis, np.array([[0, 1], [0, 0], [0, 0]])
        )
        np.testing.assert_array_equal(distances, [0, 1])

        distances = calculate_cumulative_x_distances_along_path(
            basis, np.array([[0, 2], [0, 0], [0, 0]])
        )
        np.testing.assert_array_equal(distances, [0, 2])

        distances = calculate_cumulative_x_distances_along_path(
            basis, np.array([[0, 2], [0, 0], [0, 0]]), wrap_distances=True
        )
        np.testing.assert_array_equal(distances, [0, 1])
