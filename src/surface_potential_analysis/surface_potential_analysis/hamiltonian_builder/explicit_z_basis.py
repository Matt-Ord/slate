from functools import cache, cached_property
from typing import Generic, Literal, TypeVar

import hamiltonian_generator
import numpy as np
from scipy.constants import hbar

from surface_potential_analysis.basis import (
    BasisUtil,
    ExplicitBasis,
    MomentumBasis,
    PositionBasis,
    TruncatedBasis,
)
from surface_potential_analysis.basis_config import BasisConfig, BasisConfigUtil
from surface_potential_analysis.hamiltonian import HamiltonianWithBasis
from surface_potential_analysis.potential import Potential

_L0 = TypeVar("_L0", bound=int)
_L1 = TypeVar("_L1", bound=int)
_L2 = TypeVar("_L2", bound=int)
_L3 = TypeVar("_L3", bound=int)
_L4 = TypeVar("_L4", bound=int)
_L5 = TypeVar("_L5", bound=int)


class _SurfaceHamiltonianUtil(Generic[_L0, _L1, _L2, _L3, _L4, _L5]):
    _potential: Potential[_L0, _L1, _L2]

    _basis: BasisConfig[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, MomentumBasis[_L2]],
    ]

    def __init__(
        self,
        potential: Potential[_L0, _L1, _L2],
        basis: BasisConfig[
            TruncatedBasis[_L3, MomentumBasis[_L0]],
            TruncatedBasis[_L4, MomentumBasis[_L1]],
            ExplicitBasis[_L5, MomentumBasis[_L2]],
        ],
    ) -> None:
        self._potential = potential
        self._basis = basis
        # self._potential_offset = potential_offset
        if 2 * (self._resolution[0] - 1) > self._potential["basis"][0]["n"]:
            print(self._resolution[0], self._potential["basis"][0]["n"])
            raise AssertionError(
                "Potential does not have enough resolution in x direction"
            )

        if 2 * (self._resolution[1] - 1) > self._potential["basis"][1]["n"]:
            print(self._resolution[1], self._potential["basis"][1]["n"])
            raise AssertionError(
                "Potential does not have enough resolution in y direction"
            )

    @property
    def points(self) -> np.ndarray[tuple[_L0, _L1, _L2], np.dtype[np.float_]]:
        return self._potential["points"]

    @property
    def Nx(self) -> int:
        return self.points.shape[0]  # type:ignore

    @property
    def Ny(self) -> int:
        return self.points.shape[1]  # type:ignore

    @property
    def Nz(self) -> int:
        return self.points.shape[2]  # type:ignore

    @property
    def dz(self) -> float:
        util = BasisUtil(self._potential["basis"][2])
        return np.linalg.norm(util.fundamental_dx)  # type:ignore

    def hamiltonian(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> HamiltonianWithBasis[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, MomentumBasis[_L2]],
    ]:
        diagonal_energies = np.diag(self._calculate_diagonal_energy(bloch_phase))
        other_energies = self._calculate_off_diagonal_energies_fast()

        energies = diagonal_energies + other_energies

        if False and not np.allclose(energies, energies.conjugate().T):
            raise AssertionError("Hamiltonian is not hermitian")

        return {"array": energies, "basis": self._basis}

    @cached_property
    def eigenstate_indexes(
        self,
    ) -> np.ndarray[tuple[Literal[3], int], np.dtype[np.int_]]:
        util = BasisConfigUtil(self._basis)

        x0t, x1t, zt = np.meshgrid(
            util.x0_basis.nk_points,  # type: ignore
            util.x1_basis.nk_points,  # type: ignore
            util.x2_basis.nk_points,  # type: ignore
            indexing="ij",
        )
        return np.array([x0t.ravel(), x1t.ravel(), zt.ravel()])  # type: ignore

    def _calculate_diagonal_energy(
        self, bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        kx0_coords, kx1_coords, nkz_coords = self.eigenstate_indexes

        util = BasisConfigUtil(self.basis)

        dkx0 = util.dk0
        dkx1 = util.dk1
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]

        kx_points = dkx0[0] * kx0_coords + dkx1[0] * kx1_coords + bloch_phase[0]
        x_energy = (hbar * kx_points) ** 2 / (2 * mass)
        ky_points = dkx0[1] * kx0_coords + dkx1[1] * kx1_coords + bloch_phase[1]
        y_energy = (hbar * ky_points) ** 2 / (2 * mass)
        z_energy = (hbar * sho_omega) * (nkz_coords + 0.5)
        return x_energy + y_energy + z_energy  # type: ignore

    @cache
    def get_sho_potential(self) -> np.ndarray[tuple[int], np.dtype[np.float_]]:
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]
        return 0.5 * mass * sho_omega**2 * np.square(self.z_distances)  # type: ignore

    @cache
    def get_sho_subtracted_points(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.float_]]:
        return np.subtract(self.points, self.get_sho_potential())  # type: ignore

    @cache
    def get_ft_potential(
        self,
    ) -> np.ndarray[tuple[int, int, int], np.dtype[np.complex128]]:
        subtracted_potential = self.get_sho_subtracted_points()
        return np.fft.ifft2(subtracted_potential, axes=(0, 1))  # type: ignore

    @cache
    def _calculate_off_diagonal_energies_fast(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
        mass = self._config["mass"]
        sho_omega = self._config["sho_omega"]
        return np.array(  # type: ignore
            hamiltonian_generator.calculate_off_diagonal_energies(
                self.get_ft_potential().tolist(),
                self._resolution,
                self.dz,
                mass,
                sho_omega,
                self.z_offset,
            )
        )

    @cache
    def _calculate_off_diagonal_entry(
        self, nz1: int, nz2: int, ndkx0: int, ndkx1: int
    ) -> float:
        """Calculates the off diagonal energy using the 'folded' points ndkx, ndky"""
        ft_pot_points = self.get_ft_potential()[ndkx0, ndkx1]
        hermite1 = self.basis[2]["vectors"][nz1]
        hermite2 = self.basis[2]["vectors"][nz2]

        fourier_transform = float(np.sum(hermite1 * hermite2 * ft_pot_points))

        return self.dz * fourier_transform

    def _calculate_off_diagonal_energies(
        self,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.complex_]]:
        n_coordinates = len(self.eigenstate_indexes.T)
        hamiltonian = np.zeros(shape=(n_coordinates, n_coordinates))

        for index1, [nkx0_0, nkx1_0, nz1] in enumerate(self.eigenstate_indexes.T):
            for index2, [nkx0_1, nkx1_1, nz2] in enumerate(self.eigenstate_indexes.T):
                # Number of jumps in units of dkx for this matrix element

                # As k-> k+ Nx * dkx the ft potential is left unchanged
                # Therefore we 'wrap round' ndkx into the region 0<ndkx<Nx
                # Where Nx is the number of x points

                # In reality we want to make sure ndkx < Nx (ie we should
                # make sure we generate enough points in the interpolation)
                ndkx0 = (nkx0_1 - nkx0_0) % self.Nx
                ndkx1 = (nkx1_1 - nkx1_0) % self.Ny

                hamiltonian[index1, index2] = self._calculate_off_diagonal_entry(
                    nz1, nz2, ndkx0, ndkx1
                )

        return hamiltonian  # type: ignore


def total_surface_hamiltonian(
    potential: Potential[_L0, _L1, _L2],
    bloch_phase: np.ndarray[tuple[Literal[3]], np.dtype[np.float_]],
    basis: BasisConfig[
        TruncatedBasis[_L3, MomentumBasis[_L0]],
        TruncatedBasis[_L4, MomentumBasis[_L1]],
        ExplicitBasis[_L5, MomentumBasis[_L2]],
    ],
) -> HamiltonianWithBasis[
    TruncatedBasis[_L3, MomentumBasis[_L0]],
    TruncatedBasis[_L4, MomentumBasis[_L1]],
    ExplicitBasis[_L5, MomentumBasis[_L2]],
]:
    util = _SurfaceHamiltonianUtil(potential, basis)
    return util.hamiltonian(bloch_phase)