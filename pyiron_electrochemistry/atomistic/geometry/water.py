# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_atomistics.atomistics.job.atomistic import AtomisticGenericJob
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from typing import Tuple, Union

__author__ = "Sudarsan Surendralal"
__maintainer__ = "Sudarsan Surendralal"
__email__ = "surendralal@mpie.de"
__status__ = "production"
__date__ = "Nov 1, 2021"


class WaterGeometryCalculator:
    """
    Class to analyze the geometries of water molecules in an atomistic simulation.
    """

    def __init__(self, job: AtomisticGenericJob, fixed_bonds: bool = True, water_bond_cutoff: float = 1.3):
        """Initializing the class

        Args:
            job (pyiron_atomistics.atomistics.job.atomistic.AtomisticGenericJob): The given atomistic job
            fixed_bonds (bool): True of the water bonds remain unbroken throughout the simulation
            water_bond_cutoff (float): The cutoff radius of a sphere centered at the nucleus of an oxygen atom
                                        used to determine the number of hydrogen atoms covalently bonded to it
        """
        self._job = job
        self._fixed_bonds = fixed_bonds
        self._water_bond_cutoff = water_bond_cutoff
        self._water_oxygen_indices = []
        self._water_hydrogen_indices = []
        self._oh_vec_1, self._oh_vec_2 = [], []
        self._intra_oh_distances, self._intra_oh_angles = [], []
        if fixed_bonds:
            self._compute_water_bonds()
        else:
            raise NotImplementedError("Currently this class can only analyze trajectories"
                                      " where the water bonds are intact")

    @property
    def structure(self) -> Atoms:
        """
        The initial structure of the trajectory

        Returns:
            pyiron_atomistics.atomistics.structure.atoms.Atoms
        """
        return self._job.structure

    @property
    def water_oxygen_indices(self) -> Union[np.ndarray, list]:
        """Indices of oxygen atoms that are part of water molecules."""
        return self._water_oxygen_indices

    @property
    def water_hydrogen_indices(self) -> Union[np.ndarray, list]:
        """Indices of hydrogen atoms that are part of water molecules."""
        return self._water_hydrogen_indices

    def _compute_water_bonds(self) -> None:
        neighbors = self.structure.get_neighbors(num_neighbors=5)
        oxy_indices = self.structure.select_index("O")
        hyd_indices = self.structure.select_index("H")
        oxy_neigh_indices = np.array(neighbors.indices)[oxy_indices]
        oxy_neigh_distances = np.array(neighbors.distances)[oxy_indices]
        within_cutoff_bool = oxy_neigh_distances <= self._water_bond_cutoff
        oxy_hyd_indices_list = [np.intersect1d(oxy_neigh_indices[i, bool_ind], hyd_indices)
                                for i, bool_ind in enumerate(within_cutoff_bool)]
        water_oxy_indices = list()
        water_hyd_indices = list()
        for i, oxy_hyd_ind in enumerate(oxy_hyd_indices_list):
            if len(oxy_hyd_ind) == 2:
                water_oxy_indices.append(oxy_indices[i])
                water_hyd_indices.append(oxy_hyd_ind)
        self._water_oxygen_indices = np.array(water_oxy_indices)
        self._water_hydrogen_indices = np.array(water_hyd_indices)
        self._oh_vec_1, self._oh_vec_2 = self._get_intra_oh_vec()
        self._intra_oh_distances = np.stack(np.array([np.linalg.norm(val, axis=2)
                                                      for val in [self._oh_vec_1, self._oh_vec_2]]))
        self._intra_oh_angles = get_angle_traj_vectors(self._oh_vec_1, self._oh_vec_2)

    def _get_intra_oh_vec(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = self._job.output.unwrapped_positions
        oh_vec_1 = positions[:, self._water_hydrogen_indices[:, 0], :] - positions[:, self._water_oxygen_indices, :]
        oh_vec_2 = positions[:, self._water_hydrogen_indices[:, 1], :] - positions[:, self._water_oxygen_indices, :]
        return oh_vec_1, oh_vec_2

    @property
    def intra_oh_distances(self) -> Union[list, np.ndarray]:
        """Returns list of intra-molecular OH distances."""
        return self._intra_oh_distances

    @property
    def bond_angles(self) -> Union[list, np.ndarray]:
        """Returns list of water bond angles (in radians)."""
        return self._intra_oh_angles


def get_angle_traj_vectors(vec_1: np.ndarray, vec_2: np.ndarray) -> np.ndarray:
    """
    Returns the angles between the trajectories of two vectors of the same shape

    Args:
        vec_1 (ndarray): Vector 1
        vec_2 (ndarray): Vector 2

    Returns:
        ndarray: The agnle (in radians) between the two vectors

    """
    return np.arccos(np.sum(vec_1 * vec_2, axis=-1) / (np.linalg.norm(vec_1, axis=-1) * np.linalg.norm(vec_2, axis=-1)))
