# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import os
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_base.generic.hdfio import FileHDFio
from pyiron_base._tests import ToyJob, TestWithProject
from pyiron_electrochemistry.atomistic.geometry.water import WaterGeometryCalculator, get_angle_traj_vectors
import unittest


class WaterToyJob(ToyJob):
    def __init__(self, project, job_name):
        super(WaterToyJob, self).__init__(project, job_name)
        filename = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../static/water_bulk_tip3p_traj",
        )
        abs_filename = os.path.abspath(filename)
        self._hdf_obj = FileHDFio(abs_filename)
        self._structure = Atoms().from_hdf(self._hdf_obj["input"])

    @property
    def structure(self):
        return self._structure

    # This function is executed
    def run_static(self):
        self.status.running = True
        self.output.unwrapped_positions = self._hdf_obj["output/generic/unwrapped_positions"]
        self.status.finished = True
        self.to_hdf()


class TestWaterGeometry(TestWithProject):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        job = cls.project.create_job(job_type=WaterToyJob, job_name="water_bulk")
        job.run()
        cls.water_geo = WaterGeometryCalculator(job)
        struct = cls.water_geo.structure.copy()
        cls.oh_vec_1 = list()
        cls.oh_vec_2 = list()
        cls.oh_angles = list()

        for pos in job.output.unwrapped_positions:
            oh_vec_1 = list()
            oh_vec_2 = list()
            oh_angles = list()
            struct.positions = pos
            for i, oxy_ind in enumerate(cls.water_geo.water_oxygen_indices):
                vec_1 = struct.get_distance(oxy_ind, cls.water_geo.water_hydrogen_indices[i, 0], vector=True)
                vec_2 = struct.get_distance(oxy_ind, cls.water_geo.water_hydrogen_indices[i, 1], vector=True)
                oh_vec_1.append(vec_1)
                oh_vec_2.append(vec_2)
                oh_angles.append(struct.get_angle(cls.water_geo.water_hydrogen_indices[i, 0], oxy_ind,
                                                  cls.water_geo.water_hydrogen_indices[i, 1]))

            cls.oh_vec_1.append(oh_vec_1)
            cls.oh_vec_2.append(oh_vec_2)
            cls.oh_angles.append(oh_angles)

    def test_consistency(self):
        self.assertEqual(self.water_geo.structure.get_chemical_formula(), 'H54O27')
        self.assertEqual(len(self.water_geo.water_oxygen_indices), 27)
        self.assertEqual(len(self.water_geo.water_hydrogen_indices[:, 0]), 27)
        self.assertEqual(len(self.water_geo.water_hydrogen_indices[:, 1]), 27)
        self.assertEqual(len(np.intersect1d(self.water_geo.structure.select_index("H"),
                                            self.water_geo.water_hydrogen_indices[:, 0])), 27)
        self.assertEqual(len(np.intersect1d(self.water_geo.structure.select_index("H"),
                                            self.water_geo.water_hydrogen_indices[:, 1])), 27)
        self.assertEqual(np.intersect1d(self.water_geo.water_hydrogen_indices[:, 1],
                                        self.water_geo.water_hydrogen_indices[:, 0]).tolist(), [])

    def test_get_intra_oh_vec(self):
        oh_vec_1, oh_vec_2 = self.water_geo._get_intra_oh_vec()
        self.assertTrue(np.allclose(oh_vec_1, np.array(self.oh_vec_1)))
        self.assertTrue(np.allclose(oh_vec_2, np.array(self.oh_vec_2)))

    def test_intra_oh_distances(self):
        self.assertEqual(self.water_geo.intra_oh_distances.shape, (2, 11, 27))
        self.assertEqual(self.water_geo.intra_oh_distances.max(), 1.0571081688094743)
        self.assertEqual(self.water_geo.intra_oh_distances.min(), 0.9323863104101667)

    def test_intra_oh_angles(self):
        self.assertEqual(self.water_geo.bond_angles.shape, (11, 27))
        self.assertTrue(np.allclose(self.water_geo.bond_angles, np.array(self.oh_angles) * np.pi / 180))

    def test_get_angles_traj_vectors(self):
        self.assertTrue(np.allclose(np.array(self.oh_angles) * np.pi / 180,
                                    get_angle_traj_vectors(np.array(self.oh_vec_1), np.array(self.oh_vec_2))))


if __name__ == '__main__':
    unittest.main()
