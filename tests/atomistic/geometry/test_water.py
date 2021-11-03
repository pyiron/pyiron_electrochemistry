# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import os
from pyiron_atomistics.atomistics.structure.atoms import Atoms
from pyiron_base.generic.hdfio import FileHDFio
from pyiron_base._tests import ToyJob, TestWithProject
from pyiron_electrochemistry.atomistic.geometry.water import WaterGeometryCalculator
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

    def test_consistency(self):
        self.assertEqual(self.water_geo.structure.get_chemical_formula(), 'H54O27')

    def test_intra_oh_distances(self):
        self.assertEqual(self.water_geo.intra_oh_distances.shape, (2, 11, 27))
        self.assertEqual(self.water_geo.intra_oh_distances.max(), 1.0571081688094743)
        self.assertEqual(self.water_geo.intra_oh_distances.min(), 0.9323863104101667)


if __name__ == '__main__':
    unittest.main()
