import os
from unittest import TestCase

from netCDF4 import Dataset, chartostring

import numpy as np
import numpy.ma as ma

from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.netcdf_mixin import NetCDFMixin

from .data_path import data_path


class NetcdfModel(NetCDFMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="NetcdfModel",
            model_folder=data_path()
        )

    def read(self):
        super().read()

        # just add the parameters ourselves for now (values taken from test_pi_mixin)
        params = {'k': 1.01, 'x': 1.02, 'SV_V_y': 22.02, 'j': 12.01, 'b': 13.01, 'y': 12.02, 'SV_H_y': 22.02}
        for key, value in params.items():
            self.io.set_parameter(key, value)

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("loc_a__x", self.times()[-1])
        f = xf ** 2
        return f

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        return compiler_options


class TestNetCDFMixin(TestCase):
    def setUp(self):
        self.problem = NetcdfModel()
        self.tolerance = 1e-5

    def test_read(self):
        self.problem.read()

        datastore = self.problem.io
        self.assertTrue(np.all(datastore.get_timeseries_values('loc_a__u_min') == -3.0))
        self.assertTrue(np.all(datastore.get_timeseries_values('loc_b__u_min') == -2.0))
        self.assertTrue(np.all(datastore.get_timeseries_values('loc_a__u_max') == 3.0))
        self.assertTrue(np.all(datastore.get_timeseries_values('loc_b__u_max') == 2.0))

        expected_values = np.zeros((22,), dtype=float)
        expected_values[0] = 1.02
        expected_values[2] = 0.03
        self.assertTrue(np.array_equal(datastore.get_timeseries_values('loc_a__x'), expected_values))
        self.assertTrue(np.all(np.isnan(datastore.get_timeseries_values('loc_b__x'))))

        expected_values = np.zeros((22,), dtype=float)
        expected_values[2] = 0.03
        self.assertTrue(np.array_equal(datastore.get_timeseries_values('loc_a__w'), expected_values))
        self.assertTrue(np.all(np.isnan(datastore.get_timeseries_values('loc_b__w'))))

        self.assertTrue(np.all(datastore.get_timeseries_values('loc_a__constant_input') == 1.0))
        self.assertTrue(np.all(datastore.get_timeseries_values('loc_b__constant_input') == 1.5))

    def test_write(self):
        self.problem.optimize()
        self.results = self.problem.extract_results()

        # open the exported file
        filename = os.path.join(
            data_path(),
            self.problem.timeseries_export_basename + ".nc"
        )
        dataset = Dataset(filename)

        written_variables = dataset.variables.keys()
        self.assertEqual(len(written_variables), 10)
        self.assertTrue('time' in written_variables)
        self.assertTrue('station_id' in written_variables)
        self.assertTrue('lon' in written_variables)
        self.assertTrue('lat' in written_variables)
        self.assertTrue('y' in written_variables)
        self.assertTrue('constant_output' in written_variables)
        self.assertTrue('u' in written_variables)
        self.assertTrue('z' in written_variables)
        self.assertTrue('switched' in written_variables)
        self.assertTrue('x_delayed' in written_variables)

        ids_var = dataset.variables['station_id']
        self.assertEqual(ids_var.shape, (3, 5))
        self.assertEqual(ids_var.cf_role, 'timeseries_id')
        station_ids = []
        for i in range(3):
            station_ids.append(str(chartostring(ids_var[i])))

        self.assertTrue('loc_a'in station_ids)
        self.assertTrue('loc_b' in station_ids)
        self.assertTrue('loc_c' in station_ids)

        # order of location ids is random each time the test runs...
        loc_a_index = station_ids.index('loc_a')
        loc_b_index = station_ids.index('loc_b')
        loc_c_index = station_ids.index('loc_c')

        self.assertAlmostEqual(dataset.variables['lon'][loc_a_index], 4.3780269, delta=self.tolerance)

        y = dataset.variables['y']
        self.assertEqual(y.shape, (22, 3))
        for i in range(3):
            data = ma.filled(y[:, i], np.nan)
            if i == loc_c_index:
                self.assertAlmostEqual(data[0], 1.98, delta=self.tolerance)
                for j in range(1, 22):
                    self.assertAlmostEqual(data[j], 3.0, delta=self.tolerance)
            else:
                self.assertTrue(np.all(np.isnan(data)))

        u = dataset.variables['u']
        self.assertEqual(u.shape, (22, 3))
        for i in range(3):
            data = ma.filled(u[:, i], np.nan)
            if i == loc_b_index:
                self.assertTrue(np.all(~np.isnan(data)))
            else:
                self.assertTrue(np.all(np.isnan(data)))

        constant_output = dataset.variables['constant_output']
        self.assertEqual(constant_output.shape, (22, 3))
        for i in range(3):
            data = ma.filled(constant_output[:, i], np.nan)
            if i == loc_a_index:
                self.assertTrue(np.all(data == 1.0))
            else:
                self.assertTrue(np.all(np.isnan(data)))

        time = dataset.variables['time']
        self.assertEqual(time.units, 'seconds since 2013-05-09 22:00:00')
        self.assertEqual(time.standard_name, 'time')
        self.assertEqual(time.axis, 'T')
        self.assertTrue(np.allclose(time[:], np.arange(0, 22*3600, 3600, dtype=float)))
