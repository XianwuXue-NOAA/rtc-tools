import logging
from unittest import TestCase

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem
)
from rtctools.optimization.io_mixin import IOMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

from .data_path import data_path

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class DummyIOMixin(IOMixin):
    def read(self):
        # fill with dummy data
        times = np.array([-7200, -3600, 0, 3600, 7200, 9800])
        self.io.set_times(times)
        forecast_index = 2
        self.io.set_forecast_index(forecast_index)

        values = {
            'constant_input': [1.1, 1.4, 0.9, 1.2, 1.5, 1.7],
            'u_Min': [0.5, 0.2, 0.3, 0.1, 0.4, 0.0],
            'u_Max': [2.1, 2.2, 2.0, 2.4, 2.5, 2.3],
            'alias': [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]  # alias of 'x'
        }

        for key, value in values.items():
            self.io.set_timeseries_values(key, np.array(value))

    def write(self):
        pass


class Model(DummyIOMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self, **kwargs):
        kwargs["model_name"] = kwargs.get("model_name", "Model")
        kwargs["input_folder"] = data_path()
        kwargs["output_folder"] = data_path()
        kwargs["model_folder"] = data_path()
        super().__init__(**kwargs)

    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times()[-1])
        f = xf ** 2
        return f

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        return compiler_options


class TestOptimizationProblem(TestCase):
    """
    Tests the default methods from OptimizationProblem
    """

    def setUp(self):
        self.problem = Model()
        self.problem.read()
        self.tolerance = 1e-6

    def test_get_timeseries(self):
        timeseries = self.problem.get_timeseries('constant_input')
        expected_times = [-7200, -3600, 0, 3600, 7200, 9800]
        self.assertTrue(np.array_equal(timeseries.times, expected_times))
        expected_values = [1.1, 1.4, 0.9, 1.2, 1.5, 1.7]
        self.assertTrue(np.array_equal(timeseries.values, expected_values))

        timeseries_x = self.problem.get_timeseries('x')
        self.assertTrue(np.array_equal(timeseries_x.times, expected_times))
        expected_values = [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]
        self.assertTrue(np.array_equal(timeseries_x.values, expected_values))

    def test_set_timeseries_with_timeseries(self):
        times = self.problem.io.get_times()
        values = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1]
        self.problem.set_timeseries('newVar', Timeseries(times, values))

        actual_series = self.problem.get_timeseries('newVar')
        self.assertTrue(np.array_equal(actual_series.values, values))
        self.assertTrue(np.array_equal(actual_series.times, times))

        # test if it was actually stored in the internal data store
        actual_values = self.problem.io.get_timeseries_values('newVar')
        self.assertTrue(np.array_equal(actual_values, values))

        # now let's do this again but only give part of the values
        values = [1.1, 2.1, 3.1]

        # with check_consistency=True (default) we should get a ValueError
        with self.assertRaises(ValueError):
            self.problem.set_timeseries('partialSeries', Timeseries(times[-3:], values))

        self.problem.set_timeseries('partialSeries', Timeseries(times[-3:], values), check_consistency=False)

        actual_series = self.problem.get_timeseries('partialSeries')
        self.assertTrue(np.array_equal(actual_series.times, times))
        self.assertTrue(np.array_equal(actual_series.values[-3:], values))
        self.assertTrue(np.all(np.isnan(actual_series.values[:-3])))

    def test_set_timeseries_with_array(self):
        times = self.problem.times()
        values = np.ones(times.shape)
        self.problem.set_timeseries('newVar', values)

        actual_series = self.problem.get_timeseries('newVar')
        forecast_index = self.problem.io.get_forecast_index()
        self.assertTrue(np.array_equal(actual_series.values[forecast_index:], values))
        self.assertTrue(np.all(np.isnan(actual_series.values[:forecast_index])))

    def test_timeseries_at(self):
        times = self.problem.io.get_times()
        values = times.astype(dtype=np.float64) / 10
        self.problem.set_timeseries('myVar', Timeseries(times, values))
        actual = self.problem.timeseries_at('myVar', times[0])
        self.assertEqual(actual, times[0] / 10)

        actual = self.problem.timeseries_at('myVar', (times[0] + times[1]) / 2)
        self.assertEqual(actual, (values[0] + values[1]) / 2)

    def test_bounds(self):
        bounds = self.problem.bounds()
        self.assertEqual(bounds['x'], [float("-inf"), float("inf")])

        min_u = bounds['u'][0]
        max_u = bounds['u'][1]

        expected_times = [0, 3600, 7200, 9800]
        self.assertTrue(np.array_equal(min_u.times, expected_times))
        self.assertTrue(np.array_equal(max_u.times, expected_times))

        expected_min_values = [0.3, 0.1, 0.4, 0.0]
        self.assertTrue(np.array_equal(min_u.values, expected_min_values))

        expected_max_values = [2.0, 2.4, 2.5, 2.3]
        self.assertTrue(np.array_equal(max_u.values, expected_max_values))

    def test_history(self):
        history = self.problem.history(0)

        expected_times = [-7200, -3600, 0]
        self.assertTrue(np.array_equal(history['x'].times, expected_times))
        self.assertTrue(np.array_equal(history['constant_input'].times, expected_times))

        expected_history_x = [3.1, 3.2, 3.3]
        self.assertTrue(np.array_equal(history['x'].values, expected_history_x))
        expected_history_u = [1.1, 1.4, 0.9]
        self.assertTrue(np.array_equal(history['constant_input'].values, expected_history_u))

    def test_seed(self):
        # add another variable containing some nans
        self.problem.io.set_timeseries_values('some_missing', np.array([np.nan, 0.1, 0.2, np.nan, 3.1, np.nan]))
        self.problem.dae_variables['free_variables'].append(ca.MX().sym('some_missing'))

        seed = self.problem.seed(0)
        self.assertTrue(np.array_equal(seed['x'].values, [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]))
        self.assertTrue(np.array_equal(seed['alias'].values, [3.1, 3.2, 3.3, 3.4, 3.5, 3.6]))
        self.assertTrue(np.array_equal(seed['some_missing'].values, [0, 0.1, 0.2, 0, 3.1, 0]))

    def test_constant_inputs(self):
        constant_inputs = self.problem.constant_inputs(0)
        self.assertTrue(np.array_equal(constant_inputs['constant_input'].values, [1.1, 1.4, 0.9, 1.2, 1.5, 1.7]))
