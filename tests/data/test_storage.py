import logging
from unittest import TestCase

import numpy as np

from pymoca.backends.casadi.alias_relation import AliasRelation

from rtctools.data.storage import DataStoreAccessor

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class DummyDataStore(DataStoreAccessor):
    @property
    def alias_relation(self):
        return AliasRelation()


class TestDummyDataStore(TestCase):

    def setUp(self):
        self.datastore = DummyDataStore(input_folder='dummyInput', output_folder='dummyOutput')
        self.tolerance = 1e-6

    def test_times(self):
        expected_times = np.array([-7200, -3600, 0, 3600, 7200, 9800])
        self.datastore.io.set_times(expected_times)
        actual_times = self.datastore.io.get_times()
        self.assertTrue(np.array_equal(actual_times, expected_times))

    def test_forecast_index(self):
        forecast_index = self.datastore.io.get_forecast_index()
        self.assertEqual(forecast_index, 0)  # default forecast_index should be 0

        times = np.array([-7200, -3600, 0, 3600, 7200, 9800])
        self.datastore.io.set_times(times)
        initial_time = self.datastore.initial_time
        self.assertEqual(initial_time, -7200)

        self.datastore.io.set_forecast_index(3)
        self.assertEqual(self.datastore.io.get_forecast_index(), 3)
        self.assertEqual(self.datastore.initial_time, 3600)

    def test_timeseries(self):
        # expect a KeyError when getting a timeseries that has not been set
        with self.assertRaises(KeyError):
            self.datastore.io.get_timeseries_values('someNoneExistentVariable')

        # expect a RunTimeError when setting timeseries values before setting times
        with self.assertRaises(RuntimeError):
            self.datastore.io.set_timeseries_values('myNewVariable', np.array([3.1, 2.4, 2.5]))

        self.datastore.io.set_times(np.array([-3600, 0, 7200]))
        expected_values = np.array([3.1, 2.4, 2.5])
        self.datastore.io.set_timeseries_values('myNewVariable', expected_values)
        actual_values = self.datastore.io.get_timeseries_values('myNewVariable')
        self.assertTrue(np.array_equal(actual_values, expected_values))

        # expect a KeyError when getting timeseries for an ensemble member that doesn't exist
        with self.assertRaises(KeyError):
            self.datastore.io.get_timeseries_values('myNewVariable', 1)

        expected_values = np.array([1.1, 1.4, 1.5])
        self.datastore.io.set_timeseries_values('ensembleVariable', expected_values, ensemble_member=1)
        with self.assertRaises(KeyError):
            self.datastore.io.get_timeseries_values('ensembleVariable', 0)
        self.assertTrue(np.array_equal(self.datastore.io.get_timeseries_values('ensembleVariable', 1), expected_values))

        # expect a warning when overwriting a timeseries with check_duplicates=True (default)
        new_values = np.array([2.1, 1.1, 0.1])
        with self.assertLogs(logger, level='WARN') as cm:
            self.datastore.io.set_timeseries_values('myNewVariable', new_values)
            self.assertEqual(cm.output,
                             ['WARNING:rtctools:Attempting to set time series values for ensemble member 0 '
                              'and variable myNewVariable twice. Ignoring second set of values.'])
        self.assertFalse(np.array_equal(self.datastore.io.get_timeseries_values('myNewVariable'), new_values))

        # disable check to allow overwriting old values
        self.datastore.io.set_timeseries_values('myNewVariable', new_values, check_duplicates=False)
        self.assertTrue(np.array_equal(self.datastore.io.get_timeseries_values('myNewVariable'), new_values))

    def test_parameters(self):
        # expect a KeyError when getting a parameter that has not been set
        with self.assertRaises(KeyError):
            self.datastore.io.get_parameter('someNoneExistentParameter')

        self.datastore.io.set_parameter('myNewParameter', 1.4)
        self.assertEqual(self.datastore.io.get_parameter('myNewParameter'), 1.4)

        # expect a KeyError when getting parameters for an ensemble member that doesn't exist
        with self.assertRaises(KeyError):
            self.datastore.io.get_parameter('myNewParameter', 1)

        self.datastore.io.set_parameter('ensembleParameter', 1.2, ensemble_member=1)
        with self.assertRaises(KeyError):
            self.datastore.io.get_parameter('ensembleParameter', 0)
        self.assertEqual(self.datastore.io.get_parameter('ensembleParameter', 1), 1.2)

        # expect a warning when overwriting a parameter with check_duplicates=True (default)
        with self.assertLogs(logger, level='WARN') as cm:
            self.datastore.io.set_parameter('myNewParameter', 2.5)
            self.assertEqual(cm.output,
                             ['WARNING:rtctools:Attempting to set parameter value for ensemble member 0 '
                              'and name myNewParameter twice. Ignoring second set of values.'])
        self.assertEqual(self.datastore.io.get_parameter('myNewParameter'), 1.4)

        # disable check to allow overwriting old values
        self.datastore.io.set_parameter('myNewParameter', 2.2, check_duplicates=False)
        self.assertEqual(self.datastore.io.get_parameter('myNewParameter'), 2.2)

    # todo add tests that use newly added methods: get_variables, get_ensemble_size and get_parameter_names()
