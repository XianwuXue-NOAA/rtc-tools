import os
from datetime import datetime, timedelta
from unittest import TestCase

from netCDF4 import Dataset

import numpy as np

import rtctools.data.netcdf as netcdf

from .data_path import data_path


class TestImportDataset(TestCase):

    def setUp(self):
        self.dataset = netcdf.ImportDataset(data_path(), 'timeseries_import')

    def test_init(self):
        time_var = self.dataset.time_variable
        self.assertEqual(time_var._name, 'time')
        self.assertEqual(time_var.standard_name, 'time')
        self.assertEqual(time_var.long_name, 'time')
        self.assertEqual(time_var.axis, 'T')
        self.assertEqual(time_var.units, 'minutes since 1970-01-01 00:00:00.0 +0000')

        station_var = self.dataset.station_variable
        self.assertEqual(station_var._name, 'station_id')
        self.assertEqual(station_var.long_name, 'station identification code')
        self.assertEqual(station_var.cf_role, 'timeseries_id')

    def test_read_times(self):
        datetimes = self.dataset.read_import_times()

        forecast_datetime = datetime(2013, 1, 15)
        expected_datetimes = [forecast_datetime + timedelta(hours=i) for i in range(25)]
        self.assertTrue(np.array_equal(datetimes, expected_datetimes))

    def test_find_timeseries_variables(self):
        variables = self.dataset.find_timeseries_variables()
        self.assertEqual(variables, ['waterlevel'])

    def test_stations(self):
        stations = self.dataset.read_station_data()

        ids = stations.station_ids
        self.assertEqual(len(ids), 3)
        self.assertTrue('LocA' in ids)
        self.assertTrue('LocB' in ids)
        self.assertTrue('LocC' in ids)

        for id in ids:
            read_attributes = stations.attributes[id].keys()
            self.assertTrue(len(read_attributes), 5)
            self.assertTrue('lat' in read_attributes)
            self.assertTrue('lon' in read_attributes)
            self.assertTrue('x' in read_attributes)
            self.assertTrue('y' in read_attributes)
            self.assertTrue('z' in read_attributes)

        self.assertEqual(stations.attributes['LocA']['lat'], 53.0)


class TestExportDataset(TestCase):
    def get_exported_dataset(self):
        filename = os.path.join(
            data_path(),
            'timeseries_export.nc'
        )
        return Dataset(filename)

    def setUp(self):
        self.dataset = netcdf.ExportDataset(data_path(), 'timeseries_export')

    def test_write_times(self):
        times = np.array([-120, -300, -60, 300, 360])
        self.dataset.write_times(times, -180.0, datetime(2018, 12, 21, 17, 30))
        self.dataset.close()

        dataset = self.get_exported_dataset()
        self.assertTrue('time' in dataset.variables)

        time_var = dataset.variables['time']
        self.assertEqual(time_var.units, 'seconds since 2018-12-21 17:28:00')
        self.assertEqual(time_var.axis, 'T')
        self.assertEqual(time_var.standard_name, 'time')
        self.assertTrue(np.array_equal(time_var[:], times + 300))

    # todo create tests for write_station_data, create_variables and write_output_values
