import logging
import os

import rtctools.data.netcdf as netcdf
from rtctools.data import rtc
from rtctools.optimization.io_mixin import IOMixin

logger = logging.getLogger("rtctools")

# todo add support for ensembles


class NetCDFMixin(IOMixin):
    """
    Adds NetCDF I/O to your optimization problem.

    During preprocessing, a file named timeseries_import.nc is read from the ``input`` subfolder.
    During postprocessing a file named timeseries_export.nc is written to the ``output`` subfolder.

    Both the input and output nc files are expected to follow the FEWS format for scalar data in a Netcdf file, i.e.:
     - They must contain a variable with the station id's (location id's) which can be recognized by the attribute
       'cf_role' set to 'timeseries_id'.
     - They must contain a time variable with attributes 'standard_name' = 'time' and 'axis' = 'T'

    From the input file, all 2d variables with dimensions equal to the station id's and time variable are read.

    To determine the rtc-tools variable name, the NetCDF mixin uses the station id (location id) and name of the
    timeseries variable in the file (parameter). An rtcDataConfig.xml file can be given in the input folder to
    configure variable names for specific location and parameter combinations. If this file is present, and contains
    a configured variable name for a read timeseries, this variable name will be used. If the file is present, but does
    not contain a configured variable name, a default variable name is constructed and a warning is given to alert the
    user that the current rtcDataConfig may contain a mistake. To suppress this warning if this is intentional, set the
    check_missing_variable_names attribute to False. Finally, if no file is present, the default variable name will
    always be used, and no warnings will be given. With debug logging enabled, the NetCDF mixin will report the chosen
    variable name for each location and parameter combination.

    To construct the default variable name, the station id is concatenated with the name of the variable in the NetCDF
    file, separted by the location_parameter_delimeter (set to a double underscore - '__' - by default). For example,
    if a NetCDF file contains two stations 'loc_1' and 'loc_2', and a timeseries variable called 'water_level', this
    will result in two rtc-tools variables called 'loc_1__water_level' and 'loc_2__water_level' (with the default
    location_parameter_delimiter of '__').

    :cvar location_parameter_delimiter:
        Delimiter used between location and parameter id when constructing the variable name.
    :cvar check_missing_variable_names:
        Warn if an rtcDataConfig.xml file is given but does not contain a variable name for a read timeseries.
        Default is ``True``
    :cvar netcdf_validate_timeseries:
        Check consistency of timeseries. Default is ``True``

    """

    #: Delimiter used between location and parameter id when constructing the variable name.
    location_parameter_delimiter = '__'

    #: Warn if an rtcDataConfig.xml file is given but does not contain a variable name for a read timeseries.
    check_missing_variable_names = True

    #: Check consistency of timeseries.
    netcdf_validate_timeseries = True

    def __init__(self, **kwargs):
        # call parent class for default behaviour
        super().__init__(**kwargs)

        path = os.path.join(self._input_folder, "rtcDataConfig.xml")
        self.__data_config = rtc.DataConfig(self._input_folder) if os.path.isfile(path) else None

    def read(self):
        # Call parent class first for default behaviour
        super().read()

        dataset = netcdf.ImportDataset(self._input_folder, self.timeseries_import_basename)

        # convert and store the import times
        self.__import_datetimes = dataset.read_import_times()
        times = self.io.datetime_to_sec(self.__import_datetimes, self.__import_datetimes[self.io.get_forecast_index()])
        self.io.set_times(times)

        if self.netcdf_validate_timeseries:
            # check if strictly increasing
            for i in range(len(times) - 1):
                if times[i] >= times[i + 1]:
                    raise Exception('NetCDFMixin: Time stamps must be strictly increasing.')

        self.__dt = times[1] - times[0] if len(times) >= 2 else 0
        for i in range(len(times) - 1):
            if times[i + 1] - times[i] != self.__dt:
                self.__dt = None
                break

        # store the station data for later use
        self.__stations = dataset.read_station_data()

        # read all available timeseries from the dataset
        timeseries_var_keys = dataset.find_timeseries_variables()

        # todo add support for ensembles

        for parameter in timeseries_var_keys:
            for i, location_id in enumerate(self.__stations.station_ids):
                default_name = location_id + self.location_parameter_delimiter + parameter
                if self.__data_config is not None:
                    try:
                        name = self.__data_config.parameter(parameter, location_id)
                    except KeyError:
                        if self.check_missing_variable_names:
                            logger.warning('No configured variable name found in rtcDataConfig.xml for location id "{}"'
                                           ' and parameter id "{}", using default variable name "{}" instead. '
                                           '(To suppress this warning set check_missing_variable_names to False.)'
                                           .format(location_id, parameter, default_name))
                        name = default_name
                else:
                    name = default_name

                values = dataset.read_timeseries_values(i, parameter)
                self.io.set_timeseries_values(name, values)
                logger.debug('Read timeseries data for location id "{}" and parameter "{}", '
                             'stored under variable name "{}"'
                             .format(location_id, parameter, name))

        logger.debug("NetCDFMixin: Read timeseries")

    def write(self):
        dataset = netcdf.ExportDataset(self._output_folder, self.timeseries_export_basename)

        times = self.times()

        forecast_index = self.io.get_forecast_index()
        dataset.write_times(times, self.initial_time, self.__import_datetimes[forecast_index])

        output_variables = [sym.name() for sym in self.output_variables]
        output_location_parameter_ids = {var_name: self.extract_station_id(var_name) for var_name in output_variables}
        output_station_ids = {loc_par[0] for loc_par in output_location_parameter_ids.values()}
        dataset.write_station_data(self.__stations, output_station_ids)

        output_parameter_ids = {loc_par[1] for loc_par in output_location_parameter_ids.values()}
        dataset.create_variables(output_parameter_ids)

        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            for var_name in output_variables:
                # determine the output values
                try:
                    values = results[var_name]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times, self.times(var_name), values, self.interpolation_method(var_name))
                except KeyError:
                    try:
                        ts = self.get_timeseries(var_name, ensemble_member)
                        if len(ts.times) != len(times):
                            values = self.interpolate(
                                times, ts.times, ts.values)
                        else:
                            values = ts.values
                    except KeyError:
                        logger.error(
                            'NetCDFMixin: Output requested for non-existent variable {}. '
                            'Will not be in output file.'.format(var_name))
                        continue

                # determine where to put this output
                location_parameter_id = output_location_parameter_ids[var_name]
                location_id = location_parameter_id[0]
                parameter_id = location_parameter_id[1]
                dataset.write_output_values(location_id, parameter_id, values)

        dataset.close()

    def extract_station_id(self, variable_name: str) -> tuple:
        """
        Returns the station id corresponding to the given RTC-Tools variable name.

        :param variable_name: The name of the RTC-Tools variable
        :return: the station id
        """
        try:
            return self.__data_config.pi_variable_ids(variable_name)[:2]
        except KeyError:
            return tuple(variable_name.split(self.location_parameter_delimiter))

    @property
    def equidistant(self):
        return self.__dt is not None
