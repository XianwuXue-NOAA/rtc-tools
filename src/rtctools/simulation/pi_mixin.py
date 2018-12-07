import logging

import numpy as np

import rtctools.data.pi as pi
import rtctools.data.rtc as rtc
from rtctools.simulation.io_mixin import IOMixin

logger = logging.getLogger("rtctools")


class PIMixin(IOMixin):
    """
    Adds `Delft-FEWS Published Interface
    <https://publicwiki.deltares.nl/display/FEWSDOC/The+Delft-Fews+Published+Interface>`_
    I/O to your simulation problem.

    During preprocessing, files named ``rtcDataConfig.xml``, ``timeseries_import.xml``,  and``rtcParameterConfig.xml``
    are read from the ``input`` subfolder.  ``rtcDataConfig.xml`` maps
    tuples of FEWS identifiers, including location and parameter ID, to RTC-Tools time series identifiers.

    During postprocessing, a file named ``timeseries_export.xml`` is written to the ``output`` subfolder.

    :cvar pi_binary_timeseries: Whether to use PI binary timeseries format.  Default is ``False``.
    :cvar pi_parameter_config_basenames:
        List of parameter config file basenames to read. Default is [``rtcParameterConfig``].
    :cvar pi_check_for_duplicate_parameters: Check if duplicate parameters are read. Default is ``True``.
    :cvar pi_validate_timeseries: Check consistency of timeseries.  Default is ``True``.
    """

    #: Whether to use PI binary timeseries format
    pi_binary_timeseries = False

    #: Location of rtcParameterConfig files
    pi_parameter_config_basenames = ['rtcParameterConfig']

    #: Check consistency of timeseries
    pi_validate_timeseries = True

    #: Check for duplicate parameters
    pi_check_for_duplicate_parameters = True

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        # Load rtcDataConfig.xml.  We assume this file does not change over the
        # life time of this object.
        self.__data_config = rtc.DataConfig(self._input_folder)

    def read(self):
        # Call parent class first for default behaviour.
        super().read()

        # rtcParameterConfig
        self.__parameter_config = []
        try:
            for pi_parameter_config_basename in self.pi_parameter_config_basenames:
                self.__parameter_config.append(pi.ParameterConfig(
                    self._input_folder, pi_parameter_config_basename))
        except FileNotFoundError:
            raise FileNotFoundError(
                "PIMixin: {}.xml not found in {}.".format(pi_parameter_config_basename, self._input_folder))

        # Make a parameters dict for later access
        for parameter_config in self.__parameter_config:
            for location_id, model_id, parameter_id, value in parameter_config:
                try:
                    parameter = self.__data_config.parameter(parameter_id, location_id, model_id)
                except KeyError:
                    parameter = parameter_id
                self.io.set_parameter(parameter, value)

        try:
            self.__timeseries_import = pi.Timeseries(
                self.__data_config, self._input_folder, self.timeseries_import_basename,
                binary=self.pi_binary_timeseries, pi_validate_times=self.pi_validate_timeseries)
        except FileNotFoundError:
            raise FileNotFoundError('PIMixin: {}.xml not found in {}'.format(
                self.timeseries_import_basename, self._input_folder))

        self.__timeseries_export = pi.Timeseries(
            self.__data_config, self._output_folder, self.timeseries_export_basename,
            binary=self.pi_binary_timeseries, pi_validate_times=False, make_new_file=True)

        # Convert timeseries timestamps to seconds since t0 for internal use
        self.io.set_forecast_index(self.__timeseries_import.forecast_index)
        timeseries_import_times = np.asarray(self.io.datetime_to_sec(
            self.__timeseries_import.times,
            self.__timeseries_import.forecast_datetime
        ))
        self.io.set_times(timeseries_import_times)

        # Timestamp check
        if self.pi_validate_timeseries:
            for i in range(len(timeseries_import_times) - 1):
                if timeseries_import_times[i] >= timeseries_import_times[i + 1]:
                    raise ValueError(
                        'PIMixin: Time stamps must be strictly increasing.')

        # Check if the timeseries are equidistant
        dt = timeseries_import_times[1] - timeseries_import_times[0]
        if self.pi_validate_timeseries:
            for i in range(len(timeseries_import_times) - 1):
                if timeseries_import_times[i + 1] - timeseries_import_times[i] != dt:
                    raise ValueError(
                        'PIMixin: Expecting equidistant timeseries, the time step '
                        'towards {} is not the same as the time step(s) before. Set '
                        'unit to nonequidistant if this is intended.'.format(
                            self.__timeseries_import.times[i + 1]))

        # Stick timeseries into an AliasDict
        debug = logger.getEffectiveLevel() == logging.DEBUG
        for variable, values in self.__timeseries_import.items():
            self.io.set_timeseries_values(variable, values, check_duplicates=False)
            if debug and variable in self.get_variables():
                logger.debug('PIMixin: Timeseries {} replaced another aliased timeseries.'.format(variable))

    def write(self):
        # Call parent class first for default behaviour.
        super().write()

        # Start of write output
        # Write the time range for the export file.
        self.__timeseries_export.times = self.__timeseries_import.times[self.io.get_forecast_index():]

        # Write other time settings
        self.__timeseries_export.forecast_datetime = self.__timeseries_import.forecast_datetime
        self.__timeseries_export.dt = self.__timeseries_import.dt
        self.__timeseries_export.timezone = self.__timeseries_import.timezone

        # Write the ensemble properties for the export file.
        self.__timeseries_export.ensemble_size = 1
        self.__timeseries_export.contains_ensemble = self.__timeseries_import.contains_ensemble

        # For all variables that are output variables the values are
        # extracted from the results.
        for variable in self.output_variables:
            values = self.output[variable]
            # Check if ID mapping is present
            try:
                self.__data_config.pi_variable_ids(variable)
            except KeyError:
                logger.debug(
                    'PIMixin: variable {} has no mapping defined in rtcDataConfig '
                    'so cannot be added to the output file.'.format(variable))
                continue

            # Add series to output file
            self.__timeseries_export.set(variable, values, unit=self.__timeseries_import.get_unit(variable))

        # Write output file to disk
        self.__timeseries_export.write()

    @property
    def timeseries_import(self):
        """
        :class:`pi.Timeseries` object containing the input data.
        """
        return self.__timeseries_import

    @property
    def timeseries_import_times(self):
        """
        List of time stamps for which input data is specified.

        The time stamps are in seconds since t0, and may be negative.
        """
        return self.io.get_times()

    @property
    def timeseries_export(self):
        """
        :class:`pi.Timeseries` object for holding the output data.
        """
        return self.__timeseries_export

    def set_timeseries(self, variable, values, output=True, check_consistency=True, unit=None):
        if check_consistency:
            if len(self.times()) != len(values):
                raise ValueError(
                    'PIMixin: Trying to set/append values {} with a different '
                    'length than the forecast length. Please make sure the '
                    'values cover forecastDate through endDate with timestep {}.'.format(
                        variable, self.__timeseries_import.dt))

        if unit is None:
            unit = self.__timeseries_import.get_unit(variable)

        if output:
            try:
                self.__data_config.pi_variable_ids(variable)
            except KeyError:
                logger.debug(
                    'PIMixin: variable {} has no mapping defined in rtcDataConfig '
                    'so cannot be added to the output file.'.format(variable))
            else:
                self.__timeseries_export.set(variable, values, unit=unit)

        self.__timeseries_import.set(variable, values, unit=unit)
        self.io.set_timeseries_values(variable, values)

    def get_timeseries(self, variable):
        return self.io.get_timeseries_values(variable)

    def extract_results(self):
        """
        Extracts the results of output

        :returns: An AliasDict of output variables and results array format.
        """
        return self.__output
