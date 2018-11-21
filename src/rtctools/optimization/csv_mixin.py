import logging
import os
from datetime import timedelta

import numpy as np

import rtctools.data.csv as csv
from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached
from rtctools.optimization.io_mixin import IOMixin

logger = logging.getLogger("rtctools")


class CSVMixin(IOMixin):
    """
    Adds reading and writing of CSV timeseries and parameters to your optimization problem.

    During preprocessing, files named ``timeseries_import.csv``, ``initial_state.csv``,
    and ``parameters.csv`` are read from the ``input`` subfolder.

    During postprocessing, a file named ``timeseries_export.csv`` is written to the ``output`` subfolder.

    In ensemble mode, a file named ``ensemble.csv`` is read from the ``input`` folder.  This file
    contains two columns. The first column gives the name of the ensemble member, and the second
    column its probability.  Furthermore, the other XML files appear one level deeper inside the
    filesystem hierarchy, inside subfolders with the names of the ensemble members.

    :cvar csv_delimiter:
        Column delimiter used in CSV files.  Default is ``,``.
    :cvar csv_equidistant:
        Whether or not the timeseries data is equidistant.  Default is ``True``.
    :cvar csv_ensemble_mode:
        Whether or not to use ensembles.  Default is ``False``.
    :cvar csv_validate_timeseries:
        Check consistency of timeseries.  Default is ``True``.
    """

    #: Column delimiter used in CSV files
    csv_delimiter = ','

    #: Whether or not the timeseries data is equidistant
    csv_equidistant = True

    #: Whether or not to use ensembles
    csv_ensemble_mode = False

    #: Check consistency of timeseries
    csv_validate_timeseries = True

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    def read(self):
        # Call parent class first for default behaviour.
        super().read()

        # Helper function to check if initial state array actually defines
        # only the initial state
        def check_initial_state_array(initial_state):
            """
            Check length of initial state array, throw exception when larger than 1.
            """
            if initial_state.shape:
                raise Exception(
                    'CSVMixin: Initial state file {} contains more than one row of data. '
                    'Please remove the data row(s) that do not describe the initial state.'.format(
                        os.path.join(self._input_folder, 'initial_state.csv')))

        # Read CSV files
        self.__initial_state = []
        if self.csv_ensemble_mode:
            self.__ensemble = np.genfromtxt(
                os.path.join(self._input_folder, 'ensemble.csv'),
                delimiter=",", deletechars='', dtype=None, names=True, encoding=None)

            logger.debug("CSVMixin: Read ensemble description")

            for ensemble_member_index, ensemble_member_name in enumerate(self.__ensemble['name']):
                _timeseries = csv.load(
                    os.path.join(
                        self._input_folder,
                        ensemble_member_name,
                        self.timeseries_import_basename + ".csv",
                    ),
                    delimiter=self.csv_delimiter,
                    with_time=True,
                )
                self.__timeseries_times = _timeseries[_timeseries.dtype.names[0]]
                self.set_times(
                    self.datetime_to_sec(
                        self.__timeseries_times,
                        self.__timeseries_times[self.get_forecast_index()]
                    )
                )
                for key in _timeseries.dtype.names[1:]:
                    self.set_timeseries_values(
                        key,
                        np.asarray(_timeseries[key], dtype=np.float64),
                        ensemble_member_index
                    )
            logger.debug("CSVMixin: Read timeseries")

            for ensemble_member_index, ensemble_member_name in enumerate(self.__ensemble['name']):
                try:
                    _parameters = csv.load(os.path.join(
                        self._input_folder, ensemble_member_name, 'parameters.csv'), delimiter=self.csv_delimiter)
                    for key in _parameters.dtype.names:
                        self.set_parameter(key, float(_parameters[key]), ensemble_member_index)
                except IOError:
                    pass
            logger.debug("CSVMixin: Read parameters.")

            for ensemble_member_name in self.__ensemble['name']:
                try:
                    _initial_state = csv.load(os.path.join(
                        self._input_folder, ensemble_member_name, 'initial_state.csv'), delimiter=self.csv_delimiter)
                    check_initial_state_array(_initial_state)
                    _initial_state = {key: float(_initial_state[key]) for key in _initial_state.dtype.names}
                except IOError:
                    _initial_state = {}
                self.__initial_state.append(AliasDict(self.alias_relation, _initial_state))
            logger.debug("CSVMixin: Read initial state.")
        else:
            _timeseries = csv.load(
                os.path.join(
                    self._input_folder, self.timeseries_import_basename + ".csv"
                ),
                delimiter=self.csv_delimiter,
                with_time=True,
            )
            self.__timeseries_times = _timeseries[_timeseries.dtype.names[0]]
            self.set_times(
                self.datetime_to_sec(
                    self.__timeseries_times,
                    self.__timeseries_times[self.get_forecast_index()]
                )
            )
            for key in _timeseries.dtype.names[1:]:
                self.set_timeseries_values(key, np.asarray(_timeseries[key], dtype=np.float64))
            logger.debug("CSVMixin: Read timeseries.")

            try:
                _parameters = csv.load(os.path.join(
                    self._input_folder, 'parameters.csv'), delimiter=self.csv_delimiter)
                logger.debug("CSVMixin: Read parameters.")
                for key in _parameters.dtype.names:
                    self.set_parameter(key, float(_parameters[key]))
            except IOError:
                pass

            try:
                _initial_state = csv.load(os.path.join(
                    self._input_folder, 'initial_state.csv'), delimiter=self.csv_delimiter)
                logger.debug("CSVMixin: Read initial state.")
                check_initial_state_array(_initial_state)
                _initial_state = {key: float(_initial_state[key]) for key in _initial_state.dtype.names}
            except IOError:
                _initial_state = {}
            self.__initial_state.append(AliasDict(self.alias_relation, _initial_state))

        timeseries_times_sec = self.get_times()

        # Timestamp check
        if self.csv_validate_timeseries:
            for i in range(len(timeseries_times_sec) - 1):
                if timeseries_times_sec[i] >= timeseries_times_sec[i + 1]:
                    raise Exception(
                        'CSVMixin: Time stamps must be strictly increasing.')

        if self.csv_equidistant:
            # Check if the timeseries are truly equidistant
            if self.csv_validate_timeseries:
                dt = timeseries_times_sec[1] - timeseries_times_sec[0]
                for i in range(len(timeseries_times_sec) - 1):
                    if timeseries_times_sec[i + 1] - timeseries_times_sec[i] != dt:
                        raise Exception(
                            'CSVMixin: Expecting equidistant timeseries, the time step towards '
                            '{} is not the same as the time step(s) before. Set csv_equidistant = False '
                            'if this is intended.'.format(self.__timeseries_times[i + 1]))

    @property
    def equidistant(self):
        return self.csv_equidistant

    @property
    def ensemble_size(self):
        if self.csv_ensemble_mode:
            return len(self.__ensemble['probability'])
        else:
            return 1

    def ensemble_member_probability(self, ensemble_member):
        if self.csv_ensemble_mode:
            return self.__ensemble['probability'][ensemble_member]
        else:
            return 1.0

    @cached
    def parameters(self, ensemble_member):
        # Call parent class first for default values.
        parameters = super().parameters(ensemble_member)

        # Load parameters from parameter config
        for parameter in self.dae_variables['parameters']:
            parameter = parameter.name()
            try:
                parameters[parameter] = self.get_parameter(parameter, ensemble_member)
            except KeyError:
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read parameter {} ".format(parameter))
        return parameters

    @cached
    def initial_state(self, ensemble_member):
        # Call parent class first for default values.
        initial_state = super().initial_state(ensemble_member)

        # Load parameters from parameter config
        for variable in self.dae_variables['free_variables']:
            variable = variable.name()
            try:
                initial_state[variable] = self.__initial_state[ensemble_member][variable]
            except (KeyError, ValueError):
                pass
            else:
                if logger.getEffectiveLevel() == logging.DEBUG:
                    logger.debug("CSVMixin: Read initial state {}".format(variable))
        return initial_state

    def write(self):
        # Call parent class first for default behaviour.
        super().write()

        # Write output
        times = self.times()

        def write_output(ensemble_member, folder):
            results = self.extract_results(ensemble_member)
            names = ['time'] + sorted({sym.name() for sym in self.output_variables})
            formats = ['O'] + (len(names) - 1) * ['f8']
            dtype = {'names': names, 'formats': formats}
            data = np.zeros(len(times), dtype=dtype)
            data['time'] = [self.__timeseries_times[self.get_forecast_index()] + timedelta(seconds=s) for s in times]
            for output_variable in self.output_variables:
                output_variable = output_variable.name()
                try:
                    values = results[output_variable]
                    if len(values) != len(times):
                        values = self.interpolate(
                            times, self.times(output_variable), values, self.interpolation_method(output_variable))
                except KeyError:
                    try:
                        ts = self.get_timeseries(
                            output_variable, ensemble_member)
                        if len(ts.times) != len(times):
                            values = self.interpolate(
                                times, ts.times, ts.values)
                        else:
                            values = ts.values
                    except KeyError:
                        logger.error(
                            "Output requested for non-existent variable {}".format(output_variable))
                        continue
                data[output_variable] = values

            fname = os.path.join(folder, self.timeseries_export_basename + ".csv")
            csv.save(fname, data, delimiter=self.csv_delimiter, with_time=True)

        if self.csv_ensemble_mode:
            for ensemble_member, ensemble_member_name in enumerate(self.__ensemble['name']):
                write_output(ensemble_member, os.path.join(
                    self._output_folder, ensemble_member_name))
        else:
            write_output(0, self._output_folder)
