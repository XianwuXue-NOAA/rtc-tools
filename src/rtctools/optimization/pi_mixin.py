import logging
import math
from datetime import timedelta
from typing import Dict, Union

import numpy as np

import rtctools.data.pi as pi
import rtctools.data.rtc as rtc
from rtctools.optimization.io_mixin import IOMixin

logger = logging.getLogger("rtctools")


class PIMixin(IOMixin):
    """
    Adds `Delft-FEWS Published Interface
    <https://publicwiki.deltares.nl/display/FEWSDOC/The+Delft-Fews+Published+Interface>`_
    I/O to your optimization problem.

    During preprocessing, files named ``rtcDataConfig.xml``, ``timeseries_import.xml``,
    ``rtcParameterConfig.xml``, and ``rtcParameterConfig_Numerical.xml`` are read from the
    ``input`` subfolder.  ``rtcDataConfig.xml`` maps tuples of FEWS identifiers, including
    location and parameter ID, to RTC-Tools time series identifiers.

    During postprocessing, a file named ``timeseries_export.xml`` is written to the ``output``
    subfolder.

    :cvar pi_binary_timeseries:
        Whether to use PI binary timeseries format. Default is ``False``.
    :cvar pi_parameter_config_basenames:
        List of parameter config file basenames to read. Default is [``rtcParameterConfig``].
    :cvar pi_parameter_config_numerical_basename:
        Numerical config file basename to read. Default is ``rtcParameterConfig_Numerical``.
    :cvar pi_check_for_duplicate_parameters:
        Check if duplicate parameters are read. Default is ``True``.
    :cvar pi_validate_timeseries:
        Check consistency of timeseries. Default is ``True``.
    """

    #: Whether to use PI binary timeseries format
    pi_binary_timeseries = False

    #: Location of rtcParameterConfig files
    pi_parameter_config_basenames = ["rtcParameterConfig"]
    pi_parameter_config_numerical_basename = "rtcParameterConfig_Numerical"

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

    def imported_seed_options(self) -> Dict[str, Union[str, float]]:
        """
        Returns a dictionary of options controlling the seeding process.
        +------------------------------------------+------------+---------------+
        | Option                                   | Type       | Default value |
        +==========================================+============+===============+
        | ``import_seed``                          | ``Bool``   | ``False``     |
        +------------------------------------------+------------+---------------+
        | ``extend_seed_forwards``                 | ``Bool``   | ``True``      |
        +------------------------------------------+------------+---------------+
        | ``extend_seed_backwards``                | ``Bool``   | ``False``     |
        +------------------------------------------+------------+---------------+
        | ``seed_variables_in_timeseries_import``  | ``Bool``   | ``False``      |
        +------------------------------------------+------------+---------------+
        The seeding process is controlled by the seeding_options. If ``import_seed``
        is true then, The imported seed will be merged with the timeseries_import and used for
        seeding for the first priority. By default these files should be given the name
        "imported_seed". This can be changed by overwiting "imported_seed_basename".
        The seed could for example be a copy of the timeseries_export.xml from a previous run.

        If the imported seed ends before the current forecast horizon and  ``extend_seed_forwards``
        is True then the final value in the imported seed will be extrapolated to the end of the
        time horizon.

        If the imported seed starts after the current forecast horizon and ``extend_seed_backwards``
        is True then the initial value in the imported seed will be extrapolated to fill that gap
        at the beginning of the timehorizon.

        Note that extending a seed backwards or seeding variables included in the timeseries_import
        may lead to undesirable effects if a controlled input is seeded.

        :returns: A dictionary of options for importing an isolated seed.
        """

        return {
            "import_seed": False,
            "extend_seed_forwards": False,
            "extend_seed_backwards": False,
        }

    def read(self):
        # Call parent class first for default behaviour.
        super().read()

        # rtcParameterConfig
        self.__parameter_config = []
        try:
            for pi_parameter_config_basename in self.pi_parameter_config_basenames:
                self.__parameter_config.append(
                    pi.ParameterConfig(self._input_folder, pi_parameter_config_basename)
                )
        except IOError:
            raise Exception(
                "PIMixin: {}.xml not found in {}.".format(
                    pi_parameter_config_basename, self._input_folder
                )
            )

        try:
            self.__parameter_config_numerical = pi.ParameterConfig(
                self._input_folder, self.pi_parameter_config_numerical_basename
            )
        except IOError:
            self.__parameter_config_numerical = None

        try:
            self.__timeseries_import = pi.Timeseries(
                self.__data_config,
                self._input_folder,
                self.timeseries_import_basename,
                binary=self.pi_binary_timeseries,
                pi_validate_times=self.pi_validate_timeseries,
            )
        except IOError:
            raise Exception(
                "PIMixin: {}.xml not found in {}.".format(
                    self.timeseries_import_basename, self._input_folder
                )
            )

        self.__timeseries_export = pi.Timeseries(
            self.__data_config,
            self._output_folder,
            self.timeseries_export_basename,
            binary=self.pi_binary_timeseries,
            pi_validate_times=False,
            make_new_file=True,
        )

        # Convert timeseries timestamps to seconds since t0 for internal use
        timeseries_import_times = self.__timeseries_import.times

        # Timestamp check
        if self.pi_validate_timeseries:
            for i in range(len(timeseries_import_times) - 1):
                if timeseries_import_times[i] >= timeseries_import_times[i + 1]:
                    raise Exception("PIMixin: Time stamps must be strictly increasing.")

        if self.__timeseries_import.dt:
            # Check if the timeseries are truly equidistant
            if self.pi_validate_timeseries:
                dt = timeseries_import_times[1] - timeseries_import_times[0]
                for i in range(len(timeseries_import_times) - 1):
                    if timeseries_import_times[i + 1] - timeseries_import_times[i] != dt:
                        raise Exception(
                            "PIMixin: Expecting equidistant timeseries, the time step "
                            "towards {} is not the same as the time step(s) before. Set "
                            "unit to nonequidistant if this is intended.".format(
                                timeseries_import_times[i + 1]
                            )
                        )

        # Offer input timeseries to IOMixin
        self.io.reference_datetime = self.__timeseries_import.forecast_datetime

        # If an imported seed has been provided merge it with the timeseries_import
        # TODO if seed is missing or wrong then use default seed instead of raising exceptions
        # TODO be careful for adding seeds for variables with fixed=false!
        imported_seed_options = self.imported_seed_options()
        if imported_seed_options["import_seed"]:
            try:
                self.__imported_seed_timeseries = pi.Timeseries(
                    self.__data_config,
                    self._input_folder,
                    self.imported_seed_basename,
                    binary=self.pi_binary_timeseries,
                    pi_validate_times=self.pi_validate_timeseries,
                )
            except IOError:
                raise Exception(
                    "PIMixin: {}.xml not found in {}.".format(
                        self.imported_seed_basename, self._input_folder
                    )
                )

            imported_seed_times = self.__imported_seed_timeseries.times

            # Timestamp check
            if self.pi_validate_timeseries:
                for i in range(len(imported_seed_times) - 1):
                    if imported_seed_times[i] >= imported_seed_times[i + 1]:
                        raise Exception("PIMixin: Time stamps must be strictly increasing.")

            # Check if the timeseries are truly equidistant
            if self.pi_validate_timeseries:
                dt = imported_seed_times[1] - imported_seed_times[0]
                for i in range(len(imported_seed_times) - 1):
                    if imported_seed_times[i + 1] - imported_seed_times[i] != dt:
                        raise Exception(
                            "PIMixin: Expecting equidistant timeseries, the time step "
                            "towards {} is not the same as the time step(s) before. Seeding using "
                            "an imported result is only supported for equidistant timesteps".format(
                                imported_seed_times[i + 1]
                            )
                        )
                    # Check if timestep is same as timeseries_import
                    if dt != self.__timeseries_import.dt:
                        raise Exception(
                            "PIMixin: The timesteps in timeseries_import {} differ from the "
                            "timesteps in the imported previous result {}. This is not "
                            "supported".format(self.__timeseries_import.dt, dt)
                        )

            imported_seed_times_t0 = imported_seed_times[0]
            t0_difference = self.io.reference_datetime - imported_seed_times_t0
            index_difference = int(t0_difference / dt)

            # Check that timeseries_import values are in the seed
            if len(imported_seed_times) < abs(index_difference):
                # TODO should not be an exception
                raise Exception(
                    "Imported result does not overlap with {} range. "
                    "Default seed is used".format(self.timeseries_import_basename)
                )
            times = timeseries_import_times

            # timeseries_import_variables_dict = {}
            for ensemble_member in range(self.__timeseries_import.ensemble_size):
                for variable, values in self.__timeseries_import.items(ensemble_member):
                    self.io.set_timeseries(
                        variable, timeseries_import_times, values, ensemble_member
                    )

            for ensemble_member in range(self.__imported_seed_timeseries.ensemble_size):
                for variable, values in self.__imported_seed_timeseries.items(ensemble_member):
                    write_ts = True
                    if index_difference >= 0:
                        values = np.asarray(values[index_difference:], dtype=np.float64)
                        if len(times) < len(values):
                            values = values[: len(times)]
                        elif len(times) > len(values):
                            if imported_seed_options["extend_seed_forwards"]:
                                # extend the last entry
                                values = np.append(
                                    values, [values[-1]] * (len(times) - len(values))
                                )
                            else:
                                values = np.append(values, np.nan * (len(times) - len(values)))
                    else:
                        values = np.asarray(values, dtype=np.float64)
                        if imported_seed_options["extend_seed_backwards"]:
                            # extend first entry back to t0
                            values = np.append([values[0]] * abs(index_difference), values)
                        else:
                            values = np.append(np.nan * abs(index_difference), values)
                        if len(times) < len(values):
                            values = values[: len(times)]
                        elif len(times) > len(values):
                            if imported_seed_options["extend_seed_forwards"]:
                                # extend the last entry
                                values = np.append(
                                    values, [values[-1]] * (len(times) - len(values))
                                )
                            else:
                                values = np.append(values, np.nan * (len(times) - len(values)))
                    for (
                        timeseries_import_variable,
                        timeseries_import_values,
                    ) in self.__timeseries_import.items(ensemble_member):
                        if timeseries_import_variable == variable:
                            if not imported_seed_options["seed_variables_in_timeseries_import"]:
                                write_ts = False
                            elif np.any(timeseries_import_values):
                                values = [
                                    a if not math.isnan(a) else b
                                    for a, b in zip(timeseries_import_values, values)
                                ]
                            else:
                                write_ts = False
                            break
                    if write_ts:
                        self.io.set_timeseries(
                            variable, timeseries_import_times, values, ensemble_member
                        )

            logger.info("PIMixin: updated imported timeseries with data from imported seed.")

        for ensemble_member in range(self.__timeseries_import.ensemble_size):
            if not imported_seed_options["import_seed"]:
                for variable, values in self.__timeseries_import.items(ensemble_member):
                    self.io.set_timeseries(
                        variable, timeseries_import_times, values, ensemble_member
                    )

            # store the parameters in the internal data store. Note that we
            # are effectively broadcasting parameters, as ParameterConfig does
            # not support parameters varying per ensemble member
            for parameter_config in self.__parameter_config:
                for location_id, model_id, parameter_id, value in parameter_config:
                    try:
                        parameter = self.__data_config.parameter(
                            parameter_id, location_id, model_id
                        )
                    except KeyError:
                        parameter = parameter_id

                    self.io.set_parameter(
                        parameter,
                        value,
                        ensemble_member,
                        check_duplicates=self.pi_check_for_duplicate_parameters,
                    )

    def solver_options(self):
        # Call parent
        options = super().solver_options()

        # Only do this if we have a rtcParameterConfig_Numerical
        if self.__parameter_config_numerical is None:
            return options

        # Load solver options from parameter config
        for _location_id, _model, option, value in self.__parameter_config_numerical:
            options[option] = value

        # Done
        return options

    def write(self):
        # Call parent class first for default behaviour.
        super().write()

        # Get time stamps
        times = self.times()
        if len(set(times[1:] - times[:-1])) == 1:
            dt = timedelta(seconds=times[1] - times[0])
        else:
            dt = None

        # Start of write output
        # Write the time range for the export file.
        self.__timeseries_export.times = [
            self.__timeseries_import.times[self.__timeseries_import.forecast_index]
            + timedelta(seconds=s)
            for s in times
        ]

        # Write other time settings
        self.__timeseries_export.forecast_datetime = self.__timeseries_import.forecast_datetime
        self.__timeseries_export.dt = dt
        self.__timeseries_export.timezone = self.__timeseries_import.timezone

        # Write the ensemble properties for the export file.
        if self.ensemble_size > 1:
            self.__timeseries_export.contains_ensemble = True
        self.__timeseries_export.ensemble_size = self.ensemble_size
        self.__timeseries_export.contains_ensemble = self.ensemble_size > 1

        # Start looping over the ensembles for extraction of the output values.
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)

            # For all variables that are output variables the values are
            # extracted from the results.
            for variable in [sym.name() for sym in self.output_variables]:
                for alias in self.alias_relation.aliases(variable):
                    try:
                        values = results[alias]
                        if len(values) != len(times):
                            values = self.interpolate(
                                times, self.times(alias), values, self.interpolation_method(alias)
                            )
                    except KeyError:
                        try:
                            ts = self.get_timeseries(alias, ensemble_member)
                            if len(ts.times) != len(times):
                                values = self.interpolate(times, ts.times, ts.values)
                            else:
                                values = ts.values
                        except KeyError:
                            logger.error(
                                "PIMixin: Output requested for non-existent alias {}. "
                                "Will not be in output file.".format(alias)
                            )
                            continue

                    # Check if ID mapping is present
                    try:
                        self.__data_config.pi_variable_ids(alias)
                    except KeyError:
                        logger.debug(
                            "PIMixin: variable {} has no mapping defined in rtcDataConfig "
                            "so cannot be added to the output file.".format(alias)
                        )
                        continue

                    # Add series to output file.
                    # NOTE: We use the unit of the zeroth ensemble member, as
                    # we might be outputting more ensembles than we read in.
                    self.__timeseries_export.set(
                        alias,
                        values,
                        unit=self.__timeseries_import.get_unit(alias, ensemble_member=0),
                        ensemble_member=ensemble_member,
                    )

        # Write output file to disk
        self.__timeseries_export.write()

    def set_timeseries(self, variable: str, *args, unit: str = None, **kwargs):
        if unit is not None:
            self.__timeseries_import.set_unit(variable, unit, 0)

        super().set_timeseries(variable, *args, **kwargs)

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
        return self.io.times_sec

    @property
    def timeseries_export(self):
        """
        :class:`pi.Timeseries` object for holding the output data.
        """
        return self.__timeseries_export

    def set_unit(self, variable: str, unit: str):
        """
        Set the unit of a time series.

        :param variable:        Time series ID.
        :param unit:            Unit.
        """
        assert hasattr(
            self, "_PIMixin__timeseries_import"
        ), "set_unit can only be called after read() in pre() has finished."
        self.__timeseries_import.set_unit(variable, unit, 0)
        self.__timeseries_export.set_unit(variable, unit, 0)
