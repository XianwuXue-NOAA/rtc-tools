import logging
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from typing import Iterable, Set, Union

import numpy as np

from rtctools._internal.alias_tools import AliasDict, AliasRelation

logger = logging.getLogger("rtctools")


class DataStore(metaclass=ABCMeta):
    """
    Base class for all problems.
    Adds an internal data store where which timeseries, parameters and initial states can be stored and read from.

    :cvar timeseries_import_basename:
        Import file basename. Default is ``timeseries_import``.
    :cvar timeseries_export_basename:
        Export file basename. Default is ``timeseries_export``.
    """

    #: Import file basename
    timeseries_import_basename = 'timeseries_import'
    #: Export file basename
    timeseries_export_basename = 'timeseries_export'

    def __init__(self, **kwargs):
        # Save arguments
        self._input_folder = kwargs['input_folder'] if 'input_folder' in kwargs else 'input'
        self._output_folder = kwargs['output_folder'] if 'output_folder' in kwargs else 'output'

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("Expecting input files to be located in '" + self._input_folder + "'.")
            logger.debug("Writing output files to '" + self._output_folder + "'.")

        # Should all be set by subclass via setters
        self.__forecast_index = 0
        self.__timeseries_times_sec = None
        self.__timeseries_values = []
        self.__parameters = []
        # todo add support for storing initial states
        # self.__initial_state = []

    def get_times(self) -> np.ndarray:
        """"
        Returns the timeseries times in seconds.

        :return timseries times in seconds, or None if there has been no call to set_times
        """
        return self.__timeseries_times_sec

    def set_times(self, times_in_sec: np.ndarray) -> None:
        """
        Sets the timeseries times in seconds in the internal data store.
        Must be called in .read() to store the times in the IOMixin before calling set_timeseries_values
        to store the values for an input timeseries.

        :param times_in_sec: np.ndarray containing the times in seconds
        """
        if self.__timeseries_times_sec is not None and not np.array_equal(times_in_sec, self.__timeseries_times_sec):
            raise RuntimeError("Attempting to overwrite the input time series times with different values. "
                               "Please ensure all input time series have the same times.")
        self.__timeseries_times_sec = times_in_sec

    def set_timeseries_values(self,
                              variable: str,
                              values: np.ndarray,
                              ensemble_member: int = 0,
                              check_duplicates: bool = True) -> None:
        """
        Stores input time series values in the internal data store.

        :param variable:         Variable name.
        :param values:           The values to be stored.
        :param ensemble_member:  The ensemble member index.
        :param check_duplicates: If True, a warning will be given when attempting to overwrite values.
                                 If False, existing values can be silently overwritten with new values.
        """
        if self.__timeseries_times_sec is None:
            raise RuntimeError("First call set_times before calling set_timeseries_values")

        if len(self.__timeseries_times_sec) != len(values):
            raise ValueError("Length of values ({}) must be the same as length of times ({})"
                             .format(len(values), len(self.__timeseries_times_sec)))

        while ensemble_member >= len(self.__timeseries_values):
            self.__timeseries_values.append(AliasDict(self.alias_relation))

        if check_duplicates and variable in self.__timeseries_values[ensemble_member].keys():
            logger.warning("Attempting to set time series values for ensemble member {} and variable {} twice. "
                           "Ignoring second set of values.".format(ensemble_member, variable))
            return

        self.__timeseries_values[ensemble_member][variable] = values

    def get_timeseries_values(self, variable: str, ensemble_member: int = 0) -> np.ndarray:
        """
        Looks up the time series values in the internal data store.
        """
        if ensemble_member >= len(self.__timeseries_values):
            raise KeyError("ensemble_member {} does not exist".format(ensemble_member))
        return self.__timeseries_values[ensemble_member][variable]

    def get_variables(self, ensemble_member: int = 0) -> Set:
        """
        Returns a set of variables for which timeseries values are stored in the internal data store

        :param ensemble_member: The ensemble member index.
        """
        if ensemble_member >= len(self.__timeseries_values):
            return set()
        return self.__timeseries_values[ensemble_member].keys()

    def get_ensemble_size(self):
        """
        Returns the number of ensemble members for which timeseries are stored in the internal data store
        """
        return len(self.__timeseries_values)

    def get_forecast_index(self) -> int:
        """"
        Looks up the forecast index from the internal data store

        :return: Current forecast index, values before this index will be considered "history".
        """
        return self.__forecast_index

    def set_forecast_index(self, forecast_index: int) -> None:
        """
        Sets the forecast index in the internal data store.
        Values (and times) before this index will be considered "history"

        :param forecast_index: New forecast index.
        """
        self.__forecast_index = forecast_index

    def set_parameter(self,
                      parameter_name: str,
                      value: float,
                      ensemble_member: int = 0,
                      check_duplicates: bool = True) -> None:
        """
        Stores the parameter value in the internal data store.

        :param parameter_name:   Parameter name.
        :param value:            The values to be stored.
        :param ensemble_member:  The ensemble member index.
        :param check_duplicates: If True, a warning will be given when attempting to overwrite values.
                                 If False, existing values can be silently overwritten with new values.
        """
        while ensemble_member >= len(self.__parameters):
            self.__parameters.append(AliasDict(self.alias_relation))

        if check_duplicates and parameter_name in self.__parameters[ensemble_member].keys():
            logger.warning("Attempting to set parameter value for ensemble member {} and name {} twice. "
                           "Ignoring second set of values.".format(ensemble_member, parameter_name))
            return

        self.__parameters[ensemble_member][parameter_name] = value

    def get_parameter(self, parameter_name: str, ensemble_member: int = 0) -> float:
        """
        Looks up the parameter value in the internal data store.
        """
        if ensemble_member >= len(self.__parameters):
            raise KeyError("ensemble_member {} does not exist".format(ensemble_member))
        return self.__parameters[ensemble_member][parameter_name]

    def get_parameter_names(self, ensemble_member: int = 0) -> Set:
        """
        Returns a set of variables for which timeseries values are stored in the internal data store

        :param ensemble_member: The ensemble member index.
        """
        if ensemble_member >= len(self.__parameters):
            return set()
        return self.__parameters[ensemble_member].keys()

    @property
    def initial_time(self) -> float:
        """
        The initial time in seconds.
        """
        if self.__timeseries_times_sec is None:
            raise RuntimeError("Attempting to access initial_time before setting times")
        return self.__timeseries_times_sec[self.__forecast_index]

    @staticmethod
    def datetime_to_sec(d: Union[Iterable[datetime], datetime], t0: datetime) -> Union[Iterable[float], float]:
        """
        Returns the date/timestamps in seconds since t0.

        :param d:  Iterable of datetimes or a single datetime object.
        :param t0: Reference datetime.
        """
        if hasattr(d, '__iter__'):
            return np.array([(t - t0).total_seconds() for t in d])
        else:
            return (d - t0).total_seconds()

    @staticmethod
    def sec_to_datetime(s: Union[Iterable[float], float], t0: datetime) -> Union[Iterable[datetime], datetime]:
        """
        Return the date/timestamps in seconds since t0 as datetime objects.

        :param s:  Iterable of ints or a single int (number of seconds before or after t0).
        :param t0: Reference datetime.
        """
        if hasattr(s, '__iter__'):
            return [t0 + timedelta(seconds=t) for t in s]
        else:
            return t0 + timedelta(seconds=s)

    @property
    @abstractmethod
    def alias_relation(self) -> AliasRelation:
        raise NotImplementedError
