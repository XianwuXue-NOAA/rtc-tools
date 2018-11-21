import bisect
import logging
from abc import ABCMeta, abstractmethod

import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.caching import cached
from rtctools.simulation.simulation_problem import SimulationProblem

logger = logging.getLogger("rtctools")


class IOMixin(SimulationProblem, metaclass=ABCMeta):
    """
    Base class for all IO methods of optimization problems.
    """

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

    def pre(self) -> None:
        # Call read method to read all input
        self.read()

    @abstractmethod
    def read(self) -> None:
        """
        Reads input data from files, storing it in the internal data store through the various set or add methods
        """
        pass

    def post(self) -> None:
        # Call write method to write all output
        self.write()

    @abstractmethod
    def write(self) -> None:
        """"
        Writes output data to files, getting the data from the data store through the various get methods
        """
        pass

    def initialize(self, config_file=None):
        # Set up experiment
        timeseries_import_times = self.get_times()
        self.__dt = timeseries_import_times[1] - timeseries_import_times[0]
        self.setup_experiment(0, timeseries_import_times[-1], self.__dt)

        parameter_variables = set(self.get_parameter_variables())

        logger.debug("Model parameters are {}".format(parameter_variables))

        for parameter in self.get_parameter_names():
            if parameter in parameter_variables:
                value = self.get_parameter(parameter)
                logger.debug("IOMixin: Setting parameter {} = {}".format(parameter, value))
                self.set_var(parameter, value)

        # Load input variable names
        self.__input_variables = set(self.get_input_variables().keys())

        # Set input values
        self.__set_input_variables(self.get_forecast_index())

        logger.debug("Model inputs are {}".format(self.__input_variables))

        # Empty output
        self.__output_variables = self.get_output_variables()
        n_times = len(self.get_times())
        self.__output = AliasDict(self.alias_relation)
        self.__output.update({variable: np.full(n_times, np.nan) for variable in self.__output_variables})

        # Call super, which will also initialize the model itself
        super().initialize(config_file)

        # Extract consistent t0 values
        for variable in self.__output_variables:
            self.__output[variable][self.get_forecast_index()] = self.get_var(variable)

    def __set_input_variables(self, t_idx):
        for variable in self.get_variables():
            if variable in self.__input_variables:
                value = self.get_timeseries_values(variable)[t_idx]
                if np.isfinite(value):
                    self.set_var(variable, value)
                else:
                    logger.debug("IOMixin: Found bad value {} at index [{}] in timeseries aliased to input {}"
                                 .format(value, t_idx, variable))

    def update(self, dt):
        # Time step
        if dt < 0:
            dt = self.__dt

        # Current time stamp
        t = self.get_current_time()

        # Get current time index
        t_idx = bisect.bisect_left(self.get_times(), t + dt)

        # Set input values
        self.__set_input_variables(t_idx)

        # Call super
        super().update(dt)

        # Extract results
        for variable in self.__output_variables:
            self.__output[variable][t_idx] = self.get_var(variable)

    @property
    def output_variables(self):
        return self.__output_variables

    @property
    def output(self):
        return self.__output

    @cached
    def parameters(self):
        """
        Return a dictionary of parameters, including parameters in the input files files.

        :returns: Dictionary of parameters
        """
        # Call parent class first for default values.
        parameters = super().parameters()

        # Load parameters from input files (stored in internal data store)
        for parameter_name in self.get_parameter_names():
            parameters[parameter_name] = self.get_parameter(parameter_name)

        if logger.getEffectiveLevel() == logging.DEBUG:
            for parameter_name in self.get_parameter_names():
                logger.debug("IOMixin: Read parameter {}".format(parameter_name))

        return parameters

    def times(self, variable=None):
        """
        Return a list of all the timesteps in seconds.

        :param variable: Variable name.

        :returns: List of all the timesteps in seconds.
        """
        return self.get_times()[self.get_forecast_index():]

    def timeseries_at(self, variable, t):
        """
        Return the value of a time series at the given time.

        :param variable: Variable name.
        :param t: Time.

        :returns: The interpolated value of the time series.

        :raises: KeyError
        """
        values = self.get_timeseries_values(variable)
        timeseries_times_sec = self.get_times()
        t_idx = bisect.bisect_left(timeseries_times_sec, t)
        if timeseries_times_sec[t_idx] == t:
            return values[t_idx]
        else:
            return np.interp(t, timeseries_times_sec, values)
