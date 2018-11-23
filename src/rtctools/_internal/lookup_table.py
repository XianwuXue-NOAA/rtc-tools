from typing import Iterable, List, Tuple, Union

import casadi as ca

import numpy as np

from rtctools._internal.caching import cached
from rtctools.optimization.timeseries import Timeseries

from scipy.optimize import brentq


class LookupTable:
    """A splined relationship between input and output variables
    """

    def __init__(self, inputs: List[ca.MX], function: ca.Function, tck: Tuple = None):
        """Create a new lookup table object.

        :param inputs:
            List of lookup table input variables.
        :param function:
            Lookup table CasADi :class:`Function`.
        """
        self.__inputs = inputs
        self.__function = function

        self.__t, self.__c, self.__k = [None] * 3

        if tck is not None:
            if len(tck) == 3:
                self.__t, self.__c, self.__k = tck
            elif len(tck) == 5:
                self.__t = tck[:2]
                self.__c = tck[2]
                self.__k = tck[3:]

    @property
    def domain(self) -> Tuple[float, float]:
        """Minimum and maximum of x (input) covered by the LookupTable"""
        return self.__get_domain()

    @cached
    def __get_domain(self) -> Tuple[float, float]:
        t = self.__t
        if t is None:
            raise AttributeError(
                "This lookup table was not instantiated with tck metadata. "
                "Domain/Range information is unavailable."
            )
        if type(t) == tuple and len(t) == 2:
            raise NotImplementedError(
                "Domain/Range information is not yet implemented for 2D LookupTables"
            )
        return np.nextafter(t[0], np.inf), np.nextafter(t[-1], -np.inf)

    @property
    def range(self) -> Tuple[float, float]:
        """Minimum and maximum of y (output) covered by the LookupTable"""
        return self.__get_range()

    @cached
    def __get_range(self) -> Tuple[float, float]:
        return self(self.domain[0]), self(self.domain[1])

    @property
    def inputs(self) -> List[ca.MX]:
        """List of lookup table input variables."""
        return self.__inputs

    @property
    def function(self) -> ca.Function:
        """Lookup table CasADi :class:`Function`.

        Example use::

            lookup_table.function(self.state("x"))

        """
        return self.__function

    @property
    @cached
    def __numeric_function_evaluator(self):
        return np.vectorize(
            lambda *args: np.nan
            if np.any(np.isnan(args))
            else np.float(self.function(*args))
        )

    def __call__(
        self, *args: Union[float, Iterable, Timeseries]
    ) -> Union[float, np.ndarray, Timeseries]:
        """
        Evaluate the lookup table.

        :param args:
            Input values.
        :returns:
            Lookup table evaluated at input values.

        Example use::

            y = lookup_table(1.0)
            [y1, y2] = lookup_table([1.0, 2.0])
            y_timeseries = lookup_table(Timeseries([1.0, 2.0], [1.0, 2.0]))

        """
        evaluator = self.__numeric_function_evaluator
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, Timeseries):
                return Timeseries(arg.times, self(arg.values))
            else:
                if hasattr(arg, "__iter__"):
                    arg = np.fromiter(arg, dtype=float)
                    return evaluator(arg)
                else:
                    arg = float(arg)
                    return evaluator(arg).item()
        else:
            if any(isinstance(arg, Timeseries) for arg in args):
                raise TypeError(
                    "Higher-order LookupTable calls do not yet support "
                    "Timeseries parameters"
                )
            elif any(hasattr(arg, "__iter__") for arg in args):
                raise TypeError(
                    "Higher-order LookupTable calls do not yet support "
                    "vector parameters"
                )
            else:
                args = np.fromiter(args, dtype=float)
                return evaluator(*args)

    def reverse_call(
        self,
        y: Union[float, Iterable, Timeseries],
        domain: Tuple[float, float] = (None, None),
        detect_range_error: bool = True,
    ) -> Union[float, np.ndarray, Timeseries]:
        """Do an inverted call on this LookupTable

        Uses SciPy brentq optimizer to simulate a reversed call.
        Note: Method does not work with higher-order LookupTables

        :param y:
            Input values.
        :param domain:
            Min and max of the output, default is the whole domain of the LookupTable
        :param detect_range_error:
            Whether to raise ValueError if the input values are out of the table range
        :returns:
            Lookup table inversely evaluated at input values.
        :raises:
            ValueError

        Example use::

            x = lookup_table.reverse_call(1.0)
            [x1, x2] = lookup_table.reverse_call([1.0, 2.0])
            x_timeseries = lookup_table.reverse_call(Timeseries([1.0, 2.0], [1.0, 2.0]))

        """
        if isinstance(y, Timeseries):
            # Recurse and return
            return Timeseries(y.times, self.reverse_call(y.values))

        # Get domain information
        l_d, u_d = domain
        if l_d is None:
            l_d = self.domain[0]
        if u_d is None:
            u_d = self.domain[1]

        # Cast y to array of float
        if hasattr(y, "__iter__"):
            y_array = np.fromiter(y, dtype=float)
        else:
            y_array = np.array([y], dtype=float)

        # Find not np.nan
        is_not_nan = ~np.isnan(y_array)
        y_array_not_nan = y_array[is_not_nan]

        # Detect if there is a range violation
        if detect_range_error:
            l_r, u_r = self.range
            lb_viol = y_array_not_nan < l_r
            ub_viol = y_array_not_nan > u_r
            all_viol = y_array_not_nan[lb_viol | ub_viol]
            if all_viol:
                raise ValueError(
                    "Values {} are not in lookup table range ({}, {})".format(
                        all_viol, l_r, u_r
                    )
                )

        # Construct function to do inverse evaluation
        evaluator = self.__numeric_function_evaluator

        def inv_evaluator(y_target):
            """inverse evaluator function"""
            return brentq(lambda x: evaluator(x) - y_target, l_d, u_d)

        inv_evaluator = np.vectorize(inv_evaluator)

        # Calculate x_array
        x_array = np.full_like(y_array, np.nan, dtype=float)
        if y_array_not_nan.size != 0:
            x_array[is_not_nan] = inv_evaluator(y_array_not_nan)

        # Return x
        if hasattr(y, "__iter__"):
            return x_array
        else:
            return x_array.item()
