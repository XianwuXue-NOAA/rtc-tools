import logging

import casadi as ca

from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.simulation.simulation_problem import CustomResidual, SimulationProblem
from rtctools.util import run_simulation_problem

logger = logging.getLogger("rtctools")


class SimpleModel(CSVMixin, SimulationProblem):
    """
    Simple model class to illustrate implementing a custom equation.

    This class illustrates how to implement a custom equation in python
    instead of Modelica.

    The model of this class is given by

    .. math::
        \frac{dx}{dt} &= y, \\
        y &= -\frac{2}{3600}x.

    The equation for :math:`x` is given in Modelica,
    while the equatin for :math:`y` is implemented in this class.
    """

    def __init__(self, **kwargs):
        self.timeseries_export_basename = "timeseries_export"
        custom_states = ["y"]
        custom_residuals = self.custom_residuals()
        super().__init__(
            model_name="SimpleModel",
            custom_states=custom_states,
            custom_residuals=custom_residuals,
            **kwargs,
        )

    def custom_residuals(self):
        """
        Define residuals for equations that are not in the Modelica file.

        In particular, define a residual for y = -2/3600 * x.
        """
        x = ca.MX.sym("x")
        y = ca.MX.sym("y")
        r_fun = ca.Function("r_fun", [x, y], [y - (-2 / 3600 * x)])
        residual = CustomResidual(r_fun, ["x", "y"])
        return [residual]


class SimpleModelLessStable(CSVMixin, SimulationProblem):
    """
    Simple model class to illustrate implementing a custom equation.

    This class is similer to SimpleModel, but illustrates an alternative way
    of implementing a custom equation that is a bit more flexible
    but also less reliable.

    It is more flexible in a sense that it enables implementing equations
    that can not be described with casadi symbol.

    The results are less reliable, since the numerical method is no
    longer fully implicit and may cause numerical instabilities
    and artificial delay.
    """

    def __init__(self, **kwargs):
        self.timeseries_export_basename = "timeseries_export_less_stable"
        super().__init__(model_name="SimpleModel", **kwargs)

    def get_y(self):
        """
        Function for computing y.

        This is implemented here instead of in the Modelica file,
        to illustrate how simulation models can be partially implemented in python.
        """
        seconds_per_hour = 3600
        return -2 / seconds_per_hour * self.get_var("x")

    def initialize(self):
        """
        Overwrite the pre method to set the initial value of y.

        Input variables that are not read from a file should be set with a numeric value
        before calling super().initialize().
        """
        dummy_value = 0.0
        self.set_var("y", dummy_value)
        super().initialize()

    def update(self, dt):
        """
        Overwrite the update method to manually update y.
        """
        self.set_var("y", self.get_y())  # Update new value of y using current value of x.
        super().update(dt)  # Update x.


# Run
run_simulation_problem(SimpleModel, log_level=logging.DEBUG)
run_simulation_problem(SimpleModelLessStable, log_level=logging.DEBUG)
