import logging

from rtctools.simulation.csv_mixin import CSVMixin
from rtctools.simulation.simulation_problem import SimulationProblem
from rtctools.util import run_simulation_problem

logger = logging.getLogger("rtctools")


class SimpleModel(CSVMixin, SimulationProblem):
    """
    Simple model class to illustrate customized variable updates.

    This class illustrates how to update a variable
    using a custom function defined in python instead of Modelica.

    The model of this class is given by

    .. math::
        \frac{dx}{dt} &= y, \\
        y &= -\frac{2}{3600}x.

    The equation for :math:`x` is given in Modelica,
    while the equatin for :math:`y` is implemented in this class.
    """

    def __init__(self, **kwargs):
        super().__init__(model_name="SimpleModel", **kwargs)

    def get_y(self):
        """
        Function for computing y.

        This is implemented here instead of in the Modelica file,
        to illustrate how simulation models can be partially implemented in python.
        """
        seconds_per_hour = 3600
        return -2 / seconds_per_hour * self.get_var('x')

    def initialize(self):
        """
        Overwrite the pre method to set the initial value of y.

        Input variables that are not read from a file should be set with a numeric value
        before calling super().initialize().
        """
        dummy_value = 0.0
        self.set_var('y', dummy_value)
        super().initialize()
        self.set_var('y', self.get_y())

    def update(self, dt):
        """
        Overwrite the update method to manually update y.
        """
        super().update(dt)  # Update x using still the previous value of y.
        self.set_var('y', self.get_y())  # Update y using the new value of x.


# Run
run_simulation_problem(SimpleModel, log_level=logging.DEBUG)
