# cython: embedsignature=True

from rtctools.optimization.optimization_problem import OptimizationProblem
import logging

logger = logging.getLogger("rtctools")


class LinearizationMixin(OptimizationProblem):
    """
    Adds linearized equation parameter bookkeeping to your optimization aproblem.  If your model contains 
    linearized equations, this mixin will set the parameters of these equations based on the t0 value of an associated
    timeseries.

    The mapping between linearization parameters and time series is provided in the ``linearization_parameters`` method.
    """

    def parameters(self, ensemble_member):
        parameters = super(LinearizationMixin, self).parameters(ensemble_member)

        for parameter, timeseries_id in self.linearization_parameters().iteritems():
            parameters[parameter] = self.timeseries_at(timeseries_id, self.initial_time, ensemble_member)

        return parameters

    def linearization_parameters(self):
        """
        :returns: A dictionary of parameter names mapping to time series identifiers.
        """

        return {}
