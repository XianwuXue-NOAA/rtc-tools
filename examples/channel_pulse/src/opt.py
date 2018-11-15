from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.util import run_optimization_problem


class Example(
    HomotopyMixin, CSVMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem
):
    def constraints(self, ensemble_member):
        c = super().constraints(ensemble_member)
        times = self.times()
        parameters = self.parameters(ensemble_member)
        # Mimic HEC-RAS behaviour:  Enforce steady state both at t0 and at t1.
        for i in range(int(parameters["Channel.n_level_nodes"])):
            state = f"Channel.H[{i + 1}]"
            c.append(
                (self.state_at(state, times[0]) - self.state_at(state, times[1]), 0, 0)
            )
        return c


run_optimization_problem(Example)
