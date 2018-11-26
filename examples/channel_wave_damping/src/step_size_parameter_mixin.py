import numpy as np

from rtctools.optimization.optimization_problem import OptimizationProblem


class StepSizeParameterMixin(OptimizationProblem):
    step_size = 15 * 60  # 15 minutes

    def times(self, variable=None):
        times = super().times(variable)
        return np.arange(times[0], times[-1], self.step_size)

    def parameters(self, ensemble_member):
        p = super().parameters(ensemble_member)
        p['step_size'] = self.step_size
        return p

    def solver_options(self):
        o = super().solver_options()
        o['expand'] = True
        o['ipopt.linear_solver'] = 'ma86'
        return o

    def compiler_options(self):
        o = super().compiler_options()
        o['replace_parameter_values'] = True
        return o
