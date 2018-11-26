from rtctools.optimization.optimization_problem import OptimizationProblem


class SteadyStateInitializationMixin(OptimizationProblem):
    def constraints(self, ensemble_member):
        c = super().constraints(ensemble_member)
        times = self.times()
        parameters = self.parameters(ensemble_member)
        # Force steady-state initialization at t0 and at t1.
        # TODO move to initial equation section.
        n_level_nodes = 21
        for reach in ['upstream', 'middle', 'downstream']:
            for i in range(n_level_nodes + 1):
                state = f'{reach}.Q[{i + 1}]'
                c.append(
                    (self.der_at(state, times[0]), 0, 0)
                )
            for i in range(n_level_nodes):
                state = f'{reach}.H[{i + 1}]'
                c.append(
                    (self.der_at(state, times[0]), 0, 0)
                )
        return c
