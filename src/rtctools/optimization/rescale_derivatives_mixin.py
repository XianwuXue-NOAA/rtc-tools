import logging
import math

import casadi as ca

from .optimization_problem import OptimizationProblem

logger = logging.getLogger("rtctools")


class RescaleDerivativesMixin(OptimizationProblem):

    def __init__(self, *args, **kwargs):
        self.__dae_variables = None

        self.__rescaled_nominals = {}
        self.__rescaled_states_map = {}
        self.__rescaled_states_constraints = {}

        super().__init__(*args, **kwargs)

    def path_constraints(self, ensemble_member):
        constraints = super().path_constraints(ensemble_member).copy()
        constraints.extend(self.__rescaled_states_constraints.values())
        return constraints

    def times(self, variable=None):
        try:
            return self.times(self.__rescaled_states_map[variable])
        except KeyError:
            return super().times(variable)

    def transcribe(self):
        dae_states = []
        dae_derivatives = []

        for variable, derivative in zip(super().dae_variables['states'], super().dae_variables['derivatives']):
            variable_name = variable.name()

            times = self.times(variable_name)
            dt = times[1] - times[0]

            src_nominal = self.variable_nominal(variable_name) / dt
            dst_nominal = self.derivative_nominal(variable_name)

            frac = src_nominal / dst_nominal
            if frac > 100.0:
                order_steps = math.ceil(math.log10(frac))
                assert order_steps >= 1.0

                # We move up to 1E4 to a differently scaled alias for the
                # state. This is because solvers generally care more about the
                # coefficients being [1E-2, 1E2], but are OK with state vector
                # entries up to 1E4.
                large_state_fac_exp = min(4, order_steps)
                large_state_fac = 10**large_state_fac_exp
                large_state_name = "__{}_order_1E{}".format(variable_name, large_state_fac_exp)
                large_state = ca.MX.sym(large_state_name)
                self.__rescaled_states_map[large_state_name] = variable_name

                orig_state_nominal = self.variable_nominal(variable_name)
                large_state_nominal = orig_state_nominal / large_state_fac
                self.__rescaled_nominals[large_state_name] = large_state_nominal

                # We want to end up with a well-scaled equation. So we divide
                # out the nominal (that would otherwise appear) and keep the
                # coefficients around 1. For example, a large_state_fac_exp of 4
                # would result in a factor of 100 and 0.01. A large_state_fac_exp
                # of 3 would result in 100 and 0.1.
                normal_state_no_nominal = variable / orig_state_nominal
                large_state_no_nominal = large_state / large_state_nominal

                balance_fac = 10**max(math.floor(large_state_fac_exp / 2), 1.0)
                self.__rescaled_states_constraints[variable] = ((
                    large_state_fac * normal_state_no_nominal - large_state_no_nominal) / balance_fac,
                0.0, 0.0)

                dae_states.append(variable)
                dae_derivatives.append(ca.MX.sym("__dummy_der({})".format(variable_name)))

                dae_states.append(large_state)
                dae_derivatives.append(derivative)
            else:
                dae_states.append(variable)
                dae_derivatives.append(derivative)

        self.__dae_variables = super().dae_variables.copy()
        self.__dae_variables['states'] = dae_states
        self.__dae_variables['derivatives'] = dae_derivatives

        return super().transcribe()

    @property
    def dae_variables(self):
        if self.__dae_variables is not None:
            return self.__dae_variables
        else:
            return super().dae_variables

    def bounds(self):
        bounds = super().bounds()

        for rescaled_variable, orig_variable in self.__rescaled_states_map.items():
            bounds[rescaled_variable] = bounds[orig_variable]

        return bounds

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)

        for rescaled_variable, orig_variable in self.__rescaled_states_map.items():
            seed[rescaled_variable] = seed[orig_variable]

        return seed

    def variable_nominal(self, variable):
        try:
            return self.__rescaled_nominals[variable]
        except KeyError:
            return super().variable_nominal(variable)
