import numpy as np
import casadi as ca

from example_local_control import ExampleLocalControl

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.csv_mixin import CSVMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.homotopy_mixin import HomotopyMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.util import run_optimization_problem

from steady_state_initialization_mixin import SteadyStateInitializationMixin

from step_size_parameter_mixin import StepSizeParameterMixin


class TargetLevelGoal(Goal):
    """Really Simple Target Level Goal"""

    def __init__(self, state, target_level, margin, priority):
        self.function_range = target_level - 10, target_level + 10
        self.function_nominal = 10
        self.target_min = target_level - margin / 2
        self.target_max = target_level + margin / 2
        self.state = state
        self.priority = priority

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)

    order = 2


class TargetMaximumDischargeGoal(Goal):
    """Really Simple Target Level Goal"""

    def __init__(self, state, target_level, margin, priority):
        self.function_range = 0, 1000
        self.function_nominal = 100
        self.target_max = target_level + margin / 2
        self.state = state
        self.priority = priority

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)

    order = 2


class SmoothingGoal(Goal):
    """Smoothing Goal"""

    def __init__(self, state, nominal, priority):
        self.function_nominal = nominal
        self.state = state
        self.priority = priority

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.der(self.state)

    order = 2


class MinAmplitudeGoal(Goal):
    """Amplitude Minimization Goal"""

    def __init__(self, variable, priority):
        self.function_nominal = 100.0
        self.variable = variable
        self.priority = priority

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.extra_variable(self.variable)

    order = 2


class ExampleOptimization(
    StepSizeParameterMixin,
    SteadyStateInitializationMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    """Goal Programming Approach"""

    def pre(self):
        super().pre()

        self._a_u = ca.MX.sym('a_u')
        self._a_m = ca.MX.sym('a_m')

    @property
    def extra_variables(self):
        return super().extra_variables #+ [self._a_u, self._a_m]

    def bounds(self):
        bounds = super().bounds()
        bounds['a_u'] = (0.0, 1000.0)
        bounds['a_m'] = (0.0, 1000.0)
        return bounds

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)
        #constant_inputs['Inflow_Q'].values.fill(500.0)
        #constant_inputs['Inflow_Q'].values[0] = 100.0
        #constant_inputs['Inflow_Q'].values.fill(100.0)
        #constant_inputs['Inflow_Q'].values[0] = 500.0
        return constant_inputs

    def variable_nominal(self, variable):
        if variable in set(['a_u', 'a_m']):
            return 100.0
        else:
            return super().variable_nominal(variable)

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)
        #path_constraints.append((self.state("dam_upstream.HQUp.Q") - self._a_u, -np.inf, 0.0))
        #path_constraints.append((self.state("dam_middle.HQUp.Q") - self._a_m, -np.inf, 0.0))
        return path_constraints

    def goals(self):
        goals = [
            #MinAmplitudeGoal('a_u', 2),
            #MinAmplitudeGoal('a_m', 2),
        ]

        return goals

    def path_goals(self):
        path_goals = [
            #TargetLevelGoal("dam_upstream.HQUp.H", 20.0, 1.0, 1),
            #TargetLevelGoal("dam_middle.HQUp.H", 15.0, 1.0, 1),
            TargetLevelGoal("dam_upstream.HQUp.H", 20.0, 0.0, 2),
            TargetLevelGoal("dam_middle.HQUp.H", 15.0, 0.0, 2),
            #TargetMaximumDischargeGoal("dam_upstream.HQUp.Q", 100, 0.0, 2),
            #TargetMaximumDischargeGoal("dam_middle.HQUp.Q", 100, 0.0, 2),
            #SmoothingGoal("dam_upstream.HQUp.Q", 1e-2, 4),
            #SmoothingGoal("dam_middle.HQUp.Q", 1e-2, 4),
        ]

        # TODO idea:  spmile cutoff goal

        return path_goals

    def solver_options(self):
        options = super().solver_options()
        options['ipopt.linear_solver'] = 'ma86'
        options['ipopt.nlp_scaling_method'] = 'none'
        options['ipopt.jac_c_constant'] = 'no'
        options['ipopt.jac_d_constant'] = 'yes'
        options['expand'] = True # TODO port to other examples.
        return options

    def goal_programming_options(self):
        options = super().goal_programming_options()
        options['keep_soft_constraints'] = True
        options['scale_by_problem_size'] = False
        return options


# Run
run_optimization_problem(ExampleOptimization)
#run_optimization_problem(ExampleLocalControl)
