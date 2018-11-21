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

    def __init__(self, state, target_level, margin):
        self.function_range = target_level - 10, target_level + 10
        self.function_nominal = target_level
        self.target_min = target_level - margin / 2
        self.target_max = target_level + margin / 2
        self.state = state

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)

    priority = 1


class ExampleHybrid(
    StepSizeParameterMixin,
    SteadyStateInitializationMixin,
    HomotopyMixin,
    GoalProgrammingMixin,
    CSVMixin,
    ModelicaMixin,
    CollocatedIntegratedOptimizationProblem,
):
    """Goal Programming Approach"""

    def path_goals(self):
        # Add water level goals
        goals = [
            TargetLevelGoal("dam_upstream.HQUp.H", 20.0, 0),
        ]
            
        return goals

    def solver_options(self):
        options = super().solver_options()
        options['ipopt.linear_solver'] = 'ma86'
        options['ipopt.nlp_scaling_method'] = 'none'
        options['ipopt.jac_c_constant'] = 'no'
        options['ipopt.jac_d_constant'] = 'yes'
        options['expand'] = True #TODO
        return options

    def goal_programming_options(self):
        options = super().goal_programming_options()
        options['scale_by_problem_size'] = False
        options['keep_soft_constraints'] = False
        return options


# Run
run_optimization_problem(ExampleHybrid)
#run_optimization_problem(ExampleLocalControl)
