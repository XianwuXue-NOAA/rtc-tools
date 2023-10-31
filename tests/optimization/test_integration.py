import logging

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin

from test_case import TestCase

from .data_path import data_path

logger = logging.getLogger("rtctools")


class SingleShootingBaseModel(ModelicaMixin, CollocatedIntegratedOptimizationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="HybridShootingModel",
            model_folder=data_path(),
        )

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    @property
    def integrate_states(self):
        return True

    def pre(self):
        # Do nothing
        pass

    def bounds(self):
        # Variable bounds
        return {"u": (-2.0, 2.0)}

    def seed(self, ensemble_member):
        # No particular seeding
        return {}

    def constraints(self, ensemble_member):
        # No additional constraints
        return []

    def post(self):
        # Do
        pass

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options["library_folders"] = []
        return compiler_options


class SingleShootingModel(SingleShootingBaseModel):
    def objective(self, ensemble_member):
        # Quadratic penalty on state 'x' at final time
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        return xf**2


class TestSingleShooting(TestCase):
    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)

    def setUp(self):
        self.problem = SingleShootingModel()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

class SingleShootingExtraVariablesModel(SingleShootingBaseModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._extra_var = ca.MX.sym("x_abs")

        self._path_var_a = ca.MX.sym("a_path")
        self._path_var_x = ca.MX.sym("x_path")

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member).copy()

        # Make an absolute version of the value of x at the final timestep
        xf = self.state_at("x", self.times("x")[-1], ensemble_member=ensemble_member)
        x_abs = self.extra_variable("x_abs", ensemble_member=ensemble_member)

        constraints.append((x_abs - xf, 0.0, np.inf))
        constraints.append((x_abs + xf, 0.0, np.inf))

        return constraints

    def objective(self, ensemble_member):
        # Minizize the absolute value of x at the final timestep
        return self.extra_variable("x_abs", ensemble_member=ensemble_member)

    def path_constraints(self, ensemble_member: int):
        constraints = super().path_constraints(ensemble_member).copy()

        # Add constraints between path variables and states/algebraic states
        constraints.append((self._path_var_x - self.state("x"), 0.0, 0.0))
        constraints.append((self._path_var_a - self.state("a"), 0.0, 0.0))

        return constraints

    @property
    def path_variables(self):
        return [self._path_var_a, self._path_var_x]

    @property
    def extra_variables(self):
        return [self._extra_var]


class TestSingleShootingExtraVariables(TestCase):
    def test_objective_value(self):
        objective_value_tol = 1e-6
        self.assertAlmostLessThan(abs(self.problem.objective_value), 0.0, objective_value_tol)

    def setUp(self):
        self.problem = SingleShootingExtraVariablesModel()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6


class SingleShootingGoalProgrammingModel(GoalProgrammingMixin, SingleShootingBaseModel):
    def goals(self):
        return [Goal1(), Goal2(), Goal3()]

    def path_goals(self):
        return [PathGoal1()]

    def set_timeseries(self, timeseries_id, timeseries, ensemble_member, **kwargs):
        # Do nothing
        pass


class PathGoal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state("x")

    function_range = (-1e1, 1e1)
    priority = 1
    target_min = -0.9e1
    target_max = 0.9e1


class Goal1(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.5, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 3
    target_min = 0.0


class Goal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at("x", 0.7, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 3
    target_min = 0.1


class Goal3(Goal):
    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral("x", 0.1, 1.0, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_max = 1.0


class TestGoalProgramming(TestCase):
    def setUp(self):
        self.problem = SingleShootingGoalProgrammingModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        objective_value_tol = 1e-6
        self.assertAlmostGreaterThan(
            self.problem.interpolate(
                0.7, self.problem.times(), self.problem.extract_results()["x"]
            ),
            0.1,
            objective_value_tol,
        )
