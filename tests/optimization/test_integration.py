import logging

import casadi as ca

import numpy as np

from rtctools.optimization.collocated_integrated_optimization_problem import (
    CollocatedIntegratedOptimizationProblem,
)
from rtctools.optimization.control_tree_mixin import ControlTreeMixin
from rtctools.optimization.goal_programming_mixin import Goal, GoalProgrammingMixin
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries

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


class EnsembleGoal2(Goal):
    def function(self, optimization_problem, ensemble_member):
        if ensemble_member == 0:
            time = 0.7
        elif ensemble_member == 1:
            time = 0.8
        else:
            raise Exception("Invalid ensemble member")

        return optimization_problem.state_at("x", time, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 3
    target_min = 0.1


class SingleShootingGoalProgrammingEnsembleControlTreeModel(GoalProgrammingMixin, ControlTreeMixin, SingleShootingBaseModel):
    ensemble_size = 2

    def goals(self):
        return [Goal1(), EnsembleGoal2(), Goal3()]

    def control_tree_options(self):
        options = super().control_tree_options()
        options["forecast_variables"] = ["branching_input"]
        options["branching_times"] = [0.2]
        options["k"] = 2
        return options

    def constant_inputs(self, ensemble_member: int):
        inputs = super().constant_inputs(ensemble_member)

        times = self.times()
        values = np.ones_like(times) * ensemble_member
        inputs["branching_input"] = Timeseries(times, values)

        return inputs


class TestGoalProgrammingEnsembleControlTree(TestCase):
    def setUp(self):
        self.problem = SingleShootingGoalProgrammingEnsembleControlTreeModel()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_unequal_control_and_state(self):
        results_0 = self.problem.extract_results(0)
        results_1 = self.problem.extract_results(1)

        self.assertFalse(np.array_equal(results_0["u"], results_1["u"]))
        self.assertFalse(np.array_equal(results_0["x"], results_1["x"]))
