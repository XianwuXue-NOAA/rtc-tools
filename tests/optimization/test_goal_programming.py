from data_path import data_path
from test_case import TestCase

from rtctools.optimization.optimization_problem import OptimizationProblem
from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
from rtctools.optimization.goal_programming_mixin import GoalProgrammingMixin, Goal
from rtctools.optimization.modelica_mixin import ModelicaMixin
from rtctools.optimization.timeseries import Timeseries
from casadi import MX, MXFunction
import numpy as np
import logging
import time
import sys
import os

logger = logging.getLogger("rtctools")
logger.setLevel(logging.WARNING)


class TestProblem(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblem, self).__init__(input_folder=data_path(), output_folder=data_path(
        ), model_name='TestModelWithInitial', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def delayed_feedback(self):
        # Delayed feedback
        return [('x', 'x_delayed', 0.1)]

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        return {'constant_input': Timeseries(np.hstack(([self.initial_time, self.times()])), np.hstack(([1.0], np.linspace(1.0, 0.0, 21))))}

    def bounds(self):
        bounds = super(TestProblem, self).bounds()
        bounds['u'] = (-2.0, 2.0)
        return bounds

    def goals(self):
        return [TestGoal1(), TestGoal2(), TestGoal3()]


class TestGoal1(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at('x', 0.5, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.0


class TestGoal2(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at('x', 0.7, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.1


class TestGoal3(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral('x', 0.1, 1.0, ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 1
    target_max = 1.0


class TestGoalProgramming(TestCase):

    def setUp(self):
        self.problem = TestProblem()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        objective_value_tol = 1e-6
        self.assertAlmostGreaterThan(self.problem.interpolate(0.7, self.problem.times(
        ), self.problem.extract_results()['x']), 0.1, objective_value_tol)


class TestGoalNoMinMax(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral('x', ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 1
    order = 1


class TestGoalLowMax(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral('x', ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 1
    order = 1
    # TODO: Why this number? Is it a coincidence?
    target_max = function_range[0]


# Inherit from existing TestProblem, as all properties are equal except the
# goals.
class TestProblemNoMinMax(TestProblem):

    def goals(self):
        return [TestGoalNoMinMax()]


class TestProblemLowMax(TestProblem):

    def goals(self):
        return [TestGoalLowMax()]


class TestGoalProgrammingNoMinMax(TestCase):

    def setUp(self):
        self.problem1 = TestProblemNoMinMax()
        self.problem2 = TestProblemLowMax()
        self.problem1.optimize()
        self.problem2.optimize()
        self.tolerance = 1e-6

    def test_nobounds_equal_lowmax(self):
        self.assertAlmostEqual(sum(self.problem1.extract_results()['x']), sum(
            self.problem2.extract_results()['x']), self.tolerance)


class TestGoalMinimizeU(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at('u', 0.5, ensemble_member=ensemble_member)

    function_range = (-1e2, 1e2)
    priority = 1
    order = 1


class TestGoalMinimizeX(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state_at('x', 0.5, ensemble_member=ensemble_member)

    function_range = (-1e2, 1e2)
    priority = 2
    order = 1
    target_min = 2.0


class TestProblemMinimizeU(TestProblem):

    def goals(self):
        return [TestGoalMinimizeU()]


class TestProblemMinimizeUandX(TestProblem):

    def goals(self):
        return [TestGoalMinimizeU(), TestGoalMinimizeX()]


class TestGoalProgrammingHoldMinimization(TestCase):

    def setUp(self):
        self.problem1 = TestProblemMinimizeU()
        self.problem2 = TestProblemMinimizeUandX()
        self.problem1.optimize()
        self.problem2.optimize()
        self.tolerance = 1e-6

    def test_hold_minimization_goal(self):
        # Collocation point 0.5 is at index 10
        self.assertAlmostEqual(self.problem1.extract_results()['u'][
                               10], self.problem2.extract_results()['u'][10], self.tolerance)


class PathGoal1(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('x')

    function_range = (-1e1, 1e1)
    priority = 1
    target_min = 0.0


class PathGoal2(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('x')

    function_range = (-1e1, 1e1)
    priority = 2
    target_max = Timeseries(np.linspace(0.0, 1.0, 21), 21 * [1.0])


class PathGoal3(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('u')

    function_range = (-1e1, 1e1)
    priority = 3


class TestProblemPathGoals(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblemPathGoals, self).__init__(input_folder=data_path(
        ), output_folder=data_path(), model_name='TestModelWithInitial', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def delayed_feedback(self):
        # Delayed feedback
        return [('x', 'x_delayed', 0.1)]

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        return {'constant_input': Timeseries(np.hstack(([self.initial_time, self.times()])), np.hstack(([1.0], np.linspace(1.0, 0.0, 21))))}

    def bounds(self):
        bounds = super(TestProblemPathGoals, self).bounds()
        bounds['u'] = (-2.0, 2.0)
        return bounds

    def path_goals(self):
        return [PathGoal1(), PathGoal2(), PathGoal3()]


class TestGoalProgrammingPathGoals(TestCase):

    def setUp(self):
        self.problem = TestProblemPathGoals()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        value_tol = 1e-3
        for x in self.problem.extract_results()['x']:
            self.assertAlmostGreaterThan(x, 0.0, value_tol)
            self.assertAlmostLessThan(x, 1.1, value_tol)


class PathGoal1Reversed(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('x')

    function_range = (-1e1, 1e1)
    priority = 2
    target_min = 0.0


class PathGoal2Reversed(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state('x')

    function_range = (-1e1, 1e1)
    priority = 1
    target_max = Timeseries(np.linspace(0.0, 1.0, 21), 21 * [1.0])


class TestProblemPathGoalsReversed(TestProblemPathGoals):

    def path_goals(self):
        return [PathGoal1Reversed(), PathGoal2Reversed()]


class TestGoalProgrammingPathGoalsReversed(TestGoalProgrammingPathGoals):

    def setUp(self):
        self.problem = TestProblemPathGoalsReversed()
        self.problem.optimize()
        self.tolerance = 1e-6


class TestGoalMinU(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.integral('u', ensemble_member=ensemble_member)

    function_range = (-1e1, 1e1)
    priority = 3


class TestProblemPathGoalsMixed(TestProblemPathGoals):

    def path_goals(self):
        return [PathGoal1(), PathGoal2()]

    def goals(self):
        return [TestGoalMinU()]


class TestGoalProgrammingPathGoalsMixed(TestGoalProgrammingPathGoals):

    def setUp(self):
        self.problem = TestProblemPathGoalsMixed()
        self.problem.optimize()
        self.tolerance = 1e-6


class TestProblemEnsemble(TestProblem):

    @property
    def ensemble_size(self):
        return 2

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        if ensemble_member == 0:
            return {'constant_input': Timeseries(self.times(), np.linspace(1.0, 0.0, 21))}
        else:
            return {'constant_input': Timeseries(self.times(), np.linspace(1.0, 0.5, 21))}


class TestGoalProgrammingEnsemble(TestGoalProgramming):

    def setUp(self):
        self.problem = TestProblemEnsemble()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        objective_value_tol = 1e-6
        self.assertAlmostGreaterThan(self.problem.interpolate(0.7, self.problem.times(
        ), self.problem.extract_results(0)['x']), 0.1, objective_value_tol)
        self.assertAlmostGreaterThan(self.problem.interpolate(0.7, self.problem.times(
        ), self.problem.extract_results(1)['x']), 0.1, objective_value_tol)


class PathGoalSmoothing(Goal):

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.der('u')

    function_range = (-1e1, 1e1)
    priority = 3


class TestProblemPathGoalsSmoothing(GoalProgrammingMixin, ModelicaMixin, CollocatedIntegratedOptimizationProblem):

    def __init__(self):
        super(TestProblemPathGoalsSmoothing, self).__init__(input_folder=data_path(
        ), output_folder=data_path(), model_name='TestModelWithInitial', model_folder=data_path())

    def times(self, variable=None):
        # Collocation points
        return np.linspace(0.0, 1.0, 21)

    def delayed_feedback(self):
        # Delayed feedback
        return [('x', 'x_delayed', 0.1)]

    def constant_inputs(self, ensemble_member):
        # Constant inputs
        return {'constant_input': Timeseries(np.hstack(([self.initial_time, self.times()])), np.hstack(([1.0], np.linspace(1.0, 0.0, 21))))}

    def bounds(self):
        bounds = super(TestProblemPathGoalsSmoothing, self).bounds()
        bounds['u'] = (-2.0, 2.0)
        return bounds

    def path_goals(self):
        return [PathGoal1(), PathGoal2(), PathGoalSmoothing()]


class TestGoalProgrammingSmoothing(TestCase):

    def setUp(self):
        self.problem = TestProblemPathGoalsSmoothing()
        self.problem.optimize()
        self.tolerance = 1e-6

    def test_x(self):
        value_tol = 1e-3
        for x in self.problem.extract_results()['x']:
            self.assertAlmostGreaterThan(x, 0.0, value_tol)
            self.assertAlmostLessThan(x, 1.1, value_tol)