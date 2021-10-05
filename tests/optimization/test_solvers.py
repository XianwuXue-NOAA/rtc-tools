import logging

import numpy as np

from test_case import TestCase

from .test_modelica_mixin import ModelAlgebraic

logger = logging.getLogger("rtctools")
logger.setLevel(logging.DEBUG)


class ModelCLP(ModelAlgebraic):
    def solver_options(self):
        options = super().solver_options()
        options["solver"] = "clp"
        options["casadi_solver"] = "qpsol"
        return options


class TestSolverCLP(TestCase):
    def setUp(self):
        self.problem = ModelCLP()
        self.problem.optimize()
        self.results = self.problem.extract_results()
        self.tolerance = 1e-6

    def test_solver_clp(self):
        self.assertAlmostEqual(
            self.results["y"] + self.results["u"],
            np.ones(len(self.problem.times())) * 1.0,
            self.tolerance,
        )
