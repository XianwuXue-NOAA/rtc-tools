import collections
import re

import numpy as np

from rtctools.simulation.simulation_problem import SimulationProblem

from test_case import TestCase

from .data_path import data_path


class SimulationModel(SimulationProblem):
    _force_zero_delay = True

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model",
            model_folder=data_path(),
        )

    def seed(self):
        seed = super().seed()
        seed["z"] = 1.0
        return seed

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options['library_folders'] = []
        return compiler_options


class TestSimulation(TestCase):
    def setUp(self):
        self.problem = SimulationModel()

    def test_object(self):
        self.assertIsInstance(self.problem, SimulationModel)

    def test_get_variables(self):
        all_variables = self.problem.get_variables()
        self.assertIsInstance(all_variables, collections.OrderedDict)

        self.assertEqual(
            set(all_variables),
            {
                "time",
                "constant_input",
                "k",
                "switched",
                "u",
                "u_out",
                "w",
                "der(w)",
                "x",
                "der(x)",
                "x_start",
                "y",
                "z",
                "_pymoca_delay_0[1,1]",
                "x_delayed",
            },
        )
        self.assertEqual(set(self.problem.get_parameter_variables()), {"x_start", "k"})
        self.assertEqual(set(self.problem.get_input_variables()), {"constant_input", "u"})
        self.assertEqual(
            set(self.problem.get_output_variables()),
            {"constant_output", "switched", "u_out", "y", "z", "x_delayed"},
        )

    def test_get_set_var(self):
        val = self.problem.get_var("switched")
        self.assertTrue(np.isnan(val))
        self.problem.set_var("switched", 10.0)
        val_reset = self.problem.get_var("switched")
        self.assertNotEqual(val_reset, val)

    def test_get_var_name_and_type(self):
        t = self.problem.get_var_type("switched")
        self.assertTrue(t == float)
        all_variables = self.problem.get_variables()
        idx = 0
        for var in all_variables.items():
            varname = var[0]
            if re.match(varname, "switched"):
                break
            idx += 1

        varname = self.problem.get_var_name(idx)
        self.assertEqual(varname, "switched")

    def test_get_time(self):
        # test methods for get_time
        start = 0.0
        stop = 10.0
        dt = 0.5
        self.problem.setup_experiment(start, stop, dt)
        self.problem.set_var("x_start", 0.0)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()
        self.assertAlmostEqual(self.problem.get_start_time(), start, 1e-6)
        self.assertAlmostEqual(self.problem.get_end_time(), stop, 1e-6)
        curtime = self.problem.get_current_time()
        self.assertAlmostEqual(curtime, start, 1e-6)
        while curtime < stop:
            self.problem.update(dt)
            curtime = self.problem.get_current_time()
        self.assertAlmostEqual(curtime, stop, 1e-6)

    def test_set_input(self):
        # run FMU model
        expected_values = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0, stop, dt)
        self.problem.set_var("x_start", 0.25)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem.set_var("u", 0.0)
            self.problem.update(dt)
            val = self.problem.get_var("switched")
            self.assertEqual(val, expected_values[i])

            # Test zero-delayed expression
            self.assertAlmostEqual(
                self.problem.get_var('x_delayed'), self.problem.get_var('x') * 3 + 1, 1e-6
            )
            i += 1

    def test_set_input2(self):
        # run FMU model
        expected_values = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        stop = 1.0
        dt = 0.1
        self.problem.setup_experiment(0.0, stop, dt)
        self.problem.set_var("x_start", 0.25)
        self.problem.set_var("constant_input", 0.0)
        self.problem.set_var("u", 0.0)
        self.problem.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem.set_var("u", i)
            self.problem.update(dt)
            val = self.problem.get_var("u_out")
            self.assertEqual(val, i + 1)
            val = self.problem.get_var("switched")
            self.assertEqual(val, expected_values[i])
            i += 1

    def test_seed(self):
        seed = self.problem.seed()
        self.assertEqual(seed["z"], 1.0)


class FailingSimulationModel(SimulationProblem):
    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model",
            model_folder=data_path(),
        )

    def compiler_options(self):
        compiler_options = super().compiler_options()
        compiler_options["cache"] = False
        compiler_options['library_folders'] = []
        return compiler_options


class TestFailingSimulation(TestCase):
    def test_delay_exception(self):
        with self.assertRaisesRegex(NotImplementedError, 'Delayed states are not supported'):
            self.problem = FailingSimulationModel()


class SimulationModelBase(SimulationProblem):
    _force_zero_delay = True

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_base",
            model_folder=data_path(),
        )


class SimulationModelNominal(SimulationProblem):
    _force_zero_delay = True

    def __init__(self):
        super().__init__(
            input_folder=data_path(),
            output_folder=data_path(),
            model_name="Model_nominal",
            model_folder=data_path(),
        )


class TestSimulationNominal(TestCase):
    def setUp(self):
        self.problem_base = SimulationModelBase()
        self.problem_nominal = SimulationModelNominal()

    def test_model_nominal(self):
        z_base = []
        z_nominal = []

        start = 0.0
        dt = 0.1
        stop = 1.0
        self.problem_base.setup_experiment(start, stop, dt)
        self.problem_base.set_var("x", 0)
        self.problem_base.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem_base.set_var("x", i / 100)
            self.problem_base.update(dt)
            z_base.append(self.problem_base.get_var("z"))
            i += 1

        self.problem_nominal.setup_experiment(start, stop, dt)
        self.problem_nominal.set_var("x", 0)
        self.problem_nominal.initialize()
        i = 0
        while i < int(stop / dt):
            self.problem_nominal.set_var("x", i / 100)
            self.problem_nominal.update(dt)
            z_nominal.append(self.problem_nominal.get_var("z"))
            i += 1

        self.assertAlmostEqual(np.array(z_base), np.array(z_nominal), 1e-5)
