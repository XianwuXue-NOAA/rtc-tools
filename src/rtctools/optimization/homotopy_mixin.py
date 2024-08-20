import logging
from typing import Dict, Union
import pandas as pd

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class HomotopyMixin(OptimizationProblem):
    """
    Adds homotopy to your optimization problem.  A homotopy is a continuous transformation between
    two optimization problems, parametrized by a single parameter :math:`\\theta \\in [0, 1]`.

    Homotopy may be used to solve non-convex optimization problems, by starting with a convex
    approximation at :math:`\\theta = 0.0` and ending with the non-convex problem at
    :math:`\\theta = 1.0`.

    .. note::

        It is advised to look for convex reformulations of your problem, before resorting to a use
        of the (potentially expensive) homotopy process.

    """

    def previous_result_seed(self, seed):
        # TODO generalize for used outside of homotopy
        # TODO rename function
        # TODO approach should work with csv and xml output
        # TODO approach is not robust
        prev_result = "<path_to_csv>results.csv"
        df = pd.read_csv(prev_result) # Read in the data from the previous results
        df.drop(columns=df.columns[0], axis=1, inplace=True) # Drop first column with the time
        dict_prev_result = df.to_dict('list') # Convert to dictionary

        # Assign the data from the results into the dictionary
        # TODO not robust
        # TODO detect time difference between t0 of prevous and current run
        # TODO detect if timesteps are consistent.
        # TODO detect if output was missing
        # TODO determine if separate output with all varaible outout is needed from previous run (full results dictionary)
        for key, result in dict_prev_result.items():
            times = self.times(key)
            times = times[1:]
            result = result[1:]

            seed[key] = Timeseries(times, result)

        #Return the result
        return seed

    def seed(self, ensemble_member):
        seed = super().seed(ensemble_member)
        options = self.homotopy_options()
        seeding_options = self.seeding_options()
        # TODO this is more general than homotopy - move to parent function
        # TODO set default use_previous_result_seed
        if seeding_options["use_previous_result_seed"]:
            seed = self.previous_result_seed(seed)
            return seed

        # Overwrite the seed only when the results of the latest run are
        # stored within this class. That is, when the GoalProgrammingMixin
        # class is not used or at the first run of the goal programming loop.
        elif self.__theta > options["theta_start"] and getattr(self, "_gp_first_run", True):
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if (result.ndim == 1 and len(result) == len(times)) or (
                    result.ndim == 2 and result.shape[0] == len(times)
                ):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)
                elif (result.ndim == 1 and len(result) == 1) or (
                    result.ndim == 2 and result.shape[0] == 1
                ):
                    seed[key] = result
        return seed

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        options = self.homotopy_options()
        try:
            # Only set the theta if we are in the optimization loop. We want
            # to avoid accidental usage of the parameter value in e.g. pre().
            # Note that we use a try-except here instead of hasattr, to avoid
            # explicit name mangling.
            parameters[options["homotopy_parameter"]] = self.__theta
        except AttributeError:
            pass

        return parameters

    def homotopy_options(self) -> Dict[str, Union[str, float]]:
        """
        Returns a dictionary of options controlling the homotopy process.

        +------------------------+------------+---------------+
        | Option                 | Type       | Default value |
        +========================+============+===============+
        | ``theta_start``        | ``float``  | ``0.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_0``      | ``float``  | ``1.0``       |
        +------------------------+------------+---------------+
        | ``delta_theta_min``    | ``float``  | ``0.01``      |
        +------------------------+------------+---------------+
        | ``homotopy_parameter`` | ``string`` | ``theta``     |
        +------------------------+------------+---------------+

        The homotopy process is controlled by the homotopy parameter in the model, specified by the
        option ``homotopy_parameter``.  The homotopy parameter is initialized to ``theta_start``,
        and increases to a value of ``1.0`` with a dynamically changing step size.  This step size
        is initialized with the value of the option ``delta_theta_0``.  If this step size is too
        large, i.e., if the problem with the increased homotopy parameter fails to converge, the
        step size is halved.  The process of halving terminates when the step size falls below the
        minimum value specified by the option ``delta_theta_min``.

        :returns: A dictionary of homotopy options.
        """

        return {
            "theta_start": 0.0,
            "delta_theta_0": 1.0,
            "delta_theta_min": 0.01,
            "homotopy_parameter": "theta",
        }

    def seeding_options(self) -> Dict[str, Union[str, float]]:
        # TODO call super here and move to goal_programmin_mixin, see def seed()
        """
        Returns a dictionary of options controlling the seeding process.

        +-------------------------------------+------------+---------------+
        | Option                              | Type       | Default value |
        +=====================================+============+===============+
        | ``use_previous_result_seed``        | ``Bool``   | ``False``     |
        +-------------------------------------+------------+---------------+

        The seeding process is controlled by the seeding_options. If ``use_previous_result_seed`` is
        true then, for the first priority, the solution to a previous run will be used as a seed.
        In this case the timeseries_export.xml or timeseries_export.csv from the revious run should
        be placed within the input folder of the model. The time differnce between initial start
        times of runs and timestep consistency is detected in ``previous_result_seed``.
        Otherwise, the seed is determined using information only from the current model.

        :returns: A dictionary of seeding options.
        """

        return {
            "use_previous_result_seed": False,
        }

    def dynamic_parameters(self):
        dynamic_parameters = super().dynamic_parameters()

        if self.__theta > 0:
            # For theta = 0, we don't mark the homotopy parameter as being dynamic,
            # so that the correct sparsity structure is obtained for the linear model.
            options = self.homotopy_options()
            dynamic_parameters.append(self.variable(options["homotopy_parameter"]))

        return dynamic_parameters

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Pre-processing
        if preprocessing:
            self.pre()

        options = self.homotopy_options()
        delta_theta = options["delta_theta_0"]

        # Homotopy loop
        self.__theta = options["theta_start"]

        while self.__theta <= 1.0:
            logger.info("Solving with homotopy parameter theta = {}.".format(self.__theta))

            success = super().optimize(
                preprocessing=False, postprocessing=False, log_solver_failure_as_error=False
            )
            if success:
                self.__results = [
                    self.extract_results(ensemble_member)
                    for ensemble_member in range(self.ensemble_size)
                ]

                if self.__theta == 0.0:
                    self.check_collocation_linearity = False
                    self.linear_collocation = False

                    # Recompute the sparsity structure for the nonlinear model family.
                    self.clear_transcription_cache()

            else:
                if self.__theta == options["theta_start"]:
                    break

                self.__theta -= delta_theta
                delta_theta /= 2

                if delta_theta < options["delta_theta_min"]:
                    failure_message = (
                        "Solver failed with homotopy parameter theta = {}. Theta cannot "
                        "be decreased further, as that would violate the minimum delta "
                        "theta of {}.".format(self.__theta, options["delta_theta_min"])
                    )
                    if log_solver_failure_as_error:
                        logger.error(failure_message)
                    else:
                        # In this case we expect some higher level process to deal
                        # with the solver failure, so we only log it as info here.
                        logger.info(failure_message)
                    break

            self.__theta += delta_theta

        # Post-processing
        if postprocessing:
            self.post()

        return success
