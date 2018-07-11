import sys

import numpy as np

from .goal_programming_mixin import Goal, GoalProgrammingMixin


class _SmoothDynamicsGoal(Goal):
    def __init__(self, optimization_problem, state):
        # Store attributes
        self.state = state
        self.priority = optimization_problem.smooth_dynamics_priority

        # Calculate the maximum rate of change
        bounds = optimization_problem.bounds()[state]
        state_range_magnitude = bounds[1] - bounds[0]
        min_timestep = np.min(np.diff(optimization_problem.times()))
        abs_max_der = state_range_magnitude / min_timestep

        # Store function range and nominal
        self.function_range = 0.0, 1.0  # Not used by RTC
        self.function_nominal = (
            abs_max_der / optimization_problem.smooth_dynamics_multiplier
        )

        super().__init__()

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.der(self.state)


class SmoothDynamicsMixin(GoalProgrammingMixin):
    """
    Adds dynamics smoothing to your optimization problem *using goal
    programming*.

    Often there is some remaining flexibility in the model after the goals have
    completed. This mixin applies a low-priority derivative minimization goal to
    the states returned by smooth_dynamics_states().

    :cvar smooth_dynamics_priority: Priority at which the smoothing should
        run.  Default is ``sys.maxsize``.
    :cvar smooth_dynamics_multiplier: How aggressive the smoothing should be.
        Smaller is faster, larger is more smooth. Note that picking a number
        that is too large can cause convergence failures. Default is ``1.0``.
    """

    smooth_dynamics_priority = sys.maxsize
    smooth_dynamics_multiplier = 1.0

    def smooth_dynamics_states(self):
        """
        Returns a list of names (as strings) of states that should be smoothed.

        The default is the list returned by ``self.controls``.
        """
        return self.controls

    def path_goals(self):
        g = super().path_goals()
        for state in self.smooth_dynamics_states():
            g.append(_SmoothDynamicsGoal(self, state))
        return g
