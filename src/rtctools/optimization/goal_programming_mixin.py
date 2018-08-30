import itertools
import logging
import sys
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Union

import casadi as ca

import numpy as np

from rtctools._internal.alias_tools import AliasDict

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class Goal(metaclass=ABCMeta):
    r"""
    Base class for lexicographic goal programming goals.

    A goal is defined by overriding the :func:`function` method.

    :cvar function_range:   Range of goal function.  *Required if a target is set*.
    :cvar function_nominal: Nominal value of function. Used for scaling.  Default is ``1``.
    :cvar target_min:       Desired lower bound for goal function.  Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function.  Default is ``numpy.nan``.
    :cvar priority:         Integer priority of goal.  Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal.  Default is ``1.0``.
    :cvar order:            Penalization order of goal violation.  Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.
                            Default is ``False``.
    :cvar relaxation:       Amount of slack added to the hard constraints related to the goal.
                            Must be a nonnegative value. Default is ``0.0``.

    The target bounds indicate the range within the function should stay, *if possible*.  Goals
    are, in that sense, *soft*, as opposed to standard hard constraints.

    Four types of goals can be created:

    1. Minimization goal if no target bounds are set:

       .. math::

            \min f

    2. Lower bound goal if ``target_min`` is set:

        .. math::

            m \leq f

    3. Upper bound goal if ``target_max`` is set:

        .. math::

            f \leq M

    4. Combined lower and upper bound goal if ``target_min`` and ``target_max`` are both set:

        .. math::

            m \leq f \leq M

    Lower priority goals take precedence over higher priority goals.

    Goals with the same priority are weighted off against each other in a single objective function.

    In goals where a target is set:
        * The function range interval must be provided as this is used to introduce hard constrains on the value that
          the function can take. If one is unsure about which value the function can take, it is recommended to
          overestimate this interval. However, an overestimated interval will negatively influence how accurately the
          target bounds are met.
        * The target provided must be contained in the function range.
        * The function nominal is used to scale the constraints.
        * If both a target_min and a target_max are set, the target maximum must be at least equal to minimum one.

    In minimization goals:
        * The function range is not used and therefore cannot be set.
        * The function nominal is used to scale the function value in the objective function. To ensure that all goals
          are given a similar importance, it is crucial to provide an accurate estimate of this parameter.

    The goal violation value is taken to the order'th power in the objective function of the final
    optimization problem.

    Relaxation is used to loosen the constraints that are set after the
    optimization of the goal's priority. The unit of the relaxation is equal
    to that of the goal function.

    Example definition of the point goal :math:`x(t) \geq 1.1` for :math:`t=1.0` at priority 1::

        class MyGoal(Goal):
            def function(self, optimization_problem, ensemble_member):
                # State 'x' at time t = 1.0
                t = 1.0
                return optimization_problem.state_at('x', t, ensemble_member)

            function_range = (1.0, 2.0)
            target_min = 1.1
            priority = 1

    Example definition of the path goal :math:`x(t) \geq 1.1` for all :math:`t` at priority 2::

        class MyPathGoal(Goal):
            def function(self, optimization_problem, ensemble_member):
                # State 'x' at any point in time
                return optimization_problem.state('x')

            function_range = (1.0, 2.0)
            target_min = 1.1
            priority = 2

    Note that for path goals, the ensemble member index is not passed to the call
    to :func:`OptimizationProblem.state`.  This call returns a time-independent symbol
    that is also independent of the active ensemble member.  Path goals are
    applied to all times and all ensemble members simultaneously.

    """

    @abstractmethod
    def function(self, optimization_problem: OptimizationProblem, ensemble_member: int) -> ca.MX:
        """
        This method returns a CasADi :class:`MX` object describing the goal function.

        :returns: A CasADi :class:`MX` object.
        """
        pass

    #: Range of goal function
    function_range = (np.nan, np.nan)

    #: Nominal value of function (used for scaling)
    function_nominal = 1.0

    #: Desired lower bound for goal function
    target_min = np.nan

    #: Desired upper bound for goal function
    target_max = np.nan

    #: Lower priority goals take precedence over higher priority goals.
    priority = 1

    #: Goals with the same priority are weighted off against each other in a
    #: single objective function.
    weight = 1.0

    #: The goal violation value is taken to the order'th power in the objective
    #: function.
    order = 2

    #: Critical goals must always be fully satisfied.
    critical = False

    #: Absolute relaxation applied to the optimized values of this goal
    relaxation = 0.0

    #: Timeseries ID for function value data (optional)
    function_value_timeseries_id = None

    #: Timeseries ID for goal violation data (optional)
    violation_timeseries_id = None

    @property
    def has_target_min(self) -> bool:
        """
        ``True`` if the user goal has min bounds.
        """
        if isinstance(self.target_min, Timeseries):
            return True
        else:
            return np.isfinite(self.target_min)

    @property
    def has_target_max(self) -> bool:
        """
        ``True`` if the user goal has max bounds.
        """
        if isinstance(self.target_max, Timeseries):
            return True
        else:
            return np.isfinite(self.target_max)

    @property
    def has_target_bounds(self) -> bool:
        """
        ``True`` if the user goal has min/max bounds.
        """
        return (self.has_target_min or self.has_target_max)

    @property
    def is_empty(self) -> bool:
        min_empty = (isinstance(self.target_min, Timeseries) and not np.any(np.isfinite(self.target_min.values)))
        max_empty = (isinstance(self.target_max, Timeseries) and not np.any(np.isfinite(self.target_max.values)))
        return min_empty and max_empty

    def get_function_key(self, optimization_problem: OptimizationProblem, ensemble_member: int) -> str:
        """
        Returns a key string uniquely identifying the goal function.  This
        is used to eliminate linearly dependent constraints from the optimization problem.
        """
        if hasattr(self, 'function_key'):
            return self.function_key

        # This must be deterministic.  See RTCTOOLS-485.
        if not hasattr(Goal, '_function_key_counter'):
            Goal._function_key_counter = 0
        self.function_key = '{}_{}'.format(self.__class__.__name__, Goal._function_key_counter)
        Goal._function_key_counter += 1

        return self.function_key

    def __repr__(self) -> str:
        return '{}(priority={}, target_min={}, target_max={}, function_range={})'.format(
            self.__class__, self.priority, self.target_min, self.target_max, self.function_range)


class StateGoal(Goal, metaclass=ABCMeta):
    r"""
    Base class for lexicographic goal programming path goals that act on a single model state.

    A state goal is defined by setting at least the ``state`` class variable.

    :cvar state:            State on which the goal acts.  *Required*.
    :cvar target_min:       Desired lower bound for goal function.  Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function.  Default is ``numpy.nan``.
    :cvar priority:         Integer priority of goal.  Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal.  Default is ``1.0``.
    :cvar order:            Penalization order of goal violation.  Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.
                            Default is ``False``.

    Example definition of the goal :math:`x(t) \geq 1.1` for all :math:`t` at priority 2::

        class MyStateGoal(StateGoal):
            state = 'x'
            target_min = 1.1
            priority = 2

    Contrary to ordinary ``Goal`` objects, ``PathGoal`` objects need to be initialized with an
    ``OptimizationProblem`` instance to allow extraction of state metadata, such as bounds and
    nominal values.  Consequently, state goals must be instantiated as follows::

        my_state_goal = MyStateGoal(optimization_problem)

    Note that ``StateGoal`` is a helper class.  State goals can also be defined using ``Goal`` as direct base class,
    by implementing the ``function`` method and providing the ``function_range`` and ``function_nominal``
    class variables manually.

    """

    #: The state on which the goal acts.
    state = None

    def __init__(self, optimization_problem):
        """
        Initialize the state goal object.

        :param optimization_problem: ``OptimizationProblem`` instance.
        """

        # Check whether a state has been specified
        if self.state is None:
            raise Exception('Please specify a state.')

        # Extract state range from model
        if self.has_target_bounds:
            try:
                self.function_range = optimization_problem.bounds()[self.state]
            except KeyError:
                raise Exception('State {} has no bounds or does not exist in the model.'.format(self.state))

            if self.function_range[0] is None:
                raise Exception('Please provide a lower bound for state {}.'.format(self.state))
            if self.function_range[1] is None:
                raise Exception('Please provide an upper bound for state {}.'.format(self.state))

        # Extract state nominal from model
        self.function_nominal = optimization_problem.variable_nominal(self.state)

        # Set function key
        canonical, sign = optimization_problem.alias_relation.canonical_signed(self.state)
        self.function_key = canonical if sign > 0.0 else '-' + canonical

    def function(self, optimization_problem, ensemble_member):
        return optimization_problem.state(self.state)

    def __repr__(self):
        return '{}(priority={}, state={}, target_min={}, target_max={}, function_range={})'.format(
            self.__class__, self.priority, self.state, self.target_min, self.target_max, self.function_range)


class GoalProgrammingMixin(OptimizationProblem, metaclass=ABCMeta):
    """
    Adds lexicographic goal programming to your optimization problem.
    """

    class __GoalConstraint:

        def __init__(
                self,
                goal: Goal,
                function: Callable[[OptimizationProblem], ca.MX],
                m: Union[float, Timeseries],
                M: Union[float, Timeseries],
                optimized: bool):
            self.goal = goal
            self.function = function
            self.min = m
            self.max = M
            self.optimized = optimized

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        # Initialize empty lists, so that the overridden methods may be called outside of the goal programming loop,
        # for example in pre().
        self.__subproblem_objectives = []
        self.__subproblem_constraints = []
        self.__subproblem_path_constraints = []
        self.__original_constant_input_keys = {}

        # List useful with the 'keep_eps_variable' True option
        self.__old_objective_functions = []
        self.__problem_path_timeseries = []
        self.__problem_epsilons = []
        self.__problem_path_epsilons = []
        self.__list_subproblem_epsilons = []
        self.__list_subproblem_path_epsilons = []
        # List useful with the 'keep_eps_variable' False option
        self.__subproblem_path_timeseries = []
        self.__subproblem_epsilons = []
        self.__subproblem_path_epsilons = []
        # List useful with the 'linear_obj_eps' True option
        self.__problem_epsilons_alias = []
        self.__problem_path_epsilons_alias = []
        self.__list_subproblem_epsilons_alias = []
        self.__list_subproblem_path_epsilons_alias = []
        self.__problem_constraint_epsilons_alias = []
        self.__problem_path_constraint_epsilons_alias = []

    @property
    def extra_variables(self):
        return self.__subproblem_epsilons + self.__problem_epsilons + self.__problem_epsilons_alias

    @property
    def path_variables(self):
        return (self.__subproblem_path_epsilons.copy() + self.__problem_path_epsilons.copy()
                + self.__problem_path_epsilons_alias.copy())

    def bounds(self):
        bounds = super().bounds()
        for epsilon in (self.__subproblem_epsilons + self.__subproblem_path_epsilons
                        + self.__problem_epsilons + self.__problem_path_epsilons
                        + self.__problem_epsilons_alias + self.__problem_path_epsilons_alias):
            bounds[epsilon.name()] = (0.0, 1.0)
        return bounds

    def constant_inputs(self, ensemble_member):
        constant_inputs = super().constant_inputs(ensemble_member)

        if ensemble_member not in self.__original_constant_input_keys:
            self.__original_constant_input_keys[ensemble_member] = set(constant_inputs.keys())

        # Remove min/max timeseries of previous priorities
        for k in set(constant_inputs.keys()):
            if k not in self.__original_constant_input_keys[ensemble_member]:
                del constant_inputs[k]

        # Append min/max timeseries to the constant inputs. Note that min/max
        # timeseries are shared between all ensemble members.
        if self.goal_programming_options()['keep_eps_variable']:
            for (variable, value) in self.__problem_path_timeseries:
                if not isinstance(value, Timeseries):
                    value = Timeseries(self.times(), np.full_like(self.times(), value))
                constant_inputs[variable] = value
        else:
            for (variable, value) in self.__subproblem_path_timeseries:
                if not isinstance(value, Timeseries):
                    value = Timeseries(self.times(), np.full_like(self.times(), value))
                constant_inputs[variable] = value
        return constant_inputs

    def seed(self, ensemble_member):
        if self.__first_run:
            seed = super().seed(ensemble_member)
        else:
            # Seed with previous results
            seed = AliasDict(self.alias_relation)
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if len(result) == len(times):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)

        # Seed the current epsilon
        for epsilon in (self.__subproblem_epsilons + self.__list_subproblem_epsilons
                        + self.__list_subproblem_epsilons_alias):
            seed[epsilon.name()] = 1.0

        times = self.times()
        for epsilon in (self.__subproblem_path_epsilons + self.__list_subproblem_path_epsilons
                        + self.__list_subproblem_path_epsilons_alias):
            seed[epsilon.name()] = Timeseries(times, np.ones(len(times)))

        return seed

    def __objective(self, subproblem_objectives, ensemble_member):
        if len(subproblem_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in subproblem_objectives]))

            if self.goal_programming_options()['scale_by_problem_size']:
                n_objectives = len(self.__subproblem_objectives) + len(self.__subproblem_path_objectives)
                acc_objective = acc_objective / n_objectives

            return acc_objective
        else:
            return ca.MX(0)

    def objective(self, ensemble_member):
        return self.__objective(self.__subproblem_objectives, ensemble_member)

    def __path_objective(self, subproblem_path_objectives, ensemble_member):
        if len(subproblem_path_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in subproblem_path_objectives]))

            if self.goal_programming_options()['scale_by_problem_size']:
                n_objectives = len(self.__subproblem_objectives) + len(self.__subproblem_path_objectives)
                acc_objective = acc_objective / n_objectives / len(self.times())

            return acc_objective
        else:
            return ca.MX(0)

    def path_objective(self, ensemble_member):
        return self.__path_objective(self.__subproblem_path_objectives, ensemble_member)

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        for l in self.__subproblem_constraints[ensemble_member].values():
            constraints.extend(((constraint.function(self), constraint.min, constraint.max) for constraint in l))
        if self.goal_programming_options()['keep_eps_variable']:
            # Pareto optimality constraint for minimization goals
            for old_objs, old_path_objs, val in self.__old_objective_functions:
                tot_expr = 0.0
                for ensemble_member in range(self.ensemble_size):
                    expr = self.__objective(old_objs, ensemble_member)
                    expr += ca.sum1(self.map_path_expression(self.__path_objective(old_path_objs, ensemble_member),
                                                             ensemble_member))
                    tot_expr += self.ensemble_member_probability(ensemble_member) * expr
                # Add a relaxation to the value
                val += self.goal_programming_options()['constraint_relaxation']
                constraints.append((tot_expr, -np.inf, val))
            if self.goal_programming_options()['linear_obj_eps']:
                # Epsilon alias constraints
                for constraint_eps in self.__problem_constraint_epsilons_alias[ensemble_member]:
                    constraints.append(constraint_eps(self, ensemble_member), -np.inf, 0.0)
        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)
        for l in self.__subproblem_path_constraints[ensemble_member].values():
            path_constraints.extend(((constraint.function(self), constraint.min, constraint.max) for constraint in l))
        if self.goal_programming_options()['linear_obj_eps']:
            # Epsilon alias path constraints
            for constraint_eps in self.__problem_path_constraint_epsilons_alias[ensemble_member]:
                path_constraints.append((constraint_eps(self, ensemble_member), -np.inf, 0.0))
        return path_constraints

    def solver_options(self):
        # Call parent
        options = super().solver_options()

        solver = options['solver']
        assert solver in ['bonmin', 'ipopt']

        # Make sure constant states, such as min/max timeseries for violation variables,
        # are turned into parameters for the final optimization problem.
        ipopt_options = options[solver]
        ipopt_options['fixed_variable_treatment'] = 'make_parameter'

        # Define temporary variable to avoid infinite loop between
        # solver_options and goal_programming_options.
        self._loop_breaker_solver_options = True

        if not hasattr(self, '_loop_breaker_goal_programming_options'):
            if not self.goal_programming_options()['mu_reinit']:
                ipopt_options['mu_strategy'] = 'monotone'
                ipopt_options['gather_stats'] = True
                if not self.__first_run:
                    ipopt_options['mu_init'] = self.solver_stats['iterations'][
                        'mu'][-1]

        delattr(self, '_loop_breaker_solver_options')

        # Done
        return options

    def goal_programming_options(self) -> Dict[str, Union[float, bool]]:
        """
        Returns a dictionary of options controlling the goal programming process.

        +---------------------------+-----------+---------------+
        | Option                    | Type      | Default value |
        +===========================+===========+===============+
        | ``constraint_relaxation`` | ``float`` | ``0.0``       |
        +---------------------------+-----------+---------------+
        | ``mu_reinit``             | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``fix_minimized_values``  | ``bool``  | ``True/False``|
        +---------------------------+-----------+---------------+
        | ``check_monotonicity``    | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``equality_threshold``    | ``float`` | ``1e-8``      |
        +---------------------------+-----------+---------------+
        | ``interior_distance``     | ``float`` | ``1e-6``      |
        +---------------------------+-----------+---------------+
        | ``scale_by_problem_size`` | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+
        | ``keep_eps_variable``     | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+
        | ``linear_obj_eps``        | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+

        Constraints generated by the goal programming algorithm are relaxed by applying the
        specified relaxation. Use of this option is normally not required.

        A goal is considered to be violated if the violation, scaled between 0 and 1, is greater
        than the specified tolerance. Violated goals are fixed.  Use of this option is normally not
        required.

        When using the default solver (IPOPT), its barrier parameter ``mu`` is
        normally re-initialized a every iteration of the goal programming
        algorithm, unless mu_reinit is set to ``False``.  Use of this option
        is normally not required.

        If ``fix_minimized_values`` is set to ``True``, goal functions will be set to equal their
        optimized values in optimization problems generated during subsequent priorities. Otherwise,
        only an upper bound will be set. Use of this option is normally not required.
        Note that a non-zero goal relaxation overrules this option; a non-zero relaxation will always
        result in only an upper bound being set.
        Also note that the use of this option may add non-convex constraints to the optimization problem.
        The default value for this parameter is ``True`` for the default solvers IPOPT/BONMIN. If any
        other solver is used, the default value is ``False``.

        If ``check_monotonicity`` is set to ``True``, then it will be checked whether goals with the same
        function key form a monotonically decreasing sequence with regards to the target interval.

        The option ``equality_threshold`` controls when a two-sided inequality constraint is folded into
        an equality constraint.

        The option ``interior_distance`` controls the distance from the scaled target bounds, starting
        from which the function value is considered to lie in the interior of the target space.

        If ``scale_by_problem_size`` is set to ``True``, the objective (i.e. the sum of epsilons)
        will be divided by the number of goals, and the path objective will be divided by the number
        of path goals and the number of time steps. This will make sure the objectives are always in
        the range [0, 1], at the cost of solving each goal/time step less accurately.

        The option ``keep_eps_variable`` controls how the epsilon variables introduced in the target
        goals are dealt with in subsequent priorities.
        If ``keep_eps_variable`` is set to False, each epsilon is replaced by its computed value and
        those are used to derive a new set of constraints.
        If ``keep_eps_variable`` is set to True, the epsilon are kept as variables and the constraints
        are not modified. To ensure the goal programming philosophy, i.e., Pareto optimality, a single
        constraint is added to enforce that the objective function must always be at most the objective
        value. This method allows for a larger solution space, at the cost of having a (possibly) more complex
        optimization problem. Indeed, more variables are kept around throughout the optimization and any
        objective function is turned into a constraint for the subsequent priorities (while in the False
        option this was the case only for the function of minimization goals).

        Option ``linear_obj_eps``can be set to True only if ``keep_eps_variable`` is set to True
        and all the minimization goals have order one.
        If ``linear_obj_eps`` is set to True, the objective function constructed by the target goals
        with order larger than one will be linearized by introducing alias epsilon variables.
        For example, if the objective function is equal to eps_1^2 + eps_2^2, then it will become
        eps*_1 + eps*_2 with the additional constraints eps_1^2 <= eps*_1 and eps_2^2 <= eps*_2.
        This option should be used, e.g., in the case a solver can handle quadratically constrainted
        problems but not quadratic objective function and quadratic constrainted problems.

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options['mu_reinit'] = True
        options['constraint_relaxation'] = 0.0  # Disable by default
        options['violation_tolerance'] = np.inf  # Disable by default
        options['fix_minimized_values'] = False
        options['check_monotonicity'] = True
        options['equality_threshold'] = 1e-8
        options['interior_distance'] = 1e-6
        options['scale_by_problem_size'] = False
        options['keep_eps_variable'] = False
        options['linear_obj_eps'] = False

        # Define temporary variable to avoid infinite loop between
        # solver_options and goal_programming_options.
        self._loop_breaker_goal_programming_options = True

        if not hasattr(self, '_loop_breaker_solver_options'):
            if self.solver_options()['solver'] in {'ipopt', 'bonmin'}:
                options['fix_minimized_values'] = True

        delattr(self, '_loop_breaker_goal_programming_options')

        return options

    def goals(self) -> List[Goal]:
        """
        User problem returns list of :class:`Goal` objects.

        :returns: A list of goals.
        """
        return []

    def path_goals(self) -> List[Goal]:
        """
        User problem returns list of path :class:`Goal` objects.

        :returns: A list of path goals.
        """
        return []

    def __add_goal_constraint(self, goal, epsilon, ensemble_member, options):
        constraints = self.__subproblem_constraints[
            ensemble_member].get(goal.get_function_key(self, ensemble_member), [])

        if isinstance(epsilon, ca.MX):
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self.__GoalConstraint(
                        goal,
                        lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (
                            goal.function(problem, ensemble_member) - problem.extra_variable(
                                epsilon.name(), ensemble_member=ensemble_member) *
                            (goal.function_range[0] - goal.target_min) - goal.target_min) / goal.function_nominal,
                        0.0, np.inf, False)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self.__GoalConstraint(
                        goal,
                        lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: (
                            goal.function(problem, ensemble_member) - problem.extra_variable(
                                epsilon.name(), ensemble_member=ensemble_member) *
                            (goal.function_range[1] - goal.target_max) - goal.target_max) / goal.function_nominal,
                        -np.inf, 0.0, False)
                    constraints.append(constraint)

            # TODO forgetting max like this.
            # Epsilon is not fixed yet.  This constraint is therefore linearly independent of any existing constraints,
            # and we add it to the list of constraints for this state.  We keep the existing constraints to ensure
            # that the attainment of previous goals is not worsened.
            self.__subproblem_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = constraints
        else:
            fix_value = False

            constraint = self.__GoalConstraint(
                goal,
                lambda problem, ensemble_member=ensemble_member, goal=goal: (
                    goal.function(problem, ensemble_member) / goal.function_nominal),
                -np.inf, np.inf, True)
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if epsilon <= options['violation_tolerance']:
                    if goal.has_target_min:
                        constraint.min = (
                            epsilon * (goal.function_range[0] - goal.target_min) +
                            goal.target_min - goal.relaxation) / goal.function_nominal
                    if goal.has_target_max:
                        constraint.max = (
                            epsilon * (goal.function_range[1] - goal.target_max) +
                            goal.target_max + goal.relaxation) / goal.function_nominal
                    if goal.has_target_min and goal.has_target_max:
                        if abs(constraint.min - constraint.max) < options['equality_threshold']:
                            avg = 0.5 * (constraint.min + constraint.max)
                            constraint.min = constraint.max = avg
                else:
                    # Equality constraint to optimized value
                    fix_value = True

                    function = ca.Function('f', [self.solver_input], [goal.function(self, ensemble_member)])
                    value = function(self.solver_output)

                    constraint.min = (value - goal.relaxation) / goal.function_nominal
                    constraint.max = (value + goal.relaxation) / goal.function_nominal
            else:
                # Epsilon encodes the position within the function range.
                fix_value = True

                if options['fix_minimized_values'] and goal.relaxation == 0.0:
                    constraint.min = epsilon / goal.function_nominal
                    constraint.max = epsilon / goal.function_nominal
                else:
                    constraint.min = -np.inf
                    constraint.max = (epsilon + goal.relaxation) / goal.function_nominal

            # Epsilon is fixed.  Override previous {min,max} constraints for
            # this state.
            if not fix_value:
                for existing_constraint in constraints:
                    if goal is not existing_constraint.goal and existing_constraint.optimized:
                        constraint.min = max(constraint.min, existing_constraint.min)
                        constraint.max = min(constraint.max, existing_constraint.max)

                        # Ensure new constraint does not loosen or shift
                        # previous hard constraints due to numerical errors.
                        constraint.min = min(constraint.min, existing_constraint.max)
                        constraint.max = max(constraint.max, existing_constraint.min)

            # Ensure consistency of bounds.  Bounds may become inconsistent due to
            # small numerical computation errors.
            constraint.min = min(constraint.min, constraint.max)

            self.__subproblem_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = [constraint]

    def __min_max_arrays(self, g):
        times = self.times()

        m, M = None, None
        if isinstance(g.target_min, Timeseries):
            m = self.interpolate(
                times, g.target_min.times, g.target_min.values, -np.inf, -np.inf)
        else:
            m = g.target_min * np.ones(len(times))
        if isinstance(g.target_max, Timeseries):
            M = self.interpolate(
                times, g.target_max.times, g.target_max.values, np.inf, np.inf)
        else:
            M = g.target_max * np.ones(len(times))

        return m, M

    def __add_path_goal_constraint(self, goal, epsilon, ensemble_member, options, min_series=None, max_series=None):
        # Generate list of min and max values
        times = self.times()

        goal_m, goal_M = self.__min_max_arrays(goal)

        constraints = self.__subproblem_path_constraints[
            ensemble_member].get(goal.get_function_key(self, ensemble_member), [])

        if isinstance(epsilon, ca.MX):
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self.__GoalConstraint(
                        goal,
                        lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: ca.if_else(
                            problem.variable(min_series) > -sys.float_info.max,
                            (goal.function(problem, ensemble_member) -
                             problem.variable(epsilon.name()) * (
                                goal.function_range[0] - problem.variable(min_series)) -
                             problem.variable(min_series)) / goal.function_nominal,
                            0.0),
                        0.0, np.inf, False)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self.__GoalConstraint(
                        goal,
                        lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon: ca.if_else(
                            problem.variable(max_series) < sys.float_info.max,
                            (goal.function(problem, ensemble_member) -
                             problem.variable(epsilon.name()) * (
                                goal.function_range[1] - problem.variable(max_series)) -
                             problem.variable(max_series)) / goal.function_nominal,
                            0.0),
                        -np.inf, 0.0, False)
                    constraints.append(constraint)

            # TODO forgetting max like this.
            # Epsilon is not fixed yet.  This constraint is therefore linearly independent of any existing constraints,
            # and we add it to the list of constraints for this state.  We keep the existing constraints to ensure
            # that the attainment of previous goals is not worsened.
            self.__subproblem_path_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = constraints
        else:
            fix_value = False

            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                m, M = np.full_like(times, -np.inf, dtype=np.float64), np.full_like(times, np.inf, dtype=np.float64)

                # Compute each min, max value separately for every time step
                for i, t in enumerate(times):
                    if np.isfinite(goal_m[i]) or np.isfinite(goal_M[i]):
                        if epsilon[i] <= options['violation_tolerance']:
                            if np.isfinite(goal_m[i]):
                                m[i] = (epsilon[i] * (goal.function_range[0] -
                                                      goal_m[i]) + goal_m[i] - goal.relaxation) / goal.function_nominal
                            if np.isfinite(goal_M[i]):
                                M[i] = (epsilon[i] * (goal.function_range[1] -
                                                      goal_M[i]) + goal_M[i] + goal.relaxation) / goal.function_nominal
                            if np.isfinite(goal_m[i]) and np.isfinite(goal_M[i]):
                                if abs(m[i] - M[i]) < options['equality_threshold']:
                                    avg = 0.5 * (m[i] + M[i])
                                    m[i] = M[i] = avg
                        else:
                            # Equality constraint to optimized value
                            # TODO this does not perform well.
                            variables = self.dae_variables['states'] + self.dae_variables['algebraics'] + \
                                self.dae_variables['control_inputs'] + self.dae_variables['constant_inputs']
                            values = [self.state_at(
                                variable, t, ensemble_member=ensemble_member) for variable in variables]
                            [function] = ca.substitute(
                                [goal.function(self, ensemble_member)], variables, values)
                            function = ca.Function('f', [self.solver_input], [function])
                            value = function(self.solver_output)

                            m[i] = (value - goal.relaxation) / goal.function_nominal
                            M[i] = (value + goal.relaxation) / goal.function_nominal
            else:
                # Epsilon encodes the position within the function range.
                fix_value = True

                if options['fix_minimized_values'] and goal.relaxation == 0.0:
                    m = epsilon / goal.function_nominal
                    M = epsilon / goal.function_nominal
                else:
                    m = -np.inf * np.ones(len(times))
                    M = (epsilon + goal.relaxation) / goal.function_nominal

            constraint = self.__GoalConstraint(
                goal,
                lambda problem, ensemble_member=ensemble_member, goal=goal: (
                    goal.function(problem, ensemble_member) / goal.function_nominal),
                Timeseries(times, m), Timeseries(times, M), True)

            # Epsilon is fixed. Propagate/override previous {min,max}
            # constraints for this state.
            if not fix_value:
                for existing_constraint in constraints:
                    if goal is not existing_constraint.goal and existing_constraint.optimized:
                        constraint.min = Timeseries(
                            times, np.maximum(constraint.min.values, existing_constraint.min.values))
                        constraint.max = Timeseries(
                            times, np.minimum(constraint.max.values, existing_constraint.max.values))

                        # Ensure new constraint does not loosen or shift
                        # previous hard constraints due to numerical errors.
                        constraint.min = Timeseries(
                            times, np.minimum(constraint.min.values, existing_constraint.max.values))
                        constraint.max = Timeseries(
                            times, np.maximum(constraint.max.values, existing_constraint.min.values))

            # Ensure consistency of bounds.  Bounds may become inconsistent due to
            # small numerical computation errors.
            constraint.min = Timeseries(times, np.minimum(constraint.min.values, constraint.max.values))

            self.__subproblem_path_constraints[ensemble_member][
                goal.get_function_key(self, ensemble_member)] = [constraint]

    def __validate_goals(self, goals):
        goals = sorted(goals, key=lambda x: x.priority)

        options = self.goal_programming_options()

        # Validate goal definitions
        for goal in goals:
            m, M = goal.function_range

            # The function range should not be a symbolic expression
            assert (not isinstance(m, ca.MX) or m.is_constant())
            assert (not isinstance(M, ca.MX) or M.is_constant())

            m, M = float(m), float(M)

            if goal.function_nominal <= 0:
                raise Exception("Nonpositive nominal value specified for goal {}".format(goal))

            if goal.critical and not goal.has_target_bounds:
                raise Exception("Minimization goals cannot be critical")

            if goal.has_target_bounds:
                if not np.isfinite(m) or not np.isfinite(M):
                    raise Exception("No function range specified for goal {}".format(goal))

                if m >= M:
                    raise Exception("Invalid function range for goal {}.".format(goal))
            else:
                if goal.function_range != (np.nan, np.nan):
                    raise Exception("Specifying function range not allowed for goal {}".format(goal))

            try:
                int(goal.priority)
            except ValueError:
                raise Exception("Priority of not int or castable to int for goal {}".format(goal))

        options = self.goal_programming_options()
        if not options['keep_eps_variable']:
            if options['linear_obj_eps']:
                raise Exception("The 'linear_obj_eps' option can be set to True only if "
                                "the 'keep_eps_variable' option is also set to True")

        # Check consistency and monotonicity of goals. Scalar target min/max
        # of normal goals are also converted to arrays to unify checks with
        # path goals.
        if options['check_monotonicity']:
            for e in range(self.ensemble_size):
                # Store the previous goal of a certain function key we
                # encountered, such that we can compare to it.
                fk_goal_map = {}

                for goal in goals:
                    fk = goal.get_function_key(self, e)
                    prev = fk_goal_map.get(fk)
                    fk_goal_map[fk] = goal

                    if prev is not None:
                        goal_m, goal_M = self.__min_max_arrays(goal)
                        other_m, other_M = self.__min_max_arrays(prev)

                        indices = np.where(np.logical_not(np.logical_or(
                            np.isnan(goal_m), np.isnan(other_m))))
                        if goal.has_target_min:
                            if np.any(goal_m[indices] < other_m[indices]):
                                raise Exception(
                                    'Target minimum of goal {} must be greater or equal than '
                                    'target minimum of goal {}.'.format(goal, prev))

                        indices = np.where(np.logical_not(np.logical_or(
                            np.isnan(goal_M), np.isnan(other_M))))
                        if goal.has_target_max:
                            if np.any(goal_M[indices] > other_M[indices]):
                                raise Exception(
                                    'Target maximum of goal {} must be less or equal than '
                                    'target maximum of goal {}'.format(goal, prev))

        for goal in goals:
            goal_m, goal_M = self.__min_max_arrays(goal)

            if goal.has_target_min and goal.has_target_max:
                indices = np.where(np.logical_not(np.logical_or(
                    np.isnan(goal_m), np.isnan(goal_M))))

                if np.any(goal_m[indices] > goal_M[indices]):
                    raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

            if goal.has_target_min:
                indices = np.where(np.logical_not(np.isnan(goal_m)))
                if np.any(goal_m[indices] <= goal.function_range[0]):
                    raise Exception(
                        'Target minimum should be greater than the lower bound of the function range for goal {}'
                        .format(goal))
            if goal.has_target_max:
                indices = np.where(np.logical_not(np.isnan(goal_M)))
                if np.any(goal_M[indices] >= goal.function_range[1]):
                    raise Exception(
                        'Target maximum should be smaller than the upper bound of the function range for goal {}'
                        .format(goal))

            if goal.relaxation < 0.0:
                raise Exception('Relaxation of goal {} should be a nonnegative value'.format(goal))

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do pre-processing
        if preprocessing:
            self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        # Validate goal definitions
        self.__validate_goals(goals)
        self.__validate_goals(path_goals)

        priorities = {int(goal.priority) for goal in itertools.chain(goals, path_goals) if not goal.is_empty}

        for priority in sorted(priorities):
            subproblems.append((
                priority,
                [goal for goal in goals if int(goal.priority) == priority and not goal.is_empty],
                [goal for goal in path_goals if int(goal.priority) == priority and not goal.is_empty]))

        options = self.goal_programming_options()

        # Solve the subproblems one by one
        logger.info("Starting goal programming")

        success = False

        times = self.times()

        self.__subproblem_constraints = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__subproblem_path_constraints = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__first_run = True
        self.__results_are_current = False
        self.__original_constant_input_keys = {}

        if options['keep_eps_variable']:
            # Expanding list of variables / objective functions
            self.__old_objective_functions = []
            self.__problem_path_timeseries = []
            self.__problem_epsilons = []
            self.__problem_path_epsilons = []
        if options['linear_obj_eps']:
            # Expanding list of epsilon alias variables
            self.__problem_constraint_epsilons_alias = [[] for ensemble_member in range(self.ensemble_size)]
            self.__problem_path_constraint_epsilons_alias = [[] for ensemble_member in range(self.ensemble_size)]

        for i, (priority, goals, path_goals) in enumerate(subproblems):
            logger.info("Solving goals at priority {}".format(priority))

            # Call the pre priority hook
            self.priority_started(priority)

            # Reset objective function
            self.__subproblem_objectives = []
            self.__subproblem_path_objectives = []

            if options['keep_eps_variable']:
                # Reset list of epsilon variables of the current priority. Used to provide the correct seed
                self.__list_subproblem_epsilons = []
                self.__list_subproblem_path_epsilons = []
            else:
                # Reset list of epsilons and path_timeseries
                self.__subproblem_epsilons = []
                self.__subproblem_path_epsilons = []
                self.__subproblem_path_timeseries = []

            if options['linear_obj_eps']:
                # Reset epsilon alias
                self.__list_subproblem_epsilons_alias = []
                self.__list_subproblem_path_epsilons_alias = []

            for j, goal in enumerate(goals):
                if goal.critical:
                    epsilon = 0.0
                else:
                    if goal.has_target_bounds:
                        epsilon = ca.MX.sym('eps_{}_{}'.format(i, j))
                        if options['keep_eps_variable']:
                            self.__problem_epsilons.append(epsilon)
                            self.__list_subproblem_epsilons.append(epsilon)
                            if options['linear_obj_eps'] and goal.order != 1:
                                epsilon_alias = ca.MX.sym('eps_alias_{}_{}'.format(i, j))
                                self.__problem_epsilons_alias.append(epsilon_alias)
                                self.__list_subproblem_epsilons_alias.append(epsilon_alias)
                        else:
                            self.__subproblem_epsilons.append(epsilon)

                if not goal.critical:
                    if goal.has_target_bounds:
                        if options['linear_obj_eps'] and goal.order != 1:
                            # If options['linear_obj_eps'] is set to True and the order is not one,
                            # than the objective function is the linear sum of the epsilon alias variables
                            self.__subproblem_objectives.append(
                                lambda problem, ensemble_member, goal=goal, epsilon_alias=epsilon_alias: (
                                        goal.weight * problem.extra_variable(epsilon_alias.name(),
                                                                             ensemble_member=ensemble_member)))
                        else:
                            self.__subproblem_objectives.append(
                                lambda problem, ensemble_member, goal=goal, epsilon=epsilon:
                                (goal.weight * ca.constpow(problem.extra_variable(epsilon.name(),
                                                                                  ensemble_member=ensemble_member),
                                                           goal.order)))
                    else:
                        if options['linear_obj_eps'] and goal.order != 1:
                            raise Exception("If 'linear_obj_eps' is set to True than "
                                            "all the minimization goals must have order one")
                        else:
                            self.__subproblem_objectives.append(lambda problem, ensemble_member, goal=goal: (
                                    goal.weight * ca.constpow(goal.function(problem, ensemble_member)
                                                              / goal.function_nominal, goal.order)))

                if goal.has_target_bounds:
                    for ensemble_member in range(self.ensemble_size):
                        self.__add_goal_constraint(
                            goal, epsilon, ensemble_member, options)
                        if options['linear_obj_eps'] and not goal.critical and goal.order != 1:
                            # If 'linear_obj_eps' is set to True, the order is not one and the goal is not critical,
                            # for each epsilon we add the constraint epsilon^order <= epsilon_alias
                            self.__problem_constraint_epsilons_alias[ensemble_member].append(
                                lambda problem, ensemble_member, goal=goal,
                                epsilon=epsilon, epsilon_alias=epsilon_alias: (ca.constpow(
                                    problem.extra_variable(epsilon.name(), ensemble_member=ensemble_member),
                                    goal.order) - problem.extra_variable(epsilon_alias.name(),
                                                                         ensemble_member=ensemble_member)))

            for j, goal in enumerate(path_goals):
                if goal.critical:
                    epsilon = np.zeros(len(self.times()))
                else:
                    if goal.has_target_bounds:
                        epsilon = ca.MX.sym('path_eps_{}_{}'.format(i, j))
                        if options['keep_eps_variable']:
                            self.__problem_path_epsilons.append(epsilon)
                            self.__list_subproblem_path_epsilons.append(epsilon)
                            if options['linear_obj_eps'] and goal.order != 1:
                                epsilon_alias = ca.MX.sym('path_eps_alias_{}_{}'.format(i, j))
                                self.__problem_path_epsilons_alias.append(epsilon_alias)
                                self.__list_subproblem_path_epsilons_alias.append(epsilon_alias)
                        else:
                            self.__subproblem_path_epsilons.append(epsilon)

                if goal.has_target_min:
                    min_series = 'path_min_{}_{}'.format(i, j)

                    if isinstance(goal.target_min, Timeseries):
                        target_min = Timeseries(goal.target_min.times, goal.target_min.values)
                        target_min.values[
                            np.logical_or(
                                np.isnan(target_min.values),
                                np.isneginf(target_min.values))
                            ] = -sys.float_info.max
                    else:
                        target_min = goal.target_min

                    if options['keep_eps_variable']:
                        self.__problem_path_timeseries.append((min_series, target_min))
                    else:
                        self.__subproblem_path_timeseries.append((min_series, target_min))
                else:
                    min_series = None
                if goal.has_target_max:
                    max_series = 'path_max_{}_{}'.format(i, j)

                    if isinstance(goal.target_max, Timeseries):
                        target_max = Timeseries(goal.target_max.times, goal.target_max.values)
                        target_max.values[
                            np.logical_or(
                                np.isnan(target_max.values),
                                np.isposinf(target_max.values))
                            ] = sys.float_info.max
                    else:
                        target_max = goal.target_max

                    if options['keep_eps_variable']:
                        self.__problem_path_timeseries.append((max_series, target_max))
                    else:
                        self.__subproblem_path_timeseries.append((max_series, target_max))

                else:
                    max_series = None

                if not goal.critical:
                    if goal.has_target_bounds:
                        if options['linear_obj_eps'] and goal.order != 1:
                            # If 'linear_obj_eps' is set to True and the order is not one, than the objective
                            # function is the linear sum of the epsilon alias variables
                            self.__subproblem_objectives.append(
                                lambda problem, ensemble_member, goal=goal, epsilon_alias=epsilon_alias: (
                                        goal.weight * ca.sum1(problem.state_vector(epsilon_alias.name(),
                                                                                   ensemble_member=ensemble_member))))
                        else:
                            self.__subproblem_objectives.append(
                                lambda problem, ensemble_member, goal=goal, epsilon=epsilon:
                                (goal.weight * ca.sum1(ca.constpow(
                                    problem.state_vector(epsilon.name(), ensemble_member=ensemble_member),
                                    goal.order))))
                    else:
                        if options['linear_obj_eps'] and goal.order != 1:
                            raise Exception("If 'linear_obj_eps' is set to True than "
                                            "all the minimization goals must have order one")
                        else:
                            self.__subproblem_path_objectives.append(
                                lambda problem, ensemble_member, goal=goal: (goal.weight * ca.constpow(
                                    goal.function(problem, ensemble_member) / goal.function_nominal, goal.order)))

                if goal.has_target_bounds:
                    for ensemble_member in range(self.ensemble_size):
                        self.__add_path_goal_constraint(
                            goal, epsilon, ensemble_member, options, min_series, max_series)
                        if options['linear_obj_eps'] and not goal.critical and goal.order != 1:
                            # If 'linear_obj_eps' is set to True, the order is not one and the goal is not critical,
                            # for each epsilon we add the constraint epsilon^order <= epsilon_alias
                            self.__problem_path_constraint_epsilons_alias[ensemble_member].append(
                                lambda problem, ensemble_member, goal=goal,
                                epsilon=epsilon, epsilon_alias=epsilon_alias:
                                (ca.constpow(problem.variable(epsilon.name()), goal.order) -
                                 problem.variable(epsilon_alias.name())))

            # Check no intersection between 'keep_eps_variable' option True vs False
            assert not (len(self.__problem_path_timeseries) != 0 and len(self.__subproblem_path_timeseries) != 0)
            assert not (len(self.__problem_epsilons) != 0 and len(self.__subproblem_epsilons) != 0)
            assert not (len(self.__problem_path_epsilons) != 0 and len(self.__subproblem_path_epsilons) != 0)

            # Solve subproblem
            success = super().optimize(
                preprocessing=False, postprocessing=False, log_solver_failure_as_error=log_solver_failure_as_error)
            if not success:
                break

            self.__first_run = False

            # Store results.  Do this here, to make sure we have results even
            # if a subsequent priority fails.
            self.__results_are_current = False
            self.__results = [self.extract_results(
                ensemble_member) for ensemble_member in range(self.ensemble_size)]
            self.__results_are_current = True

            # Call the post priority hook, so that intermediate results can be
            # logged/inspected.
            self.priority_completed(priority)

            if options['keep_eps_variable']:
                # Extract information about the objective value, this is used for the Pareto optimality constraint.
                # We only retain information about the objective functions defined through the goal framework as user
                # define objective functions may relay on local variables.
                val = 0.0
                for ensemble_member in range(self.ensemble_size):
                    expr = self.__objective(self.__subproblem_objectives, ensemble_member)
                    expr += ca.sum1(self.map_path_expression(self.__path_objective(self.__subproblem_path_objectives,
                                                                                   ensemble_member), ensemble_member))
                    f = ca.Function('tmp', [self.solver_input], [expr])
                    val += self.ensemble_member_probability(ensemble_member) * float(f(self.solver_output))

                self.__old_objective_functions.append((self.__subproblem_objectives.copy(),
                                                       self.__subproblem_path_objectives.copy(), val))
            else:
                # Re-add constraints, this time with epsilon values fixed
                for ensemble_member in range(self.ensemble_size):
                    for j, goal in enumerate(goals):
                        if goal.critical:
                            continue

                        if (
                                not goal.has_target_bounds or
                                goal.violation_timeseries_id is not None or
                                goal.function_value_timeseries_id is not None
                                ):
                            f = ca.Function('f', [self.solver_input], [goal.function(self, ensemble_member)])
                            function_value = float(f(self.solver_output))

                            # Store results
                            if goal.function_value_timeseries_id is not None:
                                self.set_timeseries(
                                    goal.function_value_timeseries_id,
                                    np.full_like(times, function_value), ensemble_member)

                        if goal.has_target_bounds:
                            epsilon = self.__results[ensemble_member][
                                'eps_{}_{}'.format(i, j)]

                            # Store results
                            # TODO tolerance
                            if goal.violation_timeseries_id is not None:
                                epsilon_active = epsilon
                                w = True
                                w &= (not goal.has_target_min or
                                      function_value / goal.function_nominal >
                                      goal.target_min / goal.function_nominal + options['interior_distance'])
                                w &= (not goal.has_target_max or
                                      function_value / goal.function_nominal <
                                      goal.target_max / goal.function_nominal - options['interior_distance'])
                                if w:
                                    epsilon_active = np.nan
                                self.set_timeseries(
                                    goal.violation_timeseries_id,
                                    np.full_like(times, epsilon_active), ensemble_member)

                            # Add a relaxation to appease the barrier method.
                            epsilon += options['constraint_relaxation']
                        else:
                            epsilon = function_value

                        # Add inequality constraint
                        self.__add_goal_constraint(
                            goal, epsilon, ensemble_member, options)

                    # Handle path goal function evaluation in a grouped manner to
                    # save time with the call map_path_expression(). Repeated
                    # calls will make repeated CasADi Function objects, which can
                    # be slow.
                    goal_path_functions = OrderedDict()
                    for j, goal in enumerate(path_goals):
                        if (
                                not goal.has_target_bounds or
                                goal.violation_timeseries_id is not None or
                                goal.function_value_timeseries_id is not None
                                ):
                            goal_path_functions[j] = goal.function(self, ensemble_member)

                    expr = self.map_path_expression(ca.vertcat(*goal_path_functions.values()), ensemble_member)
                    f = ca.Function('f', [self.solver_input], [expr])
                    raw_function_values = np.array(f(self.solver_output))
                    goal_function_values = {k: raw_function_values[:, i].ravel()
                                            for i, k in enumerate(goal_path_functions.keys())}

                    for j, goal in enumerate(path_goals):
                        if goal.critical:
                            continue

                        if j in goal_function_values:
                            function_value = goal_function_values[j]

                            # Store results
                            if goal.function_value_timeseries_id is not None:
                                self.set_timeseries(goal.function_value_timeseries_id, function_value, ensemble_member)

                        if goal.has_target_bounds:
                            epsilon = self.__results[ensemble_member][
                                'path_eps_{}_{}'.format(i, j)]

                            # Store results
                            if goal.violation_timeseries_id is not None:
                                epsilon_active = np.copy(epsilon)
                                m = goal.target_min
                                if isinstance(m, Timeseries):
                                    m = self.interpolate(times, goal.target_min.times, goal.target_min.values)
                                M = goal.target_max
                                if isinstance(M, Timeseries):
                                    M = self.interpolate(times, goal.target_max.times, goal.target_max.values)
                                w = np.ones_like(times)
                                if goal.has_target_min:
                                    w = np.logical_and(
                                        w, np.logical_or(
                                            np.logical_not(np.isfinite(m)),
                                            function_value / goal.function_nominal >
                                            m / goal.function_nominal + options['interior_distance']))
                                if goal.has_target_max:
                                    w = np.logical_and(
                                        w, np.logical_or(
                                            np.logical_not(np.isfinite(M)),
                                            function_value / goal.function_nominal <
                                            M / goal.function_nominal - options['interior_distance']))
                                epsilon_active[w] = np.nan
                                self.set_timeseries(goal.violation_timeseries_id, epsilon_active, ensemble_member)

                            # Add a relaxation to appease the barrier method.
                            epsilon += options['constraint_relaxation']
                        else:
                            epsilon = function_value

                        # Add inequality constraint
                        self.__add_path_goal_constraint(
                            goal, epsilon, ensemble_member, options)

        logger.info("Done goal programming")

        # Do post-processing
        if postprocessing:
            self.post()

        # Done
        return success

    def priority_started(self, priority: int) -> None:
        """
        Called when optimization for goals of certain priority is started.

        :param priority: The priority level that was started.
        """
        pass

    def priority_completed(self, priority: int) -> None:
        """
        Called after optimization for goals of certain priority is completed.

        :param priority: The priority level that was completed.
        """
        pass

    def extract_results(self, ensemble_member=0):
        if self.__results_are_current:
            logger.debug("Returning cached results")
            return self.__results[ensemble_member]

        # If self.__results is not up to date, do the super().extract_results
        # method
        return super().extract_results(ensemble_member)
