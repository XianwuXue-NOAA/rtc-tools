import itertools
import logging
import sys
from abc import ABCMeta, abstractmethod
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

    A goal is defined by overriding the :func:`function` method, and setting at least the
    ``function_range`` class variable.

    :cvar function_range:   Range of goal function.  *Required*.
    :cvar function_nominal: Nominal value of function. Used for scaling.  Default is ``1``.
    :cvar target_min:       Desired lower bound for goal function.  Default is ``numpy.nan``.
    :cvar target_max:       Desired upper bound for goal function.  Default is ``numpy.nan``.
    :cvar priority:         Integer priority of goal.  Default is ``1``.
    :cvar weight:           Optional weighting applied to the goal.  Default is ``1.0``.
    :cvar order:            Penalization order of goal violation.  Default is ``2``.
    :cvar critical:         If ``True``, the algorithm will abort if this goal cannot be fully met.
                            Default is ``False``.

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

    Goals with the same priority are weighted off against each other in a
    single objective function.

    In the minimization goals, the function nominal is used to scale the function value in the objective function.

    The goal violation value is taken to the order'th power in the objective function of the final
    optimization problem.

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
        return '{}(priority={}, state={}, target_min={}, target_max={})'.format(
            self.__class__, self.priority, self.state, self.target_min, self.target_max)


class GoalProgrammingMixin(OptimizationProblem, metaclass=ABCMeta):
    """
    Adds lexicographic goal programming to your optimization problem.
    """

    # TODO: optimized boolean is still necessary?
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
        self.__problem_epsilons = []
        self.__problem_path_epsilons = []
        self.__subproblem_epsilons = []
        self.__subproblem_path_epsilons = []
        self.__subproblem_path_timeseries = []
        self.__subproblem_objectives = []
        self.__subproblem_constraints = []
        self.__subproblem_path_constraints = []
        self.__old_subproblem_objectives = []
        self.__old_subproblem_path_objectives = []

        self.__problem_epsilons_alias = []
        self.__problem_path_epsilons_alias = []
        self.__subproblem_epsilons_alias = []
        self.__subproblem_path_epsilons_alias = []
        self.__problem_constraint_epsilons_alias = []
        self.__problem_path_constraint_epsilons_alias = []

        self.__original_constant_input_keys = {}

    @property
    def extra_variables(self):
        return self.__problem_epsilons + self.__problem_epsilons_alias

    @property
    def path_variables(self):
        return self.__problem_path_epsilons.copy() + self.__problem_path_epsilons_alias.copy()

    def bounds(self):
        bounds = super().bounds()
        for epsilon in (self.__problem_epsilons + self.__problem_path_epsilons
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

        # Seed of one for each newly introduced epsilon
        for epsilon in self.__subproblem_epsilons + self.__subproblem_epsilons_alias:
            seed[epsilon.name()] = 1.0

        times = self.times()
        for epsilon in self.__subproblem_path_epsilons + self.__subproblem_path_epsilons_alias:
            seed[epsilon.name()] = Timeseries(times, np.ones(len(times)))

        return seed

    def objective(self, ensemble_member):
        if len(self.__subproblem_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in self.__subproblem_objectives]))

            if self.goal_programming_options()['scale_by_problem_size']:
                acc_objective = acc_objective / len(self.__subproblem_objectives)

            return acc_objective
        else:
            return ca.MX(0)

    def __path_objective(self, ensemble_member):
        if len(self.__subproblem_path_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in self.__subproblem_path_objectives]))

            if self.goal_programming_options()['scale_by_problem_size']:
                acc_objective = acc_objective / len(self.__subproblem_objectives) / len(self.times())

            return acc_objective
        else:
            return ca.MX(0)

    def path_objective(self, ensemble_member):
        return self.__path_objective(ensemble_member)

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)
        for constraint in self.__subproblem_constraints[ensemble_member]:
            constraints.append((constraint.function(self, ensemble_member), constraint.min, constraint.max))
        # Pareto optimality constraint for goals at previous priorities
        if ensemble_member == 0:
            for old_obj, val in self.__old_subproblem_objectives:
                expr = self.ensemble_member_probability(ensemble_member) * \
                       ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in old_obj])) / len(old_obj)
                for iter_ens_memb in range(1, self.ensemble_size):
                    expr += self.ensemble_member_probability(iter_ens_memb) * \
                            ca.sum1(ca.vertcat(*[o(self, iter_ens_memb) for o in old_obj])) / len(old_obj)
                if expr.is_constant():
                    pass
                else:
                    constraints.append((expr - val, -np.inf, 0.0))
        # Pareto optimality constraint for path goals at previous priorities
        if ensemble_member == 0:
            for old_obj, val in self.__old_subproblem_path_objectives:
                expr = self.ensemble_member_probability(ensemble_member) * \
                       ca.sum1(self.map_path_expression(old_obj, ensemble_member))
                for iter_ens_memb in range(1, self.ensemble_size):
                    expr += self.ensemble_member_probability(iter_ens_memb) * \
                            ca.sum1(self.map_path_expression(old_obj, ensemble_member))
                if expr.is_constant():
                    pass
                else:
                    constraints.append((expr - val, -np.inf, 0.0))
        if self.goal_programming_options()['linear_obj_eps']:
            # Epsilon alias constraints
            for constraint_eps in self.__problem_constraint_epsilons_alias[ensemble_member]:
                constraints.append(constraint_eps(self, ensemble_member), -np.inf, 0.0)
        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)
        for constraint in self.__subproblem_path_constraints[ensemble_member]:
            path_constraints.append((constraint.function(self, ensemble_member), constraint.min, constraint.max))
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

        if not self.goal_programming_options()['mu_reinit']:
            ipopt_options['mu_strategy'] = 'monotone'
            ipopt_options['gather_stats'] = True
            if not self.__first_run:
                ipopt_options['mu_init'] = self.solver_stats['iterations']['mu'][-1]

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
        | ``check_monotonicity``    | ``bool``  | ``True``      |
        +---------------------------+-----------+---------------+
        | ``equality_threshold``    | ``float`` | ``1e-8``      |
        +---------------------------+-----------+---------------+
        | ``scale_by_problem_size`` | ``bool``  | ``False``     |
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

        If ``check_monotonicity`` is set to ``True``, then it will be checked whether goals with the same
        function key form a monotonically decreasing sequence with regards to the target interval.

        The option ``equality_threshold`` controls when a two-sided inequality constraint is folded into
        an equality constraint.

        If ``scale_by_problem_size`` is set to ``True``, the objective (i.e. the sum of epsilons)
        will be divided by the number of goals, and the path objective will be divided by the number
        of path goals and the number of time steps. This will make sure the objectives are always in
        the range [0, 1], at the cost of solving each goal/time step less accurately.

        If ``linear_obj_eps`` is set to True, the objective funtion of the optimization problems
        will be linear throughout all the priorities. This option linearizes the objective function
        of target goals with order larger than one by introducing alias epsilon variables. It can only
        be used when all the minimization goal have order one.

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options['mu_reinit'] = True
        options['constraint_relaxation'] = 0.0  # Disable by default
        options['violation_tolerance'] = np.inf  # Disable by default
        options['check_monotonicity'] = True
        options['equality_threshold'] = 1e-8
        options['scale_by_problem_size'] = False
        options['linear_obj_eps'] = False

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

    def __add_goal_constraint(self, goal, epsilon, ensemble_member, options):
        constraints = self.__subproblem_constraints[ensemble_member]

        if goal.critical:
            m, M = -np.inf, np.inf
            if goal.has_target_min:
                if np.isfinite(goal.target_min):
                    m = goal.target_min
                elif np.isfinite(goal.target_max):
                    M = goal.target_max
            m = (m - goal.relaxation) / goal.function_nominal
            M = (M + goal.relaxation) / goal.function_nominal
            constraint = self.__GoalConstraint(
                goal, lambda problem, ensemble_member=ensemble_member, goal=goal:
                goal.function(problem, ensemble_member) / goal.function_nominal, m, M, True)
            constraints.append(constraint)

        else:
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self.__GoalConstraint(
                        goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon:
                        (goal.function(problem, ensemble_member) -
                         problem.extra_variable(epsilon.name(), ensemble_member=ensemble_member) *
                         (goal.function_range[0] - goal.target_min) - goal.target_min) / goal.function_nominal,
                        0.0, np.inf, False)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self.__GoalConstraint(
                        goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon:
                        (goal.function(problem, ensemble_member) -
                         problem.extra_variable(epsilon.name(), ensemble_member=ensemble_member) *
                         (goal.function_range[1] - goal.target_max) - goal.target_max) / goal.function_nominal,
                        -np.inf, 0.0, False)
                    constraints.append(constraint)

    def __add_path_goal_constraint(self, goal, epsilon, ensemble_member, options, min_series=None, max_series=None):

        constraints = self.__subproblem_path_constraints[ensemble_member]

        goal_m, goal_M = self.__min_max_arrays(goal)

        times = self.times()

        if goal.critical:
            # TODO: think whether this may lead to KKT condition issues
            m, M = np.full_like(times, -np.inf, dtype=np.float64), np.full_like(times, np.inf, dtype=np.float64)
            for i, _ in enumerate(times):
                if np.isfinite(goal_m[i]):
                    m[i] = goal_m[i]
                else:
                    m[i] = goal.function_range[0]
                if np.isfinite(goal_M[i]):
                    M[i] = goal_M[i]
                else:
                    M[i] = goal.function_range[1]
                if np.isfinite(goal_m[i]) and np.isfinite(goal_M[i]):
                    if abs(m[i] - M[i]) < options['equality_threshold']:
                        avg = 0.5 * (m[i] + M[i])
                        m[i] = M[i] = avg
                m[i] = (m[i] - goal.relaxation) / goal.function_nominal
                M[i] = (M[i] + goal.relaxation) / goal.function_nominal
            constraint = self.__GoalConstraint(
                goal, lambda problem, ensemble_member=ensemble_member, goal=goal:
                goal.function(problem, ensemble_member) / goal.function_nominal,
                Timeseries(times, m), Timeseries(times, M), True)
            constraints.append(constraint)
        else:
            if goal.has_target_bounds:
                # We use a violation variable formulation, with the violation
                # variables epsilon bounded between 0 and 1.
                if goal.has_target_min:
                    constraint = self.__GoalConstraint(
                        goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon:
                        ca.if_else(problem.variable(min_series) > -sys.float_info.max,
                                   (goal.function(problem, ensemble_member) - problem.variable(epsilon.name()) *
                                    (goal.function_range[0] - problem.variable(min_series)) -
                                    problem.variable(min_series)) / goal.function_nominal, 0.0), 0.0, np.inf, False)
                    constraints.append(constraint)
                if goal.has_target_max:
                    constraint = self.__GoalConstraint(
                        goal, lambda problem, ensemble_member=ensemble_member, goal=goal, epsilon=epsilon:
                        ca.if_else(problem.variable(max_series) < sys.float_info.max,
                                   (goal.function(problem, ensemble_member) - problem.variable(epsilon.name()) *
                                    (goal.function_range[1] - problem.variable(max_series)) -
                                    problem.variable(max_series)) / goal.function_nominal, 0.0), -np.inf, 0.0, False)
                    constraints.append(constraint)

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do pre-processing
        if preprocessing:
            self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        # Validate goal definitions
        for goal in itertools.chain(goals, path_goals):
            m, M = goal.function_range

            # The function range should not be a symbolic expression
            assert (not isinstance(m, ca.MX) or m.is_constant())
            assert (not isinstance(M, ca.MX) or M.is_constant())

            m, M = float(m), float(M)

            if not np.isfinite(m) or not np.isfinite(M):
                raise Exception("No function range specified for goal {}".format(goal))

            if m >= M:
                raise Exception("Invalid function range for goal {}.".format(goal))

            if goal.function_nominal <= 0:
                raise Exception("Nonpositive nominal value specified for goal {}".format(goal))
        try:
            priorities = {int(goal.priority) for goal in itertools.chain(goals, path_goals) if not goal.is_empty}
        except ValueError:
            raise Exception("GoalProgrammingMixin: All goal priorities must be of type int or castable to int")

        for priority in sorted(priorities):
            subproblems.append((
                priority,
                [goal for goal in goals if int(goal.priority) == priority and not goal.is_empty],
                [goal for goal in path_goals if int(goal.priority) == priority and not goal.is_empty]))

        options = self.goal_programming_options()

        # Check consistency and monotonicity of goals. Scalar target min/max
        # of normal goals are also converted to arrays to unify checks with
        # path goals.
        for gs in (goals, path_goals):
            sorted_goals = sorted(gs, key=lambda x: x.priority)

            if options['check_monotonicity']:
                for e in range(self.ensemble_size):
                    # Store the previous goal of a certain function key we
                    # encountered, such that we can compare to it.
                    fk_goal_map = {}

                    for goal in sorted_goals:
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

            for goal in sorted_goals:
                goal_m, goal_M = self.__min_max_arrays(goal)

                if goal.has_target_min and goal.has_target_max:
                    indices = np.where(np.logical_not(np.logical_or(
                        np.isnan(goal_m), np.isnan(goal_M))))

                    if np.any(goal_m[indices] > goal_M[indices]):
                        raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

                if goal.has_target_min:
                    indices = np.where(np.logical_not(np.isnan(goal_m)))
                    if np.any(goal_m[indices] < goal.function_range[0]):
                        raise Exception(
                            'Target minimum is smaller than the lower bound of the function range for goal {}'.format(
                                goal))
                if goal.has_target_max:
                    indices = np.where(np.logical_not(np.isnan(goal_M)))
                    if np.any(goal_M[indices] > goal.function_range[1]):
                        raise Exception(
                            'Target maximum is greater than the upper bound of the function range for goal {}'.format(
                                goal))

        # Solve the subproblems one by one
        logger.info("Starting goal programming")

        success = False

        self.__subproblem_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__subproblem_path_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__first_run = True
        self.__results_are_current = False
        self.__original_constant_input_keys = {}
        self.__subproblem_path_timeseries = []
        # Growing list of epsilons
        self.__problem_epsilons = []
        self.__problem_path_epsilons = []
        # Growing list of objective functions from the previous priorities
        self.__old_subproblem_objectives = []
        self.__old_subproblem_path_objectives = []
        # Growing list of epsilon alias variables
        if self.goal_programming_options()['linear_obj_eps']:
            self.__problem_constraint_epsilons_alias = [[] for ensemble_member in range(self.ensemble_size)]
            self.__problem_path_constraint_epsilons_alias = [[] for ensemble_member in range(self.ensemble_size)]

        for i, (priority, goals, path_goals) in enumerate(subproblems):
            logger.info("Solving goals at priority {}".format(priority))

            # Call the pre priority hook
            self.priority_started(priority)

            # Reset objective function
            self.__subproblem_objectives = []
            self.__subproblem_path_objectives = []
            # Reset list of epsilon variables of the current priority. Used to provide the correct seed
            self.__subproblem_epsilons = []
            self.__subproblem_path_epsilons = []
            if self.goal_programming_options()['linear_obj_eps']:
                self.__subproblem_epsilons_alias = []
                self.__subproblem_path_epsilons_alias = []

            # For each goal at the current priority, we add constraints and update the objective function
            for j, goal in enumerate(goals):
                if goal.critical:
                    if not goal.has_target_bounds:
                        raise Exception("Minimization goals cannot be critical")
                    epsilon = 0.0
                else:
                    if goal.has_target_bounds:
                        epsilon = ca.MX.sym('eps_{}_{}'.format(i, j))
                        self.__problem_epsilons.append(epsilon)
                        self.__subproblem_epsilons.append(epsilon)
                        if self.goal_programming_options()['linear_obj_eps'] and goal.order != 1:
                            epsilon_alias = ca.MX.sym('eps_alias_{}_{}'.format(i, j))
                            self.__problem_epsilons_alias.append(epsilon_alias)
                            self.__subproblem_epsilons_alias.append(epsilon_alias)

                if not goal.critical:
                    if goal.has_target_bounds:
                        if self.goal_programming_options()['linear_obj_eps'] and goal.order != 1:
                            # If self.goal_programming_options()['linear_obj_eps'] is set to True
                            # and the order is not one, than the objective function is the linear
                            # sum of the epsilon alias variables
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
                        if self.goal_programming_options()['linear_obj_eps'] and goal.order != 1:
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
                        if (self.goal_programming_options()['linear_obj_eps']
                                and not goal.critical and goal.order != 1):
                            # If 'linear_obj_eps' is set to True, the order is not one and the goal is not critical,
                            # for each epsilon we add the constraint epsilon^order <= epsilon_alias
                            self.__problem_constraint_epsilons_alias[ensemble_member].append(
                                lambda problem, ensemble_member, goal=goal,
                                epsilon=epsilon, epsilon_alias=epsilon_alias: (ca.constpow(
                                    problem.extra_variable(epsilon.name(), ensemble_member=ensemble_member),
                                    goal.order) - problem.extra_variable(epsilon_alias.name(),
                                                                         ensemble_member=ensemble_member)))

            # For each path goal at the current priority, we add constraints and update the objective function
            for j, goal in enumerate(path_goals):
                if goal.critical:
                    if not goal.has_target_bounds:
                        raise Exception("Minimization goals cannot be critical")
                    epsilon = np.zeros(len(self.times()))
                else:
                    if goal.has_target_bounds:
                        epsilon = ca.MX.sym('path_eps_{}_{}'.format(i, j))
                        self.__problem_path_epsilons.append(epsilon)
                        self.__subproblem_path_epsilons.append(epsilon)
                        if self.goal_programming_options()['linear_obj_eps'] and goal.order != 1:
                            epsilon_alias = ca.MX.sym('path_eps_alias_{}_{}'.format(i, j))
                            self.__problem_path_epsilons_alias.append(epsilon_alias)
                            self.__subproblem_path_epsilons_alias.append(epsilon_alias)

                if goal.has_target_min:
                    min_series = 'path_min_{}_{}'.format(i, j)

                    if isinstance(goal.target_min, Timeseries):
                        target_min = Timeseries(goal.target_min.times, goal.target_min.values)
                        target_min.values[np.logical_or(np.isnan(target_min.values),
                                                        np.isneginf(target_min.values))] = -sys.float_info.max
                    else:
                        target_min = goal.target_min

                    self.__subproblem_path_timeseries.append(
                        (min_series, target_min))
                else:
                    min_series = None
                if goal.has_target_max:
                    max_series = 'path_max_{}_{}'.format(i, j)

                    if isinstance(goal.target_max, Timeseries):
                        target_max = Timeseries(goal.target_max.times, goal.target_max.values)
                        target_max.values[np.logical_or(np.isnan(target_max.values),
                                                        np.isposinf(target_max.values))] = sys.float_info.max
                    else:
                        target_max = goal.target_max

                    self.__subproblem_path_timeseries.append(
                        (max_series, target_max))
                else:
                    max_series = None

                if not goal.critical:
                    if goal.has_target_bounds:
                        if self.goal_programming_options()['linear_obj_eps'] and goal.order != 1:
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
                        if self.goal_programming_options()['linear_obj_eps'] and goal.order != 1:
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
                        if (self.goal_programming_options()['linear_obj_eps']
                                and not goal.critical and goal.order != 1):
                            # If 'linear_obj_eps' is set to True, the order is not one and the goal is not critical,
                            # for each epsilon we add the constraint epsilon^order <= epsilon_alias
                            self.__problem_path_constraint_epsilons_alias[ensemble_member].append(
                                lambda problem, ensemble_member, goal=goal,
                                epsilon=epsilon, epsilon_alias=epsilon_alias:
                                (ca.constpow(problem.variable(epsilon.name()), goal.order) -
                                 problem.variable(epsilon_alias.name())))

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

            # Extract information about the objective value, this is used for the Pareto optimality constraint.
            # We only retain information about the objective functions defined through the goal framework as user
            # define objective functions may relay on local variables.
            if len(self.__subproblem_objectives) > 0:
                for ensemble_member in range(self.ensemble_size):
                    if ensemble_member == 0:
                        expr = self.ensemble_member_probability(ensemble_member) * \
                               ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in self.__subproblem_objectives])) \
                               / len(self.__subproblem_objectives)
                    else:
                        expr += self.ensemble_member_probability(ensemble_member) * \
                                ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in self.__subproblem_objectives]))\
                                / len(self.__subproblem_objectives)
                f = ca.Function('tmp', [self.solver_input], [expr])
                val = float(f(self.solver_output))
                # Add a relaxation to avoid infeasibility issues arising from tolerance settings
                val += options['constraint_relaxation']
                self.__old_subproblem_objectives.append((self.__subproblem_objectives.copy(), val))

            # Extract information about the objective value, this is used for the Pareto optimality constraint.
            if len(self.__subproblem_path_objectives) > 0:
                for ensemble_member in range(self.ensemble_size):
                    if ensemble_member == 0:
                        expr = self.ensemble_member_probability(ensemble_member) * ca.sum1(self.map_path_expression(
                            self.__path_objective(ensemble_member), ensemble_member))
                    else:
                        expr += self.ensemble_member_probability(ensemble_member) * ca.sum1(self.map_path_expression(
                            self.__path_objective(ensemble_member), ensemble_member))
                f = ca.Function('tmp', [self.solver_input], [expr])
                val = float(f(self.solver_output))
                # Add a relaxation to avoid infeasibility issues arising from tolerance settings
                val += options['constraint_relaxation']
                self.__old_subproblem_path_objectives.append((self.__path_objective(ensemble_member), val))

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
