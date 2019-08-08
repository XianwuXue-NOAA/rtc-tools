import functools
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


class _EmptyEnsembleList(list):
    """
    An indexable object containing infinitely many empty lists.
    Only to be used as a placeholder.
    """

    def __getitem__(self, key):
        return []


class _EmptyEnsembleOrderedDict(OrderedDict):
    """
    An indexable object containing infinitely many empty OrderedDicts.
    Only to be used as a placeholder.
    """

    def __getitem__(self, key):
        return OrderedDict()


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

    A goal can be written in vector form. In a vector goal:
        * The goal size determines how many goals there are.
        * The goal function has shape ``(goal size, 1)``.
        * The function is either minimized or has, possibly various, targets.
        * Function nominal can either be an array with as many entries as the goal size or have a single value.
        * Function ranges can either be an array with as many entries as the goal size or have a single value.
        * In a goal, the target can either be an array with as many entries as the goal size or have a single value.
        * In a path goal, the target can also be a Timeseries whose values are either a 1-dimensional vector or have
          as many columns as the goal size.

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

    #: The size of the goal if it's a vector goal.
    size = 1

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
            return np.any(np.isfinite(self.target_min))

    @property
    def has_target_max(self) -> bool:
        """
        ``True`` if the user goal has max bounds.
        """
        if isinstance(self.target_max, Timeseries):
            return True
        else:
            return np.any(np.isfinite(self.target_max))

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


class _GoalConstraint:

    def __init__(
            self,
            goal: Goal,
            function: Callable[[OptimizationProblem], ca.MX],
            m: Union[float, np.ndarray, Timeseries],
            M: Union[float, np.ndarray, Timeseries],
            optimized: bool):

        assert isinstance(m, (float, np.ndarray, Timeseries))
        assert isinstance(M, (float, np.ndarray, Timeseries))
        assert type(m) == type(M)

        # NumPy arrays only allowed for vector goals
        if isinstance(m, np.ndarray):
            assert len(m) == goal.size
            assert len(M) == goal.size

        self.goal = goal
        self.function = function
        self.min = m
        self.max = M
        self.optimized = optimized

    def update_bounds(self, other, enforce='self'):
        # NOTE: a.update_bounds(b) is _not_ the same as  b.update_bounds(a).
        # See how the 'enforce' parameter is used.

        min_, max_ = self.min, self.max
        other_min, other_max = other.min, other.max

        if isinstance(min_, Timeseries):
            assert isinstance(max_, Timeseries)
            assert isinstance(other_min, Timeseries)
            assert isinstance(other_max, Timeseries)

            min_ = min_.values
            max_ = max_.values
            other_min = other_min.values
            other_max = other_max.values

        min_ = np.maximum(min_, other_min)
        max_ = np.minimum(max_, other_max)

        # Ensure new constraint bounds do not loosen or shift
        # previous bounds due to numerical errors.
        if enforce == 'self':
            min_ = np.minimum(max_, other_min)
            max_ = np.maximum(min_, other_max)
        else:
            min_ = np.minimum(min_, other_max)
            max_ = np.maximum(max_, other_min)

        # Ensure consistency of bounds. Bounds may become inconsistent due to
        # small numerical computation errors.
        min_ = np.minimum(min_, max_)

        if isinstance(self.min, Timeseries):
            self.min = Timeseries(self.min.times, min_)
            self.max = Timeseries(self.max.times, max_)
        else:
            self.min = min_
            self.max = max_


class GoalProgrammingMixin(OptimizationProblem, metaclass=ABCMeta):
    """
    Adds lexicographic goal programming to your optimization problem.
    """

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        # Initialize instance variables, so that the overridden methods may be
        # called outside of the goal programming loop, for example in pre().
        self.__first_run = True
        self.__results_are_current = False
        self.__subproblem_epsilons = []
        self.__subproblem_objectives = []
        self.__subproblem_soft_constraints = _EmptyEnsembleList()
        self.__subproblem_parameters = []
        self.__constraint_store = _EmptyEnsembleOrderedDict()

        self.__subproblem_path_epsilons = []
        self.__subproblem_path_objectives = []
        self.__subproblem_path_soft_constraints = _EmptyEnsembleList()
        self.__subproblem_path_timeseries = []
        self.__path_constraint_store = _EmptyEnsembleOrderedDict()

        self.__original_parameter_keys = {}
        self.__original_constant_input_keys = {}

        # Lists that are only filled when 'keep_soft_constraints' is True
        self.__problem_constraints = _EmptyEnsembleList()
        self.__problem_path_constraints = _EmptyEnsembleList()
        self.__problem_epsilons = []
        self.__problem_path_epsilons = []
        self.__problem_path_timeseries = []
        self.__problem_parameters = []

    @property
    def extra_variables(self):
        return self.__problem_epsilons + self.__subproblem_epsilons

    @property
    def path_variables(self):
        return self.__problem_path_epsilons + self.__subproblem_path_epsilons

    def bounds(self):
        bounds = super().bounds()
        for epsilon in (self.__subproblem_epsilons + self.__subproblem_path_epsilons +
                        self.__problem_epsilons + self.__problem_path_epsilons):
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

        n_times = len(self.times())

        # Append min/max timeseries to the constant inputs. Note that min/max
        # timeseries are shared between all ensemble members.
        for (variable, value) in self.__subproblem_path_timeseries + self.__problem_path_timeseries:
            if isinstance(value, np.ndarray):
                value = Timeseries(self.times(), np.broadcast_to(value, (n_times, len(value))))
            elif not isinstance(value, Timeseries):
                value = Timeseries(self.times(), np.full(n_times, value))

            constant_inputs[variable] = value
        return constant_inputs

    def parameters(self, ensemble_member):
        parameters = super().parameters(ensemble_member)

        if ensemble_member not in self.__original_parameter_keys:
            self.__original_parameter_keys[ensemble_member] = set(parameters.keys())

        # Remove min/max parameters of previous priorities
        for k in set(parameters.keys()):
            if k not in self.__original_parameter_keys[ensemble_member]:
                del parameters[k]

        # Append min/max values to the parameters. Note that min/max values
        # are shared between all ensemble members.
        for (variable, value) in self.__subproblem_parameters + self.__problem_parameters:
            parameters[variable] = value
        return parameters

    def seed(self, ensemble_member):
        if self.__first_run:
            seed = super().seed(ensemble_member)
        else:
            # Seed with previous results
            seed = AliasDict(self.alias_relation)
            for key, result in self.__results[ensemble_member].items():
                times = self.times(key)
                if ((result.ndim == 1 and len(result) == len(times))
                        or (result.ndim == 2 and result.shape[0] == len(times))):
                    # Only include seed timeseries which are consistent
                    # with the specified time stamps.
                    seed[key] = Timeseries(times, result)
                elif ((result.ndim == 1 and len(result) == 1)
                        or (result.ndim == 2 and result.shape[0] == 1)):
                    seed[key] = result

        # Seed epsilons of current priority
        for epsilon in self.__subproblem_epsilons:
            eps_size = epsilon.size1()
            if eps_size > 1:
                seed[epsilon.name()] = np.ones(eps_size)
            else:
                seed[epsilon.name()] = 1.0

        times = self.times()
        for epsilon in self.__subproblem_path_epsilons:
            eps_size = epsilon.size1()
            if eps_size > 1:
                seed[epsilon.name()] = Timeseries(times, np.ones((eps_size, len(times))))
            else:
                seed[epsilon.name()] = Timeseries(times, np.ones(len(times)))

        return seed

    def __n_objectives(self, subproblem_objectives, subproblem_path_objectives, ensemble_member):
        return ca.vertcat(*[o(self, ensemble_member) for o in subproblem_objectives]).size1() \
               + ca.vertcat(*[o(self, ensemble_member) for o in subproblem_path_objectives]).size1()

    def __objective(self, subproblem_objectives, n_objectives, ensemble_member):
        if len(subproblem_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in subproblem_objectives]))

            if self.goal_programming_options()['scale_by_problem_size']:
                acc_objective = acc_objective / n_objectives

            return acc_objective
        else:
            return ca.MX(0)

    def objective(self, ensemble_member):
        n_objectives = self.__n_objectives(self.__subproblem_objectives, self.__subproblem_path_objectives,
                                           ensemble_member)
        return self.__objective(self.__subproblem_objectives, n_objectives, ensemble_member)

    def __path_objective(self, subproblem_path_objectives, n_objectives, ensemble_member):
        if len(subproblem_path_objectives) > 0:
            acc_objective = ca.sum1(ca.vertcat(*[o(self, ensemble_member) for o in subproblem_path_objectives]))

            if self.goal_programming_options()['scale_by_problem_size']:
                # Objective is already divided by number of active time steps
                # at this point when `scale_by_problem_size` is set.
                acc_objective = acc_objective / n_objectives

            return acc_objective
        else:
            return ca.MX(0)

    def path_objective(self, ensemble_member):
        n_objectives = self.__n_objectives(self.__subproblem_objectives, self.__subproblem_path_objectives,
                                           ensemble_member)
        return self.__path_objective(self.__subproblem_path_objectives, n_objectives, ensemble_member)

    def constraints(self, ensemble_member):
        constraints = super().constraints(ensemble_member)

        additional_constraints = itertools.chain(
            self.__constraint_store[ensemble_member].values(),
            self.__problem_constraints[ensemble_member],
            self.__subproblem_soft_constraints[ensemble_member])

        for constraint in additional_constraints:
            constraints.append((constraint.function(self), constraint.min, constraint.max))

        return constraints

    def path_constraints(self, ensemble_member):
        path_constraints = super().path_constraints(ensemble_member)

        additional_path_constraints = itertools.chain(
            self.__path_constraint_store[ensemble_member].values(),
            self.__problem_path_constraints[ensemble_member],
            self.__subproblem_path_soft_constraints[ensemble_member])

        for constraint in additional_path_constraints:
            path_constraints.append((constraint.function(self), constraint.min, constraint.max))

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
        | ``violation_relaxation``  | ``float`` | ``0.0``       |
        +---------------------------+-----------+---------------+
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
        | ``keep_soft_constraints`` | ``bool``  | ``False``     |
        +---------------------------+-----------+---------------+

        Before turning a soft constraint of the goal programming algorithm into a hard constraint,
        the violation variable (also known as epsilon) of each goal is relaxed with the
        ``violation_relaxation``. Use of this option is normally not required.

        When turning a soft constraint of the goal programming algorithm into a hard constraint,
        the constraint is relaxed with ``constraint_relaxation``. Use of this option is
        normally not required. Note that:

        1. Minimization goals do not get ``constraint_relaxation`` applied when
           ``fix_minimized_values`` is True.

        2. Because of the constraints it generates, when ``keep_soft_constraints`` is True, the option
           ``fix_minimized_values`` needs to be set to False for the ``constraint_relaxation`` to
           be applied at all.

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

        If ``scale_by_problem_size`` is set to ``True``, the objective (i.e. the sum of the violation variables)
        will be divided by the number of goals, and the path objective will be divided by the number
        of path goals and the number of active time steps (per goal). This will make sure the objectives are always in
        the range [0, 1], at the cost of solving each goal/time step less accurately.

        The option ``keep_soft_constraints`` controls how the epsilon variables introduced in the target
        goals are dealt with in subsequent priorities.
        If ``keep_soft_constraints`` is set to False, each epsilon is replaced by its computed value and
        those are used to derive a new set of constraints.
        If ``keep_soft_constraints`` is set to True, the epsilons are kept as variables and the constraints
        are not modified. To ensure the goal programming philosophy, i.e., Pareto optimality, a single
        constraint is added to enforce that the objective function must always be at most the objective
        value. This method allows for a larger solution space, at the cost of having a (possibly) more complex
        optimization problem. Indeed, more variables are kept around throughout the optimization and any
        objective function is turned into a constraint for the subsequent priorities (while in the False
        option this was the case only for the function of minimization goals).

        :returns: A dictionary of goal programming options.
        """

        options = {}

        options['mu_reinit'] = True
        options['violation_relaxation'] = 0.0  # Disable by default
        options['constraint_relaxation'] = 0.0  # Disable by default
        options['violation_tolerance'] = np.inf  # Disable by default
        options['fix_minimized_values'] = False
        options['check_monotonicity'] = True
        options['equality_threshold'] = 1e-8
        options['interior_distance'] = 1e-6
        options['scale_by_problem_size'] = False
        options['keep_soft_constraints'] = False

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

    def __min_max_arrays(self, g, target_shape=None):
        """
        Broadcasts the goal target minimum and target maximum to arrays of a desired target shape.

        Depending on whether g is a vector goal or not, the output shape differs:

        - A 2-D array of size (goal.size, target_shape or 1) if the goal size
          is larger than one, i.e. a vector goal
        - A 1-D array of size (target_shape or 1, ) otherwise
        """

        times = self.times()

        m, M = None, None
        if isinstance(g.target_min, Timeseries):
            m = self.interpolate(
                times, g.target_min.times, g.target_min.values, -np.inf, -np.inf)
            if m.ndim > 1:
                m = m.transpose()
        elif isinstance(g.target_min, np.ndarray) and target_shape:
            m = np.broadcast_to(g.target_min, (target_shape, g.size)).transpose()
        elif target_shape:
            m = np.full(target_shape, g.target_min)
        else:
            m = np.array([g.target_min]).transpose()
        if isinstance(g.target_max, Timeseries):
            M = self.interpolate(
                times, g.target_max.times, g.target_max.values, np.inf, np.inf)
            if M.ndim > 1:
                M = M.transpose()
        elif isinstance(g.target_max, np.ndarray) and target_shape:
            M = np.broadcast_to(g.target_max, (target_shape, g.size)).transpose()
        elif target_shape:
            M = np.full(target_shape, g.target_max)
        else:
            M = np.array([g.target_max]).transpose()

        if g.size > 1 and m.ndim == 1:
            m = np.broadcast_to(m, (g.size, len(m)))
        if g.size > 1 and M.ndim == 1:
            M = np.broadcast_to(M, (g.size, len(M)))

        if g.size > 1:
            assert m.shape == (g.size, 1 if target_shape is None else target_shape)
        else:
            assert m.shape == (1 if target_shape is None else target_shape, )
        assert m.shape == M.shape

        return m, M

    def __validate_goals(self, goals, is_path_goal):
        goals = sorted(goals, key=lambda x: x.priority)

        options = self.goal_programming_options()

        # Validate goal definitions
        for goal in goals:
            m, M = goal.function_range

            # The function range should not be a symbolic expression
            if isinstance(m, ca.MX):
                assert m.is_constant()
                if m.size1() == 1:
                    m = float(m)
                else:
                    m = np.array(m.to_DM())

            if isinstance(M, ca.MX):
                assert M.is_constant()
                if M.size1() == 1:
                    M = float(M)
                else:
                    M = np.array(M.to_DM())

            assert isinstance(m, (float, int, np.ndarray))
            assert isinstance(M, (float, int, np.ndarray))

            if np.any(goal.function_nominal <= 0):
                raise Exception("Nonpositive nominal value specified for goal {}".format(goal))

            if goal.critical and not goal.has_target_bounds:
                raise Exception("Minimization goals cannot be critical")

            if goal.critical:
                # Allow a function range for backwards compatibility reasons.
                # Maybe raise a warning that its not actually used?
                pass
            elif goal.has_target_bounds:
                if not np.all(np.isfinite(m)) or not np.all(np.isfinite(M)):
                    raise Exception("No function range specified for goal {}".format(goal))

                if np.any(m >= M):
                    raise Exception("Invalid function range for goal {}.".format(goal))

                if goal.weight <= 0:
                    raise Exception("Goal weight should be positive for goal {}".format(goal))
            else:
                if goal.function_range != (np.nan, np.nan):
                    raise Exception("Specifying function range not allowed for goal {}".format(goal))

            if not is_path_goal:
                if isinstance(goal.target_min, Timeseries):
                    raise Exception("Target min cannot be a Timeseries for goal {}".format(goal))
                if isinstance(goal.target_max, Timeseries):
                    raise Exception("Target max cannot be a Timeseries for goal {}".format(goal))

            try:
                int(goal.priority)
            except ValueError:
                raise Exception("Priority of not int or castable to int for goal {}".format(goal))

            if options['keep_soft_constraints']:
                if goal.relaxation != 0.0:
                    raise Exception("Relaxation not allowed with `keep_soft_constraints` for goal {}".format(goal))
                if goal.violation_timeseries_id is not None:
                    raise Exception(
                        "Violation timeseries id not allowed with `keep_soft_constraints` for goal {}".format(goal))
            else:
                if goal.size > 1:
                    raise Exception("Option `keep_soft_constraints` needs to be set for vector goal {}".format(goal))

            if goal.critical and goal.size > 1:
                raise Exception("Vector goal cannot be critical for goal {}".format(goal))

        if is_path_goal:
            target_shape = len(self.times())
        else:
            target_shape = None

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
                        goal_m, goal_M = self.__min_max_arrays(goal, target_shape)
                        other_m, other_M = self.__min_max_arrays(prev, target_shape)

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
            goal_m, goal_M = self.__min_max_arrays(goal, target_shape)
            goal_lb = np.broadcast_to(goal.function_range[0], goal_m.shape[::-1]).transpose()
            goal_ub = np.broadcast_to(goal.function_range[1], goal_M.shape[::-1]).transpose()

            if goal.has_target_min and goal.has_target_max:
                indices = np.where(np.logical_not(np.logical_or(
                    np.isnan(goal_m), np.isnan(goal_M))))

                if np.any(goal_m[indices] > goal_M[indices]):
                    raise Exception("Target minimum exceeds target maximum for goal {}".format(goal))

            if goal.has_target_min and not goal.critical:
                indices = np.where(np.isfinite(goal_m))
                if np.any(goal_m[indices] <= goal_lb[indices]):
                    raise Exception(
                        'Target minimum should be greater than the lower bound of the function range for goal {}'
                        .format(goal))
                if np.any(goal_m[indices] > goal_ub[indices]):
                    raise Exception(
                        'Target minimum should be smaller than the upper bound of the function range for goal {}'
                        .format(goal))
            if goal.has_target_max and not goal.critical:
                indices = np.where(np.isfinite(goal_M))
                if np.any(goal_M[indices] >= goal_ub[indices]):
                    raise Exception(
                        'Target maximum should be smaller than the upper bound of the function range for goal {}'
                        .format(goal))
                if np.any(goal_M[indices] < goal_lb[indices]):
                    raise Exception(
                        'Target maximum should be larger than the lower bound of the function range for goal {}'
                        .format(goal))

            if goal.relaxation < 0.0:
                raise Exception('Relaxation of goal {} should be a nonnegative value'.format(goal))

    def __goal_constraints(self, goals, sym_index, options, is_path_goal):
        """
        There are three ways in which a goal turns into objectives/constraints:

        1. A goal with target bounds results in a part for the objective (the
           violation variable), and 1 or 2 constraints (target min, max, or both).
        2. A goal without target bounds (i.e. minimization goal) results in just a
           part for the objective.
        3. A critical goal results in just a (pair of) constraint(s). These are hard
           constraints, which need to be put in the constraint store to guarantee
           linear independence.
        """

        epsilons = []
        objectives = []
        soft_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        hard_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        extra_constants = []

        eps_format = "eps_{}_{}"
        min_format = "min_{}_{}"
        max_format = "max_{}_{}"

        if is_path_goal:
            eps_format = "path_" + eps_format
            min_format = "path_" + min_format
            max_format = "path_" + max_format

        for j, goal in enumerate(goals):
            if goal.critical:
                assert goal.size == 1, "Critical goals cannot be vector goals"
                epsilon = np.zeros(len(self.times()) if is_path_goal else 1)
            elif goal.has_target_bounds:
                epsilon = ca.MX.sym(eps_format.format(sym_index, j), goal.size)
                epsilons.append(epsilon)

            # Make symbols for the target bounds (if set)
            if goal.has_target_min:
                min_variable = min_format.format(sym_index, j)

                # NOTE: When using a vector goal, we want to be sure that its constraints
                # and objective end up _exactly_ equal to its non-vector equivalent. We
                # therefore have to get rid of any superfluous/trivial constraints that
                # would otherwise be generated by the vector goal.
                target_min_slice_inds = np.full(goal.size, True)

                if isinstance(goal.target_min, Timeseries):
                    target_min = Timeseries(goal.target_min.times, goal.target_min.values)
                    inds = np.logical_or(np.isnan(target_min.values), np.isneginf(target_min.values))
                    target_min.values[inds] = -sys.float_info.max
                    n_times = len(goal.target_min.times)
                    target_min_slice_inds = ~np.all(np.broadcast_to(inds.transpose(), (goal.size, n_times)), axis=1)
                elif isinstance(goal.target_min, np.ndarray):
                    target_min = goal.target_min.copy()
                    inds = np.logical_or(np.isnan(target_min), np.isneginf(target_min))
                    target_min[inds] = -sys.float_info.max
                    target_min_slice_inds = ~inds
                else:
                    target_min = goal.target_min

                extra_constants.append((min_variable, target_min))
            else:
                min_variable = None

            if goal.has_target_max:
                max_variable = max_format.format(sym_index, j)

                target_max_slice_inds = np.full(goal.size, True)

                if isinstance(goal.target_max, Timeseries):
                    target_max = Timeseries(goal.target_max.times, goal.target_max.values)
                    inds = np.logical_or(np.isnan(target_max.values), np.isposinf(target_max.values))
                    target_max.values[inds] = sys.float_info.max
                    n_times = len(goal.target_max.times)
                    target_max_slice_inds = ~np.all(np.broadcast_to(inds.transpose(), (goal.size, n_times)), axis=1)
                elif isinstance(goal.target_max, np.ndarray):
                    target_max = goal.target_max.copy()
                    inds = np.logical_or(np.isnan(target_max), np.isposinf(target_max))
                    target_max[inds] = sys.float_info.max
                    target_max_slice_inds = ~inds
                else:
                    target_max = goal.target_max

                extra_constants.append((max_variable, target_max))
            else:
                max_variable = None

            # Make objective for soft constraints and minimization goals
            if not goal.critical:
                if goal.has_target_bounds:
                    if is_path_goal and options['scale_by_problem_size']:
                        goal_m, goal_M = self.__min_max_arrays(goal, target_shape=len(self.times()))
                        goal_active = np.isfinite(goal_m) | np.isfinite(goal_M)
                        n_active = np.sum(goal_active.astype(int), axis=-1)
                        # Avoid possible division by zero if goal is inactive
                        n_active = np.maximum(n_active, 1)
                    else:
                        n_active = 1

                    def _objective_func(problem, ensemble_member,
                                        goal=goal, epsilon=epsilon, is_path_goal=is_path_goal,
                                        n_active=n_active):
                        if is_path_goal:
                            epsilon = problem.variable(epsilon.name())
                        else:
                            epsilon = problem.extra_variable(epsilon.name(), ensemble_member)

                        return goal.weight * ca.constpow(epsilon, goal.order) / n_active
                else:
                    if is_path_goal and options['scale_by_problem_size']:
                        n_active = len(self.times())
                    else:
                        n_active = 1

                    def _objective_func(problem, ensemble_member, goal=goal, is_path_goal=is_path_goal,
                                        n_active=n_active):
                        f = goal.function(problem, ensemble_member) / goal.function_nominal
                        return goal.weight * ca.constpow(f, goal.order) / n_active

                objectives.append(_objective_func)

            # Make constraints for goals with target bounds
            if goal.has_target_bounds:
                if goal.critical:
                    for ensemble_member in range(self.ensemble_size):
                        constraint = self.__goal_hard_constraint(
                            goal, epsilon, None, ensemble_member, options, is_path_goal)
                        hard_constraints[ensemble_member].append(constraint)
                else:
                    for ensemble_member in range(self.ensemble_size):
                        # We use a violation variable formulation, with the violation
                        # variables epsilon bounded between 0 and 1.
                        def _soft_constraint_func(problem, target, bound, inds,
                                                  goal=goal, epsilon=epsilon, ensemble_member=ensemble_member,
                                                  is_path_constraint=is_path_goal):
                            if is_path_constraint:
                                target = problem.variable(target)
                                eps = problem.variable(epsilon.name())
                            else:
                                target = problem.parameters(ensemble_member)[target]
                                eps = problem.extra_variable(epsilon.name(), ensemble_member)

                            inds = inds.nonzero()[0].astype(int).tolist()

                            f = goal.function(problem, ensemble_member)
                            nominal = goal.function_nominal

                            return ca.if_else(ca.fabs(target) < sys.float_info.max,
                                              (f - eps * (bound - target) - target) / nominal,
                                              0.0)[inds]

                        if goal.has_target_min and np.any(target_min_slice_inds):
                            _f = functools.partial(
                                _soft_constraint_func,
                                target=min_variable,
                                bound=goal.function_range[0],
                                inds=target_min_slice_inds)
                            constraint = _GoalConstraint(goal, _f, 0.0, np.inf, False)
                            soft_constraints[ensemble_member].append(constraint)
                        if goal.has_target_max and np.any(target_max_slice_inds):
                            _f = functools.partial(
                                _soft_constraint_func,
                                target=max_variable,
                                bound=goal.function_range[1],
                                inds=target_max_slice_inds)
                            constraint = _GoalConstraint(goal, _f, -np.inf, 0.0, False)
                            soft_constraints[ensemble_member].append(constraint)

        return epsilons, objectives, soft_constraints, hard_constraints, extra_constants

    def __goal_hard_constraint(self, goal, epsilon, existing_constraint, ensemble_member, options, is_path_goal):
        if not is_path_goal:
            epsilon = epsilon[:1]

        goal_m, goal_M = self.__min_max_arrays(goal, target_shape=epsilon.shape[0])

        if goal.has_target_bounds:
            # We use a violation variable formulation, with the violation
            # variables epsilon bounded between 0 and 1.
            m, M = np.full_like(epsilon, -np.inf, dtype=np.float64), np.full_like(epsilon, np.inf, dtype=np.float64)

            # A function range does not have to be specified for critical
            # goals. Avoid multiplying with NaN in that case.
            if goal.has_target_min:
                m = (epsilon * ((goal.function_range[0] - goal_m) if not goal.critical else 0.0)
                     + goal_m - goal.relaxation) / goal.function_nominal
            if goal.has_target_max:
                M = (epsilon * ((goal.function_range[1] - goal_M) if not goal.critical else 0.0)
                     + goal_M + goal.relaxation) / goal.function_nominal

            if goal.has_target_min and goal.has_target_max:
                # Avoid comparing with NaN
                inds = ~(np.isnan(m) | np.isnan(M))
                inds[inds] &= np.abs(m[inds] - M[inds]) < options['equality_threshold']
                if np.any(inds):
                    avg = 0.5 * (m + M)
                    m[inds] = M[inds] = avg[inds]

            m[~np.isfinite(goal_m)] = -np.inf
            M[~np.isfinite(goal_M)] = np.inf

            inds = epsilon > options['violation_tolerance']
            if np.any(inds):
                if is_path_goal:
                    expr = self.map_path_expression(goal.function(self, ensemble_member), ensemble_member)
                else:
                    expr = goal.function(self, ensemble_member)

                function = ca.Function('f', [self.solver_input], [expr])
                value = np.array(function(self.solver_output))

                m[inds] = (value - goal.relaxation) / goal.function_nominal
                M[inds] = (value + goal.relaxation) / goal.function_nominal

            m -= options['constraint_relaxation']
            M += options['constraint_relaxation']
        else:
            # Epsilon encodes the position within the function range.
            if options['fix_minimized_values'] and goal.relaxation == 0.0:
                m = epsilon / goal.function_nominal
                M = epsilon / goal.function_nominal
                self.check_collocation_linearity = False
                self.linear_collocation = False
            else:
                m = -np.inf * np.ones(epsilon.shape)
                M = (epsilon + goal.relaxation) / goal.function_nominal + options['constraint_relaxation']

        if is_path_goal:
            m = Timeseries(self.times(), m)
            M = Timeseries(self.times(), M)
        else:
            m = m[0]
            M = M[0]

        constraint = _GoalConstraint(
            goal,
            lambda problem, ensemble_member=ensemble_member, goal=goal: (
                goal.function(problem, ensemble_member) / goal.function_nominal),
            m, M, True)

        # Epsilon is fixed. Override previous {min,max} constraints for this
        # state.
        if existing_constraint:
            constraint.update_bounds(existing_constraint, enforce='other')

        return constraint

    def __update_constraint_store(self, constraint_store, constraints):
        for ensemble_member in range(self.ensemble_size):
            for other in constraints[ensemble_member]:
                fk = other.goal.get_function_key(self, ensemble_member)
                try:
                    constraint_store[ensemble_member][fk].update_bounds(other)
                except KeyError:
                    constraint_store[ensemble_member][fk] = other

    def __soft_to_hard_constraints(self, goals, sym_index, is_path_goal):
        if is_path_goal:
            constraint_store = self.__path_constraint_store
        else:
            constraint_store = self.__constraint_store

        times = self.times()
        options = self.goal_programming_options()

        eps_format = "eps_{}_{}"
        if is_path_goal:
            eps_format = "path_" + eps_format

        # Handle function evaluation in a grouped manner to save time with
        # the call map_path_expression(). Repeated calls will make
        # repeated CasADi Function objects, which can be slow.
        goal_function_values = [None] * self.ensemble_size

        for ensemble_member in range(self.ensemble_size):
            goal_functions = OrderedDict()

            for j, goal in enumerate(goals):
                if (
                        not goal.has_target_bounds or
                        goal.violation_timeseries_id is not None or
                        goal.function_value_timeseries_id is not None
                        ):
                    goal_functions[j] = goal.function(self, ensemble_member)

            if is_path_goal:
                expr = self.map_path_expression(
                    ca.vertcat(*goal_functions.values()), ensemble_member
                )
            else:
                expr = ca.transpose(ca.vertcat(*goal_functions.values()))

            f = ca.Function('f', [self.solver_input], [expr])
            raw_function_values = np.array(f(self.solver_output))
            goal_function_values[ensemble_member] = {
                k: raw_function_values[:, j].ravel()
                for j, k in enumerate(goal_functions.keys())
            }

        # Re-add constraints, this time with epsilon values fixed
        for ensemble_member in range(self.ensemble_size):
            for j, goal in enumerate(goals):
                if j in goal_function_values[ensemble_member]:
                    function_value = goal_function_values[ensemble_member][j]

                    # Store results
                    if goal.function_value_timeseries_id is not None:
                        self.set_timeseries(
                            goal.function_value_timeseries_id,
                            Timeseries(times, function_value),
                            ensemble_member
                        )

                if goal.critical:
                    continue

                if goal.has_target_bounds:
                    epsilon = self.__results[ensemble_member][
                        eps_format.format(sym_index, j)]

                    # Store results
                    if goal.violation_timeseries_id is not None:
                        function_value = goal_function_values[ensemble_member][j]
                        epsilon_active = np.copy(epsilon)
                        m = goal.target_min
                        if isinstance(m, Timeseries):
                            m = self.interpolate(times, goal.target_min.times, goal.target_min.values)
                        M = goal.target_max
                        if isinstance(M, Timeseries):
                            M = self.interpolate(times, goal.target_max.times, goal.target_max.values)
                        w = np.ones_like(function_value)
                        if goal.has_target_min:
                            # Avoid comparing with NaN while making sure that
                            # w[i] is True when m[i] is not finite.
                            m = np.array(m)
                            m[~np.isfinite(m)] = -np.inf
                            w = np.logical_and(w, (function_value / goal.function_nominal >
                                                   m / goal.function_nominal + options['interior_distance']))
                        if goal.has_target_max:
                            # Avoid comparing with NaN while making sure that
                            # w[i] is True when M[i] is not finite.
                            M = np.array(M)
                            M[~np.isfinite(M)] = np.inf
                            w = np.logical_and(w, (function_value / goal.function_nominal <
                                                   M / goal.function_nominal + options['interior_distance']))
                        epsilon_active[w] = np.nan
                        self.set_timeseries(
                            goal.violation_timeseries_id,
                            Timeseries(times, epsilon_active),
                            ensemble_member
                        )

                    # Add a relaxation to appease the barrier method.
                    epsilon += options['violation_relaxation']
                else:
                    epsilon = function_value

                fk = goal.get_function_key(self, ensemble_member)
                existing_constraint = constraint_store[ensemble_member].get(fk, None)

                constraint_store[ensemble_member][fk] = self.__goal_hard_constraint(
                    goal, epsilon, existing_constraint, ensemble_member, options, is_path_goal)

    def __add_subproblem_objective_constraint(self):
        # We want to keep the additional variables/parameters we set around
        self.__problem_epsilons.extend(self.__subproblem_epsilons)
        self.__problem_path_epsilons.extend(self.__subproblem_path_epsilons)
        self.__problem_path_timeseries.extend(self.__subproblem_path_timeseries)
        self.__problem_parameters.extend(self.__subproblem_parameters)

        for ensemble_member in range(self.ensemble_size):
            self.__problem_constraints[ensemble_member].extend(
                self.__subproblem_soft_constraints[ensemble_member])
            self.__problem_path_constraints[ensemble_member].extend(
                self.__subproblem_path_soft_constraints[ensemble_member])

        # Extract information about the objective value, this is used for the Pareto optimality constraint.
        # We only retain information about the objective functions defined through the goal framework as user
        # define objective functions may relay on local variables.
        subproblem_objectives = self.__subproblem_objectives.copy()
        subproblem_path_objectives = self.__subproblem_path_objectives.copy()

        def _constraint_func(problem,
                             subproblem_objectives=subproblem_objectives,
                             subproblem_path_objectives=subproblem_path_objectives):
            val = 0.0
            for ensemble_member in range(problem.ensemble_size):
                # NOTE: Users might be overriding objective() and/or path_objective(). Use the
                # private methods that work only on the goals.
                n_objectives = problem.__n_objectives(
                    subproblem_objectives, subproblem_path_objectives, ensemble_member)
                expr = problem.__objective(subproblem_objectives, n_objectives, ensemble_member)
                expr += ca.sum1(problem.map_path_expression(
                    problem.__path_objective(subproblem_path_objectives, n_objectives, ensemble_member),
                    ensemble_member))
                val += problem.ensemble_member_probability(ensemble_member) * expr

            return val

        f = ca.Function('tmp', [self.solver_input], [_constraint_func(self)])
        obj_val = float(f(self.solver_output))

        options = self.goal_programming_options()

        if options['fix_minimized_values']:
            constraint = _GoalConstraint(None, _constraint_func, obj_val, obj_val, True)
            self.check_collocation_linearity = False
            self.linear_collocation = False
        else:
            obj_val += options['constraint_relaxation']
            constraint = _GoalConstraint(None, _constraint_func, -np.inf, obj_val, True)

        # The goal works over all ensemble members, so we add it to the last
        # one, as at that point the inputs of all previous ensemble members
        # will have been discretized, mapped and stored.
        self.__problem_constraints[-1].append(constraint)

    def optimize(self, preprocessing=True, postprocessing=True, log_solver_failure_as_error=True):
        # Do pre-processing
        if preprocessing:
            self.pre()

        # Group goals into subproblems
        subproblems = []
        goals = self.goals()
        path_goals = self.path_goals()

        options = self.goal_programming_options()

        # Validate (in)compatible options
        if options['keep_soft_constraints'] and options['violation_relaxation']:
            raise Exception("The option 'violation_relaxation' cannot be used when 'keep_soft_constraints' is set.")

        # Validate goal definitions
        self.__validate_goals(goals, is_path_goal=False)
        self.__validate_goals(path_goals, is_path_goal=True)

        priorities = {int(goal.priority) for goal in itertools.chain(goals, path_goals) if not goal.is_empty}

        for priority in sorted(priorities):
            subproblems.append((
                priority,
                [goal for goal in goals if int(goal.priority) == priority and not goal.is_empty],
                [goal for goal in path_goals if int(goal.priority) == priority and not goal.is_empty]))

        # Solve the subproblems one by one
        logger.info("Starting goal programming")

        success = False

        self.__constraint_store = [OrderedDict() for ensemble_member in range(self.ensemble_size)]
        self.__path_constraint_store = [OrderedDict() for ensemble_member in range(self.ensemble_size)]

        # Lists for when `keep_soft_constraints` is True
        self.__problem_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_epsilons = []
        self.__problem_parameters = []
        self.__problem_path_constraints = [[] for ensemble_member in range(self.ensemble_size)]
        self.__problem_path_epsilons = []
        self.__problem_path_timeseries = []

        self.__first_run = True
        self.__results_are_current = False
        self.__original_constant_input_keys = {}
        self.__original_parameter_keys = {}
        for i, (priority, goals, path_goals) in enumerate(subproblems):
            logger.info("Solving goals at priority {}".format(priority))

            # Call the pre priority hook
            self.priority_started(priority)

            (self.__subproblem_epsilons, self.__subproblem_objectives,
             self.__subproblem_soft_constraints, hard_constraints,
             self.__subproblem_parameters) = \
                self.__goal_constraints(goals, i, options, is_path_goal=False)

            (self.__subproblem_path_epsilons, self.__subproblem_path_objectives,
             self.__subproblem_path_soft_constraints, path_hard_constraints,
             self.__subproblem_path_timeseries) = \
                self.__goal_constraints(path_goals, i, options, is_path_goal=True)

            # Put hard constraints in the constraint stores
            self.__update_constraint_store(self.__constraint_store, hard_constraints)
            self.__update_constraint_store(self.__path_constraint_store, path_hard_constraints)

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

            if options['keep_soft_constraints']:
                self.__add_subproblem_objective_constraint()
            else:
                self.__soft_to_hard_constraints(goals, i, is_path_goal=False)
                self.__soft_to_hard_constraints(path_goals, i, is_path_goal=True)

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
