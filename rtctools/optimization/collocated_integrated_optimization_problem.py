from abc import ABCMeta, abstractmethod
import casadi as ca
import numpy as np
import itertools
import logging

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.casadi_helpers import is_affine, nullvertcat, substitute_in_external, interpolate

from .optimization_problem import OptimizationProblem
from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


class CollocatedIntegratedOptimizationProblem(OptimizationProblem, metaclass=ABCMeta):
    def __init__(self, **kwargs):
        # Variables that will be optimized
        self.dae_variables['free_variables'] = self.dae_variables[
            'states'] + self.dae_variables['algebraics'] + self.dae_variables['control_inputs']

        # Cache names of states
        self.__differentiated_states = [
            variable.name() for variable in self.dae_variables['states']]
        self.__algebraic_states = [variable.name()
                                   for variable in self.dae_variables['algebraics']]
        self.__controls = [variable.name()
                           for variable in self.dae_variables['control_inputs']]

        # DAE cache
        self.__dae_residual_function_collocated = None
        self.__initial_residual_with_params_fun_map = None

        # Create dictionary of variables so that we have O(1) state lookup available
        self.__variables = AliasDict(self.alias_relation)
        for var in itertools.chain(self.dae_variables['states'], self.dae_variables['algebraics'], self.dae_variables['control_inputs'], self.dae_variables['constant_inputs'], self.dae_variables['parameters'], self.dae_variables['time']):
            self.__variables[var.name()] = var

        # Call super
        super().__init__(**kwargs)

    @abstractmethod
    def times(self, variable=None):
        """
        List of time stamps for variable.

        :param variable: Variable name.

        :returns: A list of time stamps for the given variable.
        """
        pass

    def transcribe(self):
        # DAE residual
        dae_residual = self.dae_residual

        # Initial residual
        initial_residual = self.initial_residual

        logger.info("Transcribing problem with a DAE of {} equations, {} collocation points, and {} free variables".format(
            dae_residual.size1(), len(self.times()), len(self.dae_variables['free_variables'])))

        # Reset dictionary of variables
        for var in itertools.chain(self.path_variables, self.extra_variables):
            self.__variables[var.name()] = var

        # Cache path variable names
        self.__path_variable_names = [variable.name()
                                      for variable in self.path_variables]

        # Collocation times
        collocation_times = self.times()
        n_collocation_times = len(collocation_times)

        # Create a store of all ensemble-member-specific data for all ensemble members
        # N.B. Don't use n * [{}], as it creates n refs to the same dict.
        ensemble_store = [{} for i in range(self.ensemble_size)]
        for ensemble_member in range(self.ensemble_size):
            ensemble_data = ensemble_store[ensemble_member]

            # Store parameters
            parameters = self.parameters(ensemble_member)
            parameter_values = [None] * len(self.dae_variables['parameters'])
            for i, symbol in enumerate(self.dae_variables['parameters']):
                variable = symbol.name()
                try:
                    parameter_values[i] = parameters[variable]
                except KeyError:
                    raise Exception(
                        "No value specified for parameter {}".format(variable))

            if np.any([isinstance(value, ca.MX) and not value.is_constant() for value in parameter_values]):
                parameter_values = substitute_in_external(
                    parameter_values, self.dae_variables['parameters'], parameter_values)

            if ensemble_member == 0:
                # Store parameter values of member 0, as variable bounds may depend on these.
                self.__parameter_values_ensemble_member_0 = parameter_values
            ensemble_data["parameters"] = nullvertcat(*parameter_values)

            # Store constant inputs
            constant_inputs = self.constant_inputs(ensemble_member)
            constant_inputs_interpolated = {}
            for variable in self.dae_variables['constant_inputs']:
                variable = variable.name()
                try:
                    constant_input = constant_inputs[variable]
                except KeyError:
                    raise Exception(
                        "No values found for constant input {}".format(variable))
                else:
                    values = constant_input.values
                    if isinstance(values, ca.MX) and not values.is_constant():
                        [values] = substitute_in_external(
                            [values], self.dae_variables['parameters'], parameter_values)
                    elif np.any([isinstance(value, ca.MX) and not value.is_constant() for value in values]):
                        values = substitute_in_external(
                            values, self.dae_variables['parameters'], parameter_values)
                    constant_inputs_interpolated[variable] = self.interpolate(
                        collocation_times, constant_input.times, values, 0.0, 0.0)

            ensemble_data["constant_inputs"] = constant_inputs_interpolated

        # Resolve variable bounds
        bounds = self.bounds()
        bound_keys, bound_values = zip(*bounds.items())
        lb_values, ub_values = zip(*bound_values)
        lb_values = np.array(lb_values, dtype=np.object)
        ub_values = np.array(ub_values, dtype=np.object)
        lb_mx_indices = np.where(
            [isinstance(v, ca.MX) and not v.is_constant() for v in lb_values])
        ub_mx_indices = np.where(
            [isinstance(v, ca.MX) and not v.is_constant() for v in ub_values])
        if len(lb_mx_indices[0]) > 0:
            lb_values[lb_mx_indices] = substitute_in_external(list(
                lb_values[lb_mx_indices]), self.dae_variables['parameters'], self.__parameter_values_ensemble_member_0)
        if len(ub_mx_indices[0]) > 0:
            ub_values[ub_mx_indices] = substitute_in_external(list(
                ub_values[ub_mx_indices]), self.dae_variables['parameters'], self.__parameter_values_ensemble_member_0)
        resolved_bounds = AliasDict(self.alias_relation)
        for i, key in enumerate(bound_keys):
            lb, ub = lb_values[i], ub_values[i]
            resolved_bounds[key] = (float(lb) if isinstance(
                lb, ca.MX) else lb, float(ub) if isinstance(ub, ca.MX) else ub)

        # Initialize control discretization
        control_size, discrete_control, lbx_control, ubx_control, x0_control, indices_control = self.discretize_controls(
            resolved_bounds)

        # Initialize state discretization
        state_size, discrete_state, lbx_state, ubx_state, x0_state, indices_state = self.discretize_states(
            resolved_bounds)

        # Merge state vector offset dictionary
        self.__indices = indices_control
        for ensemble_member in range(self.ensemble_size):
            for key, value in indices_state[ensemble_member].items():
                if isinstance(value, slice):
                    value = slice(value.start + control_size, value.stop + control_size)
                else:
                    value += control_size
                self.__indices[ensemble_member][key] = value

        # Initialize vector of optimization symbols
        X = ca.MX.sym('X', control_size + state_size)
        self.__solver_input = X

        # Initialize bound and seed vectors
        discrete = np.zeros(X.size1(), dtype=np.bool)

        lbx = -np.inf * np.ones(X.size1())
        ubx = np.inf * np.ones(X.size1())

        x0 = np.zeros(X.size1())

        discrete[:len(discrete_control)] = discrete_control
        discrete[len(discrete_control):] = discrete_state
        lbx[:len(lbx_control)] = lbx_control
        lbx[len(lbx_control):] = lbx_state
        ubx[:len(ubx_control)] = ubx_control
        ubx[len(lbx_control):] = ubx_state
        x0[:len(x0_control)] = x0_control
        x0[len(x0_control):] = x0_state

        # Provide a state for self.state_at() and self.der() to work with.
        self.__control_size = control_size
        self.__state_size = state_size
        self.__symbol_cache = {}

        # Free variables for the collocated optimization problem
        collocated_variables = []
        for variable in itertools.chain(self.dae_variables['states'], self.dae_variables['algebraics']):
            collocated_variables.append(variable)
        for variable in self.dae_variables['control_inputs']:
            # TODO treat these separately.
            collocated_variables.append(variable)

        if logger.getEffectiveLevel() == logging.DEBUG:
            logger.debug("Collocating variables {}".format(
                repr(collocated_variables)))

        # Derivatives.
        collocated_derivatives = []
        for k, var in enumerate(self.dae_variables['states']):
            collocated_derivatives.append(
                self.dae_variables['derivatives'][k])
        self.__algebraic_and_control_derivatives = []
        for k, var in enumerate(itertools.chain(self.dae_variables['algebraics'], self.dae_variables['control_inputs'])):
            sym = ca.MX.sym('der({})'.format(var.name()))
            self.__algebraic_and_control_derivatives.append(sym)
            collocated_derivatives.append(sym)

        # Parameter symbols
        symbolic_parameters = ca.vertcat(*self.dae_variables['parameters'])

        # Path objective
        path_objective = self.path_objective(0)

        # Path constraints
        path_constraints = self.path_constraints(0)
        path_constraint_expressions = ca.vertcat(
            *[f_constraint for (f_constraint, lb, ub) in path_constraints])

        # Delayed feedback
        delayed_feedback = self.delayed_feedback()

        # Initial time
        t0 = self.initial_time

        # Set CasADi function options
        options = self.solver_options()
        function_options = {'max_num_dir': options['optimized_num_dir']}

        if self.__dae_residual_function_collocated is None:
            dae_residual_collocated = dae_residual

            # Initialize an Function for the DAE residual (collocated part)
            if len(collocated_variables) > 0:
                self.__dae_residual_function_collocated = ca.Function('dae_residual_function_collocated', [symbolic_parameters, ca.vertcat(*(collocated_variables + collocated_derivatives + self.dae_variables['constant_inputs'] + self.dae_variables['time']))], [dae_residual_collocated], function_options)
        if len(collocated_variables) > 0:
            dae_residual_function_collocated = self.__dae_residual_function_collocated
            dae_residual_collocated_size = dae_residual_function_collocated.mx_out(
                0).size1()
        else:
            dae_residual_collocated_size = 0

        # Initialize an Function for the path objective
        # Note that we assume that the path objective expression is the same for all ensemble members
        path_objective_function = ca.Function('path_objective',
                                              [symbolic_parameters,
                                               ca.vertcat(*(collocated_variables + collocated_derivatives + self.dae_variables[
                                                   'constant_inputs'] + self.dae_variables['time'] + self.path_variables))],
                                              [path_objective], function_options)
        path_objective_function = path_objective_function.expand()

        # Initialize an Function for the path constraints
        # Note that we assume that the path constraint expression is the same for all ensemble members
        path_constraints_function = ca.Function('path_constraints',
                                                [symbolic_parameters,
                                                 ca.vertcat(*(collocated_variables + collocated_derivatives + self.dae_variables[
                                                     'constant_inputs'] + self.dae_variables['time'] + self.path_variables))],
                                                [path_constraint_expressions], function_options)
        path_constraints_function = path_constraints_function.expand()

        # Add constraints for initial conditions
        initial_residual_function = ca.Function('initial_residual_total', [symbolic_parameters, ca.vertcat(*self.dae_variables['states'] + self.dae_variables['algebraics'] + self.dae_variables[
                'control_inputs'] + collocated_derivatives + self.dae_variables['constant_inputs'] + self.dae_variables['time'])], [ca.veccat(dae_residual, initial_residual)],
                function_options)

        # Start collecting constraints
        f = []
        g = []
        lbg = []
        ubg = []

        # Process the objectives and constraints for each ensemble member separately.
        # Note that we don't use map here for the moment, so as to allow each ensemble member to define its own
        # constraints and objectives.  Path constraints are applied for all ensemble members simultaneously
        # at the moment.  We can get rid of map again, and allow every ensemble member to specify its own
        # path constraints as well, once CasADi has some kind of loop detection.
        for ensemble_member in range(self.ensemble_size):
            logger.info(
                "Transcribing ensemble member {}/{}".format(ensemble_member + 1, self.ensemble_size))

            parameters = self.__parameter_values_ensemble_member_0
            constant_inputs = ensemble_store[ensemble_member]["constant_inputs"]

            # Initial conditions specified in history timeseries
            history = self.history(ensemble_member)
            initial_state_constraint_states = []
            initial_state_constraint_values = []
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.controls):
                try:
                    history_timeseries = history[variable]
                except KeyError:
                    pass
                else:
                    sym = self.state_vector(
                        variable, ensemble_member=ensemble_member)[0]
                    initial_state_constraint_states.append(sym)
                    val = self.interpolate(
                        t0, history_timeseries.times, history_timeseries.values, np.nan, np.nan)
                    val /= self.variable_nominal(variable)
                    initial_state_constraint_values.append(val)

            # Call the external metadata function in one go, rather than two
            if len(initial_state_constraint_states) > 0:
                g.extend(initial_state_constraint_states)
                initial_state_constraint_values = substitute_in_external(
                    initial_state_constraint_values, [symbolic_parameters], [parameters])
                lbg.extend(initial_state_constraint_values)
                ubg.extend(initial_state_constraint_values)

            # Collocate initial residual
            initial_state = ca.vertcat(*[self.state_vector(variable.name())[0] for variable in collocated_variables])
            initial_differences = ca.vertcat(*[self.der_at(variable.name(), t0, ensemble_member) for variable in collocated_variables])
            initial_constant_inputs = ca.vertcat(*[constant_inputs[variable.name()][0] for variable in self.dae_variables['constant_inputs']])
            initial_path_variables = ca.vertcat(*[self.state_vector(variable.name())[0] for variable in self.path_variables])

            [initial_residual] = initial_residual_function.call([parameters,
                                                                    ca.vertcat(*(initial_state,
                                                                                initial_differences,
                                                                                initial_constant_inputs,
                                                                                0))],
                                                                    False, True)
            g.append(initial_residual)
            zeros = np.zeros(initial_residual.size())
            lbg.append(zeros)
            ubg.append(zeros)

            # Collocate first and final time step
            assert n_collocation_times == 2

            collocated_states_1 = ca.vertcat(*[self.state_vector(variable.name())[1] for variable in collocated_variables])
            dt = collocation_times[1] - collocation_times[0]
            def diff(var_name):
                v = self.state_vector(var_name)
                return (v[1] - v[0]) / dt
            collocated_finite_differences = ca.vertcat(*[diff(variable.name()) for variable in collocated_variables])
            constant_inputs_1 = ca.vertcat(*[constant_inputs[variable.name()][1] for variable in self.dae_variables['constant_inputs']])
            path_variables_1 = ca.vertcat(*[self.state_vector(variable.name())[1] for variable in self.path_variables])
            collocation_time_1 = self.times()[1]

            [collocation_constraints] = dae_residual_function_collocated.call([parameters,
                                                                    ca.vertcat(*(collocated_states_1,
                                                                                collocated_finite_differences,
                                                                                constant_inputs_1,
                                                                                collocation_time_1 - t0))],
                                                                    False, True)

            [discretized_path_objective] = path_objective_function.call([parameters,
                                                           ca.vertcat(*(collocated_states_1,
                                                                      collocated_finite_differences,
                                                                      constant_inputs_1,
                                                                      collocation_time_1 - t0,
                                                                      path_variables_1))],
                                                          False, True)

            [discretized_path_constraints] = path_constraints_function.call([parameters,
                                                             ca.vertcat(*(collocated_states_1,
                                                                        collocated_finite_differences,
                                                                        constant_inputs_1,
                                                                        collocation_time_1 - t0,
                                                                        path_variables_1))],
                                                            False, True)

            logger.info("Composing NLP segment")

            # Add collocation constraints
            if collocation_constraints.size1() > 0:
                g.append(collocation_constraints)
                zeros = np.zeros(collocation_constraints.size1())
                lbg.extend(zeros)
                ubg.extend(zeros)

            # Objective
            f_member = self.objective(ensemble_member)
            if path_objective.size1() > 0:
                initial_path_objective = path_objective_function.call([parameters,
                                                                       ca.vertcat(initial_state, initial_derivatives, initial_constant_inputs,
                                                                                  0.0,
                                                                                  initial_path_variables)], False, True)
                f_member += initial_path_objective[0] + \
                    ca.sum1(discretized_path_objective)
            f.append(self.ensemble_member_probability(
                ensemble_member) * f_member)

            if logger.getEffectiveLevel() == logging.DEBUG:
                logger.debug(
                    "Adding objective {}".format(f_member))

            # Constraints
            constraints = self.constraints(ensemble_member)
            if logger.getEffectiveLevel() == logging.DEBUG:
                for constraint in constraints:
                    logger.debug(
                        "Adding constraint {}, {}, {}".format(*constraint))

            g_constraint = [f_constraint for (
                f_constraint, lb, ub) in constraints]
            g.extend(g_constraint)

            lbg_constraint = [lb for (f_constraint, lb, ub) in constraints]
            lbg.extend(lbg_constraint)

            ubg_constraint = [ub for (f_constraint, lb, ub) in constraints]
            ubg.extend(ubg_constraint)

            # Path constraints
            # We need to call self.path_constraints() again here, as the bounds may change from ensemble member to member.
            path_constraints = self.path_constraints(ensemble_member)
            if len(path_constraints) > 0:
                # We need to evaluate the path constraints at t0, as the initial time is not included in the accumulation.
                [initial_path_constraints] = path_constraints_function.call([parameters,
                                                                             ca.vertcat(initial_state, initial_derivatives, initial_constant_inputs,
                                                                                        0.0,
                                                                                        initial_path_variables)], False, True)
                g.append(initial_path_constraints)
                g.append(discretized_path_constraints)

                lbg_path_constraints = np.empty(
                    (len(path_constraints), n_collocation_times))
                ubg_path_constraints = np.empty(
                    (len(path_constraints), n_collocation_times))
                for j, path_constraint in enumerate(path_constraints):
                    if logger.getEffectiveLevel() == logging.DEBUG:
                        logger.debug(
                            "Adding path constraint {}, {}, {}".format(*path_constraint))

                    lb = path_constraint[1]
                    if isinstance(lb, ca.MX) and not lb.is_constant():
                        [lb] = ca.substitute(
                            [lb], self.dae_variables['parameters'], self.__parameter_values_ensemble_member_0)
                    elif isinstance(lb, Timeseries):
                        lb = self.interpolate(
                            collocation_times, lb.times, lb.values, -np.inf, -np.inf)

                    ub = path_constraint[2]
                    if isinstance(ub, ca.MX) and not ub.is_constant():
                        [ub] = ca.substitute(
                            [ub], self.dae_variables['parameters'], self.__parameter_values_ensemble_member_0)
                    elif isinstance(ub, Timeseries):
                        ub = self.interpolate(
                            collocation_times, ub.times, ub.values, np.inf, np.inf)

                    lbg_path_constraints[j, :] = lb
                    ubg_path_constraints[j, :] = ub
                lbg.extend(lbg_path_constraints.transpose().ravel())
                ubg.extend(ubg_path_constraints.transpose().ravel())

        # NLP function
        logger.info("Creating NLP dictionary")

        nlp = {'x': X, 'f': ca.sum1(ca.vertcat(*f)), 'g': ca.vertcat(*g)}

        # Done
        logger.info("Done transcribing problem")

        return discrete, lbx, ubx, lbg, ubg, x0, nlp

    def clear_transcription_cache(self):
        """
        Clears the DAE ``Function``s that were cached by ``transcribe``.
        """
        self.__dae_residual_function_collocated = None
        self.__initial_residual_with_params_fun_map = None

    def extract_results(self, ensemble_member=0):
        logger.info("Extracting results")

        # Gather results in a dictionary
        control_results = self.extract_controls(ensemble_member)
        state_results = self.extract_states(ensemble_member)

        # Merge dictionaries
        results = AliasDict(self.alias_relation)
        results.update(control_results)
        results.update(state_results)

        logger.info("Done extracting results")

        # Return results dictionary
        return results

    @property
    def solver_input(self):
        return self.__solver_input

    def solver_options(self):
        options = super(CollocatedIntegratedOptimizationProblem,
                        self).solver_options()
        # Set the option in both cases, to avoid one inadvertently remaining in the cache.
        options['jac_c_constant'] = 'yes'
        return options

    @property
    def controls(self):
        return self.__controls

    def discretize_controls(self, resolved_bounds):
        # Default implementation: One single set of control inputs for all
        # ensembles
        count = 0
        for variable in self.controls:
            times = self.times(variable)
            n_times = len(times)

            count += n_times

        # We assume the seed for the controls to be identical for the entire ensemble.
        # After all, we don't use a stochastic tree if we end up here.
        seed = self.seed(ensemble_member=0)

        indices = [{} for ensemble_member in range(self.ensemble_size)]

        discrete = np.zeros(count, dtype=np.bool)

        lbx = np.full(count, -np.inf, dtype=np.float64)
        ubx = np.full(count, np.inf, dtype=np.float64)

        x0 = np.zeros(count, dtype=np.float64)

        offset = 0
        for variable in self.controls:
            times = self.times(variable)
            n_times = len(times)

            for ensemble_member in range(self.ensemble_size):
                indices[ensemble_member][variable] = slice(offset, offset + n_times)

            discrete[offset:offset +
                     n_times] = self.variable_is_discrete(variable)

            try:
                bound = resolved_bounds[variable]
            except KeyError:
                pass
            else:
                nominal = self.variable_nominal(variable)
                if bound[0] is not None:
                    if isinstance(bound[0], Timeseries):
                        lbx[offset:offset + n_times] = self.interpolate(
                            times, bound[0].times, bound[0].values, -np.inf, -np.inf) / nominal
                    else:
                        lbx[offset:offset + n_times] = bound[0] / nominal
                if bound[1] is not None:
                    if isinstance(bound[1], Timeseries):
                        ubx[offset:offset + n_times] = self.interpolate(
                            times, bound[1].times, bound[1].values, +np.inf, +np.inf) / nominal
                    else:
                        ubx[offset:offset + n_times] = bound[1] / nominal

                try:
                    seed_k = seed[variable]
                    x0[offset:offset + n_times] = self.interpolate(
                        times, seed_k.times, seed_k.values, 0, 0) / nominal
                except KeyError:
                    pass

            offset += n_times

        # Return number of control variables
        return count, discrete, lbx, ubx, x0, indices

    def extract_controls(self, ensemble_member=0):
        # Solver output
        X = self.solver_output

        # Extract control inputs
        results = {}
        offset = 0
        for variable in self.controls:
            n_times = len(self.times(variable))
            results[variable] = np.array(self.variable_nominal(
                variable) * X[offset:offset + n_times, 0]).ravel()
            offset += n_times

        # Done
        return results

    def control_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        # Default implementation: One single set of control inputs for all
        # ensembles
        t0 = self.initial_time
        X = self.solver_input

        canonical, sign = self.alias_relation.canonical_signed(variable)
        offset = 0
        for control_input in self.controls:
            times = self.times(control_input)
            if control_input == canonical:
                nominal = self.variable_nominal(control_input)
                n_times = len(times)
                variable_values = X[offset:offset + n_times]
                f_left, f_right = np.nan, np.nan
                if t < t0:
                    history = self.history(ensemble_member)
                    try:
                        history_timeseries = history[control_input]
                    except KeyError:
                        if extrapolate:
                            sym = variable_values[0]
                        else:
                            sym = np.nan
                    else:
                        if extrapolate:
                            f_left = history_timeseries.values[0]
                            f_right = history_timeseries.values[-1]
                        sym = self.interpolate(
                            t, history_timeseries.times, history_timeseries.values, f_left, f_right)
                    if not scaled and nominal != 1:
                        sym *= nominal
                else:
                    if extrapolate:
                        f_left = variable_values[0]
                        f_right = variable_values[-1]
                    sym = self.interpolate(
                        t, times, variable_values, f_left, f_right)
                    if not scaled and nominal != 1:
                        sym *= nominal
                if sign < 0:
                    sym *= -1
                return sym
            offset += len(times)

        raise KeyError(variable)

    @property
    def differentiated_states(self):
        return self.__differentiated_states

    @property
    def algebraic_states(self):
        return self.__algebraic_states

    def discretize_states(self, resolved_bounds):
        # Default implementation: States for all ensemble members
        ensemble_member_size = 0

        # Space for collocated states
        ensemble_member_size += len(self.differentiated_states + self.algebraic_states + self.__path_variable_names) * len(self.times())

        # Space for extra variables
        ensemble_member_size += len(self.extra_variables)

        # Space for initial states and derivatives
        ensemble_member_size += len(self.dae_variables['derivatives'])

        # Total space requirement
        count = self.ensemble_size * ensemble_member_size

        # Allocate arrays
        indices = [{} for ensemble_member in range(self.ensemble_size)]

        discrete = np.zeros(count, dtype=np.bool)

        lbx = -np.inf * np.ones(count)
        ubx = np.inf * np.ones(count)

        x0 = np.zeros(count)

        # Indices
        for ensemble_member in range(self.ensemble_size):
            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.__path_variable_names):
                times = self.times(variable)
                n_times = len(times)

                indices[ensemble_member][variable] = slice(offset, offset + n_times)

                offset += n_times

            for extra_variable in self.extra_variables:
                indices[ensemble_member][extra_variable.name()] = offset

                offset += 1

        # Types
        for ensemble_member in range(self.ensemble_size):
            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.__path_variable_names):
                times = self.times(variable)
                n_times = len(times)

                discrete[offset:offset +
                            n_times] = self.variable_is_discrete(variable)

                offset += n_times

            for k in range(len(self.extra_variables)):
                discrete[
                    offset + k] = self.variable_is_discrete(self.extra_variables[k].name())

        # Bounds, defaulting to +/- inf, if not set
        for ensemble_member in range(self.ensemble_size):
            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.__path_variable_names):
                times = self.times(variable)
                n_times = len(times)

                try:
                    bound = resolved_bounds[variable]
                except KeyError:
                    pass
                else:
                    nominal = self.variable_nominal(variable)
                    if bound[0] is not None:
                        if isinstance(bound[0], Timeseries):
                            lbx[offset:offset + n_times] = self.interpolate(
                                times, bound[0].times, bound[0].values, -np.inf, -np.inf) / nominal
                        else:
                            lbx[offset:offset + n_times] = bound[0] / nominal
                    if bound[1] is not None:
                        if isinstance(bound[1], Timeseries):
                            ubx[offset:offset + n_times] = self.interpolate(
                                times, bound[1].times, bound[1].values, +np.inf, +np.inf) / nominal
                        else:
                            ubx[offset:offset + n_times] = bound[1] / nominal

                offset += n_times

            for k in range(len(self.extra_variables)):
                try:
                    bound = resolved_bounds[self.extra_variables[k].name()]
                except KeyError:
                    pass
                else:
                    if bound[0] is not None:
                        lbx[offset + k] = bound[0]
                    if bound[1] is not None:
                        ubx[offset + k] = bound[1]

            # Initial guess based on provided seeds, defaulting to zero if no
            # seed is given
            seed = self.seed(ensemble_member)

            offset = ensemble_member * ensemble_member_size
            for variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.__path_variable_names):
                times = self.times(variable)
                n_times = len(times)

                try:
                    seed_k = seed[variable]
                    nominal = self.variable_nominal(variable)
                    x0[offset:offset + n_times] = self.interpolate(
                        times, seed_k.times, seed_k.values, 0, 0) / nominal
                except KeyError:
                    pass

                offset += n_times

            for k in range(len(self.extra_variables)):
                try:
                    x0[offset + k] = seed[self.extra_variables[k].name()]
                except KeyError:
                    pass

        # Return number of state variables
        return count, discrete, lbx, ubx, x0, indices

    def extract_states(self, ensemble_member=0):
        # Solver output
        X = self.solver_output

        # Discretization parameters
        control_size = self.__control_size
        ensemble_member_size = int(self.__state_size / self.ensemble_size)

        # Extract control inputs
        results = {}

        # Extract collocated variables
        offset = control_size + ensemble_member * ensemble_member_size
        for variable in itertools.chain(self.differentiated_states, self.algebraic_states):
            n_times = len(self.times(variable))
            results[variable] = np.array(self.variable_nominal(
                variable) * X[offset:offset + n_times, 0]).ravel()
            offset += n_times

        # Extract constant input aliases
        constant_inputs = self.constant_inputs(ensemble_member)
        for variable in self.dae_variables['constant_inputs']:
            variable = variable.name()
            try:
                constant_input = constant_inputs[variable]
            except KeyError:
                pass
            else:
                results[variable] = np.interp(self.times(
                    variable), constant_input.times, constant_input.values)

        # Extract path variables
        n_collocation_times = len(self.times())
        for variable in self.path_variables:
            variable = variable.name()
            results[variable] = np.array(
                X[offset:offset + n_collocation_times, 0]).ravel()
            offset += n_collocation_times

        # Extract extra variables
        for k in range(len(self.extra_variables)):
            variable = self.extra_variables[k].name()
            results[variable] = np.array(X[offset + k, 0]).ravel()

        # Done
        return results

    def state_vector(self, variable, ensemble_member=0):
        # Look up transcribe_problem() state.
        X = self.solver_input
        control_size = self.__control_size
        ensemble_member_size = int(self.__state_size / self.ensemble_size)

        # Extract state vector
        indices = self.__indices[ensemble_member][variable]
        times = self.times(variable)
        n_times = len(times)
        vector = X[indices]

        return vector

    def state_at(self, variable, t, ensemble_member=0, scaled=False, extrapolate=True):
        if isinstance(variable, ca.MX):
            variable = variable.name()
        name = "{}[{},{}]{}".format(
            variable, ensemble_member, t - self.initial_time, 'S' if scaled else '')
        if extrapolate:
            name += 'E'
        try:
            return self.__symbol_cache[name]
        except KeyError:
            # Look up transcribe_problem() state.
            t0 = self.initial_time
            X = self.solver_input
            control_size = self.__control_size
            ensemble_member_size = int(self.__state_size / self.ensemble_size)

            # Fetch appropriate symbol, or value.
            canonical, sign = self.alias_relation.canonical_signed(variable)
            found = False
            if not found:
                offset = control_size + ensemble_member * ensemble_member_size
                for free_variable in itertools.chain(self.differentiated_states, self.algebraic_states, self.__path_variable_names):
                    if free_variable == canonical:
                        times = self.times(free_variable)
                        n_times = len(times)
                        nominal = self.variable_nominal(free_variable)
                        variable_values = X[offset:offset + n_times]
                        f_left, f_right = np.nan, np.nan
                        if t < t0:
                            history = self.history(ensemble_member)
                            try:
                                history_timeseries = history[free_variable]
                            except KeyError:
                                if extrapolate:
                                    sym = variable_values[0]
                                else:
                                    sym = np.nan
                            else:
                                if extrapolate:
                                    f_left = history_timeseries.values[0]
                                    f_right = history_timeseries.values[-1]
                                sym = self.interpolate(
                                    t, history_timeseries.times, history_timeseries.values, f_left, f_right)
                            if not scaled and nominal != 1:
                                sym *= nominal
                        else:
                            if extrapolate:
                                f_left = variable_values[0]
                                f_right = variable_values[-1]
                            sym = self.interpolate(
                                t, times, variable_values, f_left, f_right)
                            if not scaled and nominal != 1:
                                sym *= nominal
                        if sign < 0:
                            sym *= -1
                        found = True
                        break
                    offset += len(self.times(free_variable))
            if not found:
                try:
                    sym = self.control_at(
                        variable, t, ensemble_member=ensemble_member, extrapolate=extrapolate)
                    found = True
                except KeyError:
                    pass
            if not found:
                constant_inputs = self.constant_inputs(ensemble_member)
                try:
                    constant_input = constant_inputs[variable]
                except KeyError:
                    pass
                else:
                    times = self.times(variable)
                    n_times = len(times)
                    f_left, f_right = np.nan, np.nan
                    if extrapolate:
                        f_left = constant_input.values[0]
                        f_right = constant_input.values[-1]
                    sym = self.interpolate(
                        t, constant_input.times, constant_input.values, f_left, f_right)
            if not found:
                parameters = self.parameters(ensemble_member)
                try:
                    sym = parameters[variable]
                    found = True
                except KeyError:
                    pass
            if not found:
                raise KeyError(variable)

            # Cache symbol.
            self.__symbol_cache[name] = sym

            return sym

    def variable(self, variable):
        return self.__variables[variable]

    def extra_variable(self, extra_variable, ensemble_member=0):
        # Look up transcribe_problem() state.
        X = self.solver_input
        control_size = self.__control_size
        ensemble_member_size = int(self.__state_size / self.ensemble_size)

        # Compute position in state vector
        offset = control_size + ensemble_member * ensemble_member_size
        for variable in itertools.chain(self.differentiated_states, self.algebraic_states):
            n_times = len(self.times(variable))
            offset += n_times

        n_collocation_times = len(self.times())
        for variable in self.path_variables:
            offset += n_collocation_times

        for k in range(len(self.extra_variables)):
            variable = self.extra_variables[k].name()
            if variable == extra_variable:
                return X[offset + k]

        raise KeyError(variable)

    def states_in(self, variable, t0=None, tf=None, ensemble_member=0):
        # Time stamps for this variale
        times = self.times(variable)

        # Set default values
        if t0 is None:
            t0 = times[0]
        if tf is None:
            tf = times[-1]

        # Find canonical variable
        canonical, sign = self.alias_relation.canonical_signed(variable)
        nominal = self.variable_nominal(canonical)
        state = nominal * self.state_vector(canonical, ensemble_member)
        if sign < 0:
            state *= -1

        # Compute combined points
        if t0 < times[0]:
            history = self.history(ensemble_member)
            try:
                history_timeseries = history[canonical]
            except KeyError:
                raise Exception(
                    "No history found for variable {}, but a historical value was requested".format(variable))
            else:
                history_times = history_timeseries.times[:-1]
                history = history_timeseries.values[:-1]
                if sign < 0:
                    history *= -1
        else:
            history_times = np.empty(0)
            history = np.empty(0)

        # Collect states within specified interval
        indices, = np.where(np.logical_and(times >= t0, times <= tf))
        history_indices, = np.where(np.logical_and(
            history_times >= t0, history_times <= tf))
        if (t0 not in times[indices]) and (t0 not in history_times[history_indices]):
            x0 = self.state_at(variable, t0, ensemble_member)
        else:
            x0 = ca.MX()
        if (tf not in times[indices]) and (tf not in history_times[history_indices]):
            xf = self.state_at(variable, tf, ensemble_member)
        else:
            xf = ca.MX()
        x = ca.vertcat(x0, history[history_indices],
                       state[indices[0]:indices[-1] + 1], xf)

        return x

    def integral(self, variable, t0=None, tf=None, ensemble_member=0):
        # Time stamps for this variale
        times = self.times(variable)

        # Set default values
        if t0 is None:
            t0 = times[0]
        if tf is None:
            tf = times[-1]

        # Find canonical variable
        canonical, sign = self.alias_relation.canonical_signed(variable)
        nominal = self.variable_nominal(canonical)
        state = nominal * self.state_vector(canonical, ensemble_member)
        if sign < 0:
            state *= -1

        # Compute combined points
        if t0 < times[0]:
            history = self.history(ensemble_member)
            try:
                history_timeseries = history[canonical]
            except KeyError:
                raise Exception(
                    "No history found for variable {}, but a historical value was requested".format(variable))
            else:
                history_times = history_timeseries.times[:-1]
                history = history_timeseries.values[:-1]
                if sign < 0:
                    history *= -1
        else:
            history_times = np.empty(0)
            history = np.empty(0)

        # Collect time stamps and states, "knots".
        indices, = np.where(np.logical_and(times >= t0, times <= tf))
        history_indices, = np.where(np.logical_and(
            history_times >= t0, history_times <= tf))
        if (t0 not in times[indices]) and (t0 not in history_times[history_indices]):
            x0 = self.state_at(variable, t0, ensemble_member)
        else:
            t0 = x0 = ca.MX()
        if (tf not in times[indices]) and (tf not in history_times[history_indices]):
            xf = self.state_at(variable, tf, ensemble_member)
        else:
            tf = xf = ca.MX()
        t = ca.vertcat(t0, history_times[history_indices], times[indices], tf)
        x = ca.vertcat(x0, history[history_indices],
                       state[indices[0]:indices[-1] + 1], xf)

        # Integrate knots using trapezoid rule
        x_avg = 0.5 * (x[:x.size1() - 1] + x[1:])
        dt = t[1:] - t[:x.size1() - 1]
        return ca.sum1(x_avg * dt)

    def der(self, variable):
        # Look up the derivative variable for the given non-derivative variable
        canonical, sign = self.alias_relation.canonical_signed(variable)
        try:
            i = self.differentiated_states.index(canonical)
            return sign * self.dae_variables['derivatives'][i]
        except ValueError:
            try:
                i = self.algebraic_states.index(canonical)
            except ValueError:
                i = len(self.algebraic_states) + self.controls.index(canonical)
            return sign * self.__algebraic_and_control_derivatives[i]

    def der_at(self, variable, t, ensemble_member=0):
        # Special case t being t0 for differentiated states
        if t == self.initial_time:
            # We have a special symbol for t0 derivatives
            X = self.solver_input
            control_size = self.__control_size
            ensemble_member_size = int(self.__state_size / self.ensemble_size)

            canonical, sign = self.alias_relation.canonical_signed(variable)
            try:
                i = self.differentiated_states.index(canonical)
            except ValueError:
                # Fall through, in case 'variable' is not a differentiated state.
                pass
            else:
                return sign * X[control_size + (ensemble_member + 1) * ensemble_member_size - len(self.dae_variables['derivatives']) + i]

        # Time stamps for this variale
        times = self.times(variable)

        if t <= self.initial_time:
            # Derivative requested for t0 or earlier.  We need the history.
            history = self.history(ensemble_member)
            try:
                htimes = history[variable].times[:-1]
            except KeyError:
                htimes = []
            history_and_times = np.hstack((htimes, times))
        else:
            history_and_times = times

        # Special case t being the initial available point.  In this case, we have
        # no derivative information available.
        if t == history_and_times[0]:
            return 0.0

        # Handle t being an interior point, or t0 for a non-differentiated
        # state
        for i in range(len(history_and_times)):
            # Use finite differences when between collocation points, and
            # backward finite differences when on one.
            if t > history_and_times[i] and t <= history_and_times[i + 1]:
                dx = self.state_at(variable, history_and_times[i + 1], ensemble_member=ensemble_member) - self.state_at(
                    variable, history_and_times[i], ensemble_member=ensemble_member)
                dt = history_and_times[i + 1] - history_and_times[i]
                return dx / dt

        # t does not belong to any collocation point interval
        raise IndexError

    def map_path_expression(self, expr, ensemble_member):
        # Expression as function of states and derivatives
        states = self.dae_variables['states'] + \
            self.dae_variables['algebraics'] + \
            self.dae_variables['control_inputs']
        states_and_path_variables = states + self.path_variables
        derivatives = self.dae_variables['derivatives'] + \
            self.__algebraic_and_control_derivatives

        f = ca.Function('f', [ca.vertcat(*states_and_path_variables), ca.vertcat(*derivatives),
                              ca.vertcat(
                                  *self.dae_variables['constant_inputs']), ca.vertcat(*self.dae_variables['parameters']),
                              self.dae_variables['time'][0]], [expr])
        fmap = f.map(len(self.times()))

        # Discretization settings
        collocation_times = self.times()
        n_collocation_times = len(collocation_times)
        dt = ca.transpose(collocation_times[1:] - collocation_times[:-1])
        t0 = self.initial_time

        # Prepare interpolated state and path variable vectors
        accumulation_states = [None] * len(states_and_path_variables)
        for i, state in enumerate(states_and_path_variables):
            state = state.name()
            times = self.times(state)
            values = self.state_vector(state, ensemble_member)
            if len(times) != n_collocation_times:
                accumulation_states[i] = interpolate(
                    times, values, collocation_times)
            else:
                accumulation_states[i] = values
            nominal = self.variable_nominal(state)
            if nominal != 1:
                accumulation_states[i] *= nominal
        accumulation_states = ca.transpose(ca.horzcat(*accumulation_states))

        # Prepare derivatives (backwards differencing, consistent with the evaluation of path expressions during transcription)
        accumulation_derivatives = [None] * len(derivatives)
        for i, state in enumerate(states):
            state = state.name()
            accumulation_derivatives[i] = ca.horzcat(self.der_at(state, t0, ensemble_member),
                                                     (accumulation_states[i, 1:] - accumulation_states[i, :-1]) / dt)
        accumulation_derivatives = ca.vertcat(*accumulation_derivatives)

        # Prepare constant inputs
        constant_inputs = self.constant_inputs(ensemble_member)
        accumulation_constant_inputs = [None] * \
            len(self.dae_variables['constant_inputs'])
        for i, variable in enumerate(self.dae_variables['constant_inputs']):
            try:
                constant_input = constant_inputs[variable.name()]
            except KeyError:
                raise Exception(
                    "No data specified for constant input {}".format(variable.name()))
            else:
                values = constant_input.values
                if isinstance(values, ca.MX) and not values.is_constant():
                    [values] = substitute_in_external(
                        [values], self.dae_variables['parameters'], self.__parameter_values_ensemble_member_0)
                elif np.any([isinstance(value, ca.MX) and not value.is_constant() for value in values]):
                    values = substitute_in_external(
                        values, self.dae_variables['parameters'], self.__parameter_values_ensemble_member_0)
                accumulation_constant_inputs[i] = self.interpolate(
                    collocation_times, constant_input.times, values, 0.0, 0.0)

        accumulation_constant_inputs = ca.transpose(
            ca.horzcat(*accumulation_constant_inputs))

        # Map
        values = fmap(accumulation_states, accumulation_derivatives,
                      accumulation_constant_inputs, ca.repmat(
                          ca.vertcat(*self.__parameter_values_ensemble_member_0), 1, n_collocation_times),
                      np.transpose(collocation_times))
        return ca.transpose(values)
