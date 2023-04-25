import logging
from abc import ABCMeta, abstractmethod, abstractproperty
from typing import Any, Dict, Iterator, List, Tuple, Union
import pickle
import os, shutil
import textwrap
import copy

import casadi as ca

import numpy as np

from rtctools._internal.alias_tools import AliasDict
from rtctools._internal.debug_check_helpers import DebugLevel, debug_check
from rtctools.data.storage import DataStoreAccessor

from .timeseries import Timeseries

logger = logging.getLogger("rtctools")


# Typical type for a bound on a variable
BT = Union[float, np.ndarray, Timeseries]

def casadi_to_lp(ps_i):
    try:
        with open(ps_i, 'rb') as f:
            d = pickle.load(f)

        pickleid= os.path.basename(ps_i).split('.')[0].rsplit('_',1)[1]

        indices = d['indices'][0]
        expand_f_g = d['func']
        lbx, ubx, lbg, ubg, x0 = d['other']
        X = ca.SX.sym('X', expand_f_g.nnz_in())
        f, g = expand_f_g(X)

        in_var = X
        out = []
        for o in [f, g]:
            Af = ca.Function('Af', [in_var], [ca.jacobian(o, in_var)])
            bf = ca.Function('bf', [in_var], [o])

            A = Af(0)
            A = ca.sparsify(A)

            b = bf(0)
            b = ca.sparsify(b)
            out.append((A, b))

        var_names = []
        for k, v in indices.items():
            if isinstance(v, int):
                var_names.append('{}__{}'.format(k, v))
            else:
                for i in range(0, v.stop - v.start, 1 if v.step is None else v.step):
                    var_names.append('{}__{}'.format(k, i))

        n_derivatives = expand_f_g.nnz_in() - len(var_names)
        for i in range(n_derivatives):
            var_names.append("DERIVATIVE__{}".format(i))


        # CPLEX does not like [] in variable names
        import re
        for i, v in enumerate(var_names):
            v = v.replace("[", "_I")
            v = v.replace("]", "I_")
            var_names[i] = v

        # OBJECTIVE
        try:
            A, b = out[0]
            objective = []
            ind = np.array(A)[0, :]

            for v, c in zip(var_names, ind):
                if c != 0:
                    objective.extend(['+' if c > 0 else '-', str(abs(c)), v])

            if objective[0] == "-":
                objective[1] = "-" + objective[1]

            objective.pop(0)
            objective_str = " ".join(objective)
            objective_str = "  " + objective_str
        except:
            print("set objective string to 1")
            objective_str = "1"

        # CONSTRAINTS
        A, b = out[1]
        ca.veccat(*lbg)
        lbg = np.array(ca.veccat(*lbg))[:, 0]
        ubg = np.array(ca.veccat(*ubg))[:, 0]


        A_csc = A.tocsc()
        A_coo = A_csc.tocoo()
        b = np.array(b)[:, 0]

        constraints = [[] for i in range(A.shape[0])]

        for i, j, c in zip(A_coo.row, A_coo.col, A_coo.data):
            constraints[i].extend(['+' if c > 0 else '-', str(abs(c)), var_names[j]])


        constraints_original = copy.deepcopy(constraints)
        for i in range(len(constraints)):
            cur_constr = constraints[i]
            l, u, b_i = lbg[i], ubg[i], b[i]

            if len(cur_constr) > 0:
                if cur_constr[0] == "-":
                    cur_constr[1] = "-" + cur_constr[1]
                cur_constr.pop(0)

            c_str = " ".join(cur_constr)

            if np.isfinite(l) and np.isfinite(u) and l == u:
                constraints[i] = "{} = {}".format(c_str, l - b_i)
            elif np.isfinite(l) and np.isfinite(u):
                constraints[i] = "{} <= {} <= {}".format(l - b_i, c_str, u - b_i)
            elif np.isfinite(l):
                constraints[i] = "{} >= {}".format(c_str, l - b_i)
            elif np.isfinite(u):
                constraints[i] = "{} <= {}".format(c_str, u - b_i)
            else:
                raise Exception(l, b, constraints[i])

        constraints_str = "  " + "\n  ".join(constraints)

        # Bounds
        bounds = []
        for v, l, u in zip(var_names, lbx, ubx):
            bounds.append("{} <= {} <= {}".format(l, v, u))
        bounds_str = "  " + "\n  ".join(bounds)

        with open("myproblem_{}.lp".format(pickleid), 'w') as o:
            o.write("Minimize\n")
            for x in textwrap.wrap(objective_str, width=255):  # lp-format has max length of 255 chars
                o.write(x + "\n")
        #    o.write(objective_str + "\n")
            o.write("Subject To\n")
            o.write(constraints_str + "\n")
            o.write("Bounds\n")
            o.write(bounds_str + "\n")
            o.write("End")
        with open("constraints.lp", 'w') as o:
            o.write(constraints_str + "\n")

        shutil.copy("myproblem_{}.lp".format(pickleid), "myproblem.lp")
        nrows = A_coo.shape[0]

        ratios = []
        minmaxs = []

        # shutil.copy("myproblem.lp", r"C:\myproblem.lp")

        #for i in range(nrows):
        #    d = np.abs(A_coo.getrow(i).data)
        #    m, M = min(d), max(d)
        #    minmaxs.append((m, M))
        #    ratios.append(abs(M/m))

        #print(max(ratios))

        return constraints, constraints_original, list(var_names)
    except:
        print("failed!")

class LookupTable:
    """
    Base class for LookupTables.
    """

    @property
    def inputs(self) -> List[ca.MX]:
        """
        List of lookup table input variables.
        """
        raise NotImplementedError

    @property
    def function(self) -> ca.Function:
        """
        Lookup table CasADi :class:`Function`.
        """
        raise NotImplementedError


class OptimizationProblem(DataStoreAccessor, metaclass=ABCMeta):
    """
    Base class for all optimization problems.
    """

    _debug_check_level = DebugLevel.MEDIUM
    _debug_check_options = {}

    def __init__(self, **kwargs):
        # Call parent class first for default behaviour.
        super().__init__(**kwargs)

        self.__mixed_integer = False

    def optimize(self, preprocessing: bool = True, postprocessing: bool = True,
                 log_solver_failure_as_error: bool = True) -> bool:
        """
        Perform one initialize-transcribe-solve-finalize cycle.

        :param preprocessing:  True to enable a call to ``pre`` preceding the optimization.
        :param postprocessing: True to enable a call to ``post`` following the optimization.

        :returns: True on success.
        """

        # Deprecations / removals
        if hasattr(self, 'initial_state'):
            raise RuntimeError("Support for `initial_state()` has been removed. Please use `history()` instead.")

        logger.info("Entering optimize()")

        # Do any preprocessing, which may include changing parameter values on
        # the model
        if preprocessing:
            self.pre()

            # Check if control inputs are bounded
            self.__check_bounds_control_input()
        else:
            logger.debug(
                'Skipping Preprocessing in OptimizationProblem.optimize()')

        # Transcribe problem
        discrete, lbx, ubx, lbg, ubg, x0, nlp = self.transcribe()

        # Create an NLP solver
        logger.debug("Collecting solver options")

        self.__mixed_integer = np.any(discrete)
        options = {}
        options.update(self.solver_options())  # Create a copy

        logger.debug("Creating solver")

        if options.pop('expand', False) or True:
            # NOTE: CasADi only supports the "expand" option for nlpsol. To
            # also be able to expand with e.g. qpsol, we do the expansion
            # ourselves here.
            logger.debug("Expanding objective and constraints to SX")

            expand_f_g = ca.Function('f_g', [nlp['x']], [nlp['f'], nlp['g']]).expand()
            X_sx = ca.SX.sym('X', *nlp['x'].shape)
            f_sx, g_sx = expand_f_g(X_sx)

            nlp['f'] = f_sx
            nlp['g'] = g_sx
            nlp['x'] = X_sx

            import pickle

            import time

            expand_f_g = ca.Function('f_g', [nlp['x']], [nlp['f'], nlp['g']]).expand()

            pickle_name = "nlp_func_{}.pickle".format(int(time.time()))
            with open(pickle_name, 'wb') as pck:

                myd = {}

                myd['indices'] = self._CollocatedIntegratedOptimizationProblem__indices

                myd['func'] = expand_f_g

                myd['other'] = (lbx, ubx, lbg, ubg, x0)


                in_var = ca.SX.sym('X', expand_f_g.nnz_in())
                bf = ca.Function('bf', [in_var], [expand_f_g(in_var)[1]])
                b = bf(0)
                b = ca.sparsify(b)
                b = np.array(b)[:, 0]

                pickle.dump(myd, pck)

            constraints, constraints_original, variable_names = casadi_to_lp(pickle_name)

        # Debug check for non-linearity in constraints
        self.__debug_check_linearity_constraints(nlp)

        # Debug check for linear independence of the constraints
        self.__debug_check_linear_independence(lbx, ubx, lbg, ubg, nlp)

        # Solver option
        my_solver = options['solver']
        del options['solver']

        # Already consumed
        del options['optimized_num_dir']

        # Iteration callback
        iteration_callback = options.pop('iteration_callback', None)

        # CasADi solver to use
        casadi_solver = options.pop('casadi_solver')
        if isinstance(casadi_solver, str):
            casadi_solver = getattr(ca, casadi_solver)

        nlpsol_options = {**options}

        if self.__mixed_integer:
            nlpsol_options['discrete'] = discrete
        if iteration_callback:
            nlpsol_options['iteration_callback'] = iteration_callback

        # Remove ipopt and bonmin defaults if they are not used
        if my_solver != 'ipopt':
            nlpsol_options.pop('ipopt', None)
        if my_solver != 'bonmin':
            nlpsol_options.pop('bonmin', None)

        solver = casadi_solver('nlp', my_solver, nlp, nlpsol_options)

        # Solve NLP
        logger.info("Calling solver")

        results = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=ca.veccat(*lbg), ubg=ca.veccat(*ubg))

        # Extract relevant stats
        self.__objective_value = float(results['f'])
        self.__solver_output = np.array(results['x']).ravel()
        self.__solver_stats = solver.stats()

        success, log_level = self.solver_success(self.__solver_stats, log_solver_failure_as_error)

        return_status = self.__solver_stats['return_status']
        if 'secondary_return_status' in self.__solver_stats:
            return_status = "{}: {}".format(return_status, self.__solver_stats['secondary_return_status'])
        wall_clock_time = "elapsed time not read"
        if 't_wall_total' in self.__solver_stats:
            wall_clock_time = "{} seconds".format(self.__solver_stats['t_wall_total'])
        elif 't_wall_solver' in self.__solver_stats:
            wall_clock_time = "{} seconds".format(self.__solver_stats['t_wall_solver'])

        if success:
            logger.log(log_level, "Solver succeeded with status {} ({}).".format(
                return_status, wall_clock_time))
        else:
            try:
                ii = [y[0] for y in self.loop_over_error].index(self.priority)
                loop_error_indicator = self.loop_over_error[ii][1]
                try:
                    loop_error = self.loop_over_error[ii][2]
                    if loop_error_indicator and loop_error in return_status:
                        log_level = logging.INFO
                except IndexError:
                    if loop_error_indicator:
                        log_level = logging.INFO
                logger.log(log_level, "Solver succeeded with status {} ({}).".format(
                    return_status, wall_clock_time))
            except (AttributeError, ValueError):
                logger.log(log_level, "Solver succeeded with status {} ({}).".format(
                    return_status, wall_clock_time))

        # You can evaluate the constraints wrt to the optimized solution
        x_optimized = np.array(results['x']).ravel()
        expand_f_g = ca.Function('f_g', [nlp['x']], [nlp['f'], nlp['g']]).expand()
        X_sx = ca.SX.sym('X', *nlp['x'].shape)
        f_sx, g_sx = expand_f_g(X_sx)
        eval_g = ca.Function('g_eval', [X_sx], [g_sx]).expand()
        evaluated_g = [x[0] for x in np.array(eval_g(x_optimized))]
        lam_g = [x[0] for x in np.array(results['lam_g'])]
        lam_x = [x[0] for x in np.array(results['lam_x'])]


        # -------------------------------------------------- OLD ------------------------------------------------
        atol = 1e-7
        rtol = 1e-7
        ubg_hits = [np.allclose(evaluated_i,ubg_i-b_i,rtol=rtol,atol=atol) for evaluated_i, ubg_i, b_i in zip(evaluated_g, ubg, b)]
        lbg_hits = [np.allclose(evaluated_i,lbg_i-b_i,rtol=rtol,atol=atol) for evaluated_i, lbg_i, b_i in zip(evaluated_g, lbg, b)]

        violates_ubg = [evaluated_i > ((ubg_i-b_i)*(1+rtol)+atol) for evaluated_i, ubg_i, b_i in zip(evaluated_g, ubg, b)]
        violates_lbg = [evaluated_i < ((lbg_i-b_i)*(1-rtol)-atol) for evaluated_i, lbg_i, b_i in zip(evaluated_g, lbg, b)]

        if any(violates_ubg):
            print("Violation of upper bound!")
        if any(violates_lbg):
            print("Violation of lbg!")

        hit_type = [1*lbg_hit + 2*ubg_hit for lbg_hit, ubg_hit in zip(lbg_hits, ubg_hits)]
        hit_type_str = {0: "", 1: "only lower bound", 2: "only upper bound", 3: "both bounds"}
        smaller_than_zero = 0
        larger_than_zero = 0

        # # DEBUGGING:
        # for i in range(0,len(evaluated_g)):
        #     if hit_type[i]:
        #         if lam_g[i] < 0:
        #             smaller_than_zero +=1
        #             print(f"smaller THAN ZERO! {lam_g[i]}")
        #         else:
        #             print(f"larger  THAN ZERO: {lam_g[i]}!")
        #             larger_than_zero +=1
        #         print(f"-> Constraint {i} is active ({hit_type_str[hit_type[i]]}): {lbg[i]} < {round(evaluated_g[i],100)} < {ubg[i]} (lam_g: {lam_g[i]})")

        #     else:
        #         # continue
        #         print(f"Constraint {i} is not active: {lbg[i]} < {round(evaluated_g[i],100)} < {ubg[i]} (lam_g: {lam_g[i]})")
        #     if lam_g[i] < -1e-5:
        #         if hit_type[i] == 1 or hit_type[i] == 3:
        #             print("Yes")
        #         else:
        #             if lam_g[i] < -1:
        #                 print("STRANGE: lagrange mult smaller than 2, while we do not hit lower bound!")
        #     elif lam_g[i] > 1e-5:
        #         if hit_type[i] == 2 or hit_type[i] == 3:
        #             print("Yes")
        #         else:
        #             if lam_g[i] > 1:
        #                 print("STRANGE: lagrange mult larger than 2, while we do not hit upper bound!")
        #     else:
        #         print(f"lagrange mult approx 0 ({lam_g[i]})")
        #         if hit_type[i] != 0:
        #             print("! BUT we do hit a bound...")
        #         else:
        #             print("and we do not hit a bound, as expected...")
        #     if lam_g[i] < -2 or lam_g[i] > 2:
        #         print("LAGRANGE MULT HAS A LARGE MAGNITUDE!")



        #     if violates_lbg[i]:
        #         print(f"Constraint {i} violates lbg: {lbg[i]} < {round(evaluated_g[i],100)} < {ubg[i]}")
        #     if violates_ubg[i]:
        #         print(f"Constraint {i} violates ubg: {lbg[i]} < {round(evaluated_g[i],100)} < {ubg[i]}")

        self.activated_lower_bounds = [True if lagrange_mult < -1.5 else False for lagrange_mult in lam_g]
        self.activated_upper_bounds = [True if lagrange_mult > 1.5 else False for lagrange_mult in lam_g]

        self.activated_lower_bounds_only = [True if lagrange_mult < -1.5 and hit_t != 3 else False for lagrange_mult, hit_t in zip(lam_g, hit_type)]
        self.activated_upper_bounds_only = [True if lagrange_mult > 1.5 and hit_t != 3 else False for lagrange_mult, hit_t in zip(lam_g, hit_type)]

        print(f"Number of activated lower bounds (only): {self.activated_lower_bounds_only.count(True)}")
        print(f"Number of activated upper bounds (only): {self.activated_upper_bounds_only.count(True)}")
        n_prints=0
        self._textual_constraints = constraints

        positive_effect = []
        negative_effect = []

        for i in range(0,len(evaluated_g)):
            if self.activated_lower_bounds_only[i]:
                print("hit lower bound: " + str(i))
                print(f"-> Constraint {i} is active ({hit_type_str[hit_type[i]]}): {lbg[i]} < {round(evaluated_g[i],100)} < {ubg[i]} (lam_g: {lam_g[i]})")
                print(f"-> Constraint {i} is ACTIVE ({hit_type_str[hit_type[i]]}): {lbg[i]-b[i]} < {round(evaluated_g[i]-b[i],100)} < {ubg[i]-b[i]} (lam_g: {lam_g[i]}) (b {b[i]})")
                print(constraints[i])
                constrain_list = constraints_original[i]
                for var_i in range(int(len(constrain_list)/3)):
                    var_sign  = constrain_list[  var_i*3]
                    var_value = constrain_list[1+var_i*3]
                    var_name   = constrain_list[2+var_i*3]
                    if var_sign == "-":
                        positive_effect.append(var_name)
                    else:
                        negative_effect.append(var_name)
                n_prints+=1
            if self.activated_upper_bounds_only[i]:
                print("hit upper bound: " + str(i))
                print(f"-> Constraint {i} is active ({hit_type_str[hit_type[i]]}): {lbg[i]} < {round(evaluated_g[i],100)} < {ubg[i]} (lam_g: {lam_g[i]})")
                print(f"-> Constraint {i} is ACTIVE ({hit_type_str[hit_type[i]]}): {lbg[i]-b[i]} < {round(evaluated_g[i]-b[i],100)} < {ubg[i]-b[i]} (lam_g: {lam_g[i]}) (b {b[i]})")
                print(constraints[i])
                constrain_list = constraints_original[i]
                for var_i in range(int(len(constrain_list)/3)):
                    var_sign  = constrain_list[  var_i*3]
                    var_value = constrain_list[1+var_i*3]
                    var_name   = constrain_list[2+var_i*3]
                    if var_sign == "+":
                        positive_effect.append(var_name)
                    else:
                        negative_effect.append(var_name)
                n_prints+=1
            if n_prints > 1000:
                break

        # ------- FOR BOTH -------
        def convert_to_dict_per_var(constrain_list):
            def add_to_dict(new_dict, variable, sign="+"):
                splitted_var = variable.split("__")
                if splitted_var[0] not in new_dict:
                    new_dict[splitted_var[0]] = {
                        'timesteps': [int(splitted_var[1])],
                        'effect_direction': sign
                    }
                else:
                    new_dict[splitted_var[0]]['timesteps'].append(int(splitted_var[1]))
                return new_dict

            new_dict = {}
            for constrain in constrain_list:
                if isinstance(constrain, list):
                    for i, variable in enumerate(constrain[2::3]):
                        add_to_dict(new_dict, variable, constrain[i*3])
                else:
                    variable = constrain
                    add_to_dict(new_dict, variable)

            # SORT VALUES, REMOVE DUPLICATES:
            for var_name in new_dict:
                new_dict[var_name]['timesteps'] = sorted(set(new_dict[var_name]['timesteps']))
            return new_dict

        positive_effect_dict = convert_to_dict_per_var(positive_effect)
        negative_effect_dict = convert_to_dict_per_var(negative_effect)
        self.negative_effect_dict = negative_effect_dict
        self.positive_effect_dict = positive_effect_dict


        # -------------------------
        # --------  NEW  ----------
        # -------------------------

        def find_lambda_exceedence(exceedence_list, lowers, uppers, variable_names, variable_values):
            variables_exceeding = []
            if any(exceedence_list):
                for i, larger_than_zero in enumerate(exceedence_list):
                    if larger_than_zero:
                        print(f'Bound for variable {variable_names[i]}={variable_values[i]} was hit!"')
                        print(f'{lowers[i]} < {variable_values[i]} < {uppers[i]}')
                        variables_exceeding.append(variable_names[i])
            return variables_exceeding

        def larger_than_zero(in_list, tol):
            return [x > tol for x in in_list]

        def smaller_than_zero(in_list, tol):
            return [x < -tol for x in in_list]

        lam_x_tol = 1.5
        # Bounds
        lam_x_larger_than_zero  = larger_than_zero(lam_x, lam_x_tol)
        lam_x_smaller_than_zero = smaller_than_zero(lam_x, lam_x_tol)
        self.upper_bound_variable_hits = find_lambda_exceedence(lam_x_larger_than_zero, lbx, ubx, variable_names, x_optimized)
        self.lower_bound_variable_hits = find_lambda_exceedence(lam_x_smaller_than_zero, lbx, ubx, variable_names, x_optimized)
        self.upper_bound_dict = convert_to_dict_per_var(self.upper_bound_variable_hits)
        self.lower_bound_dict = convert_to_dict_per_var(self.lower_bound_variable_hits)

        # Constraints (v2)
        lam_g_tol = lam_x_tol
        lam_g_larger_than_zero = larger_than_zero(lam_g, lam_g_tol)
        lam_g_smaller_than_zero = smaller_than_zero(lam_g, lam_g_tol)
        self.upper_constraint_variable_hits = find_lambda_exceedence(lam_g_larger_than_zero, lbg, ubg, constraints_original, evaluated_g)
        self.lower_constraint_variable_hits = find_lambda_exceedence(lam_g_smaller_than_zero, lbg, ubg, constraints_original, evaluated_g)
        self.upper_constraint_dict = convert_to_dict_per_var(self.upper_constraint_variable_hits)
        self.lower_constraint_dict = convert_to_dict_per_var(self.lower_constraint_variable_hits)
        pass


        # --> Unused function, but could be useful for refactoring convert_to_dict_per_var
        # get_variables_in_constraints(self.upper_constraint_variable_hits)
        # def get_variables_in_constraints(constraints):
        #     variables_in_constraints = []
        #     for constraint in constraints:
        #         variables_in_constraints.append([])
        #         for var_i in range(int(len(constraint)/3)):
        #             var_sign  = constraint[  var_i*3]
        #             var_value = constraint[1+var_i*3]
        #             var_name  = constraint[2+var_i*3]
        #             variables_in_constraints[-1].append(var_name)
        #     return variables_in_constraints

        # Do any postprocessing
        if postprocessing:
            self.post()
        else:
            logger.debug(
                'Skipping Postprocessing in OptimizationProblem.optimize()')

        # Done
        logger.info("Done with optimize()")

        return success


    def __check_bounds_control_input(self) -> None:
        # Checks if at the control inputs have bounds, log warning when a control input is not bounded.
        bounds = self.bounds()

        for variable in self.dae_variables['control_inputs']:
            variable = variable.name()
            if variable not in bounds:
                logger.warning(
                    "OptimizationProblem: control input {} has no bounds.".format(variable))

    @abstractmethod
    def transcribe(self) -> Tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, ca.MX]]:
        """
        Transcribe the continuous optimization problem to a discretized, solver-ready
        optimization problem.
        """
        pass

    def solver_options(self) -> Dict[str, Union[str, int, float, bool, str]]:
        """
        Returns a dictionary of CasADi optimization problem solver options.

        The default solver for continuous problems is `Ipopt <https://projects.coin-or.org/Ipopt/>`_.
        The default solver for mixed integer problems is `Bonmin <http://projects.coin-or.org/Bonmin/>`_.

        :returns: A dictionary of solver options. See the CasADi and
                  respective solver documentation for details.
        """
        options = {'error_on_fail': False,
                   'optimized_num_dir': 3,
                   'casadi_solver': ca.nlpsol}

        if self.__mixed_integer:
            options['solver'] = 'bonmin'

            bonmin_options = options['bonmin'] = {}
            bonmin_options['algorithm'] = 'B-BB'
            bonmin_options['nlp_solver'] = 'Ipopt'
            bonmin_options['nlp_log_level'] = 2
            bonmin_options['linear_solver'] = 'mumps'
        else:
            options['solver'] = 'ipopt'

            ipopt_options = options['ipopt'] = {}
            ipopt_options['linear_solver'] = 'mumps'
        return options

    def solver_success(self,
                       solver_stats: Dict[str, Union[str, bool]],
                       log_solver_failure_as_error: bool) -> Tuple[bool, int]:
        """
        Translates the returned solver statistics into a boolean and log level
        to indicate whether the solve was succesful, and how to log it.

        :param solver_stats: Dictionary containing information about the
                             solver status. See explanation below.
        :param log_solver_failure_as_error: Indicates whether a solve failure
            Should be logged as an error or info message.

        ``solver_stats`` typically consist of three fields:

        * return_status: ``str``
        * secondary_return_status: ``str``
        * success: ``bool``

        By default we rely on CasADi's interpretation of the return_status
        (and secondary status) to the success variable, with an exception for
        IPOPT (see below).

        The logging level is typically ``logging.INFO`` for success, and
        ``logging.ERROR`` for failure. Only for IPOPT an exception is made for
        `Not_Enough_Degrees_Of_Freedom`, which returns ``logging.WARNING`` instead.
        For example, this can happen when too many goals are specified, and
        lower priority goals cannot improve further on the current result.

        :returns: A tuple indicating whether or not the solver has succeeded, and what level to log it with.
        """
        success = solver_stats['success']
        log_level = logging.INFO if success else logging.ERROR

        if (self.solver_options()['solver'].lower() in ['bonmin', 'ipopt']
                and solver_stats['return_status'] in ['Not_Enough_Degrees_Of_Freedom']):
            log_level = logging.WARNING

        if log_level == logging.ERROR and not log_solver_failure_as_error:
            log_level = logging.INFO

        return success, log_level

    @abstractproperty
    def solver_input(self) -> ca.MX:
        """
        The symbolic input to the NLP solver.
        """
        pass

    @abstractmethod
    def extract_results(self, ensemble_member: int = 0) -> Dict[str, np.ndarray]:
        """
        Extracts state and control input time series from optimizer results.

        :returns: A dictionary of result time series.
        """
        pass

    @property
    def objective_value(self) -> float:
        """
        The last obtained objective function value.
        """
        return self.__objective_value

    @property
    def solver_output(self) -> np.ndarray:
        """
        The raw output from the last NLP solver run.
        """
        return self.__solver_output

    @property
    def solver_stats(self) -> Dict[str, Any]:
        """
        The stats from the last NLP solver run.
        """
        return self.__solver_stats

    def pre(self) -> None:
        """
        Preprocessing logic is performed here.
        """
        pass

    @abstractproperty
    def dae_residual(self) -> ca.MX:
        """
        Symbolic DAE residual of the model.
        """
        pass

    @abstractproperty
    def dae_variables(self) -> Dict[str, List[ca.MX]]:
        """
        Dictionary of symbolic variables for the DAE residual.
        """
        pass

    @property
    def path_variables(self) -> List[ca.MX]:
        """
        List of additional, time-dependent optimization variables, not covered by the DAE model.
        """
        return []

    @abstractmethod
    def variable(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the given variable.

        :param variable: Variable name.

        :returns: The associated CasADi :class:`MX` symbol.
        """
        raise NotImplementedError

    @property
    def extra_variables(self) -> List[ca.MX]:
        """
        List of additional, time-independent optimization variables, not covered by the DAE model.
        """
        return []

    @property
    def output_variables(self) -> List[ca.MX]:
        """
        List of variables that the user requests to be included in the output files.
        """
        return []

    def delayed_feedback(self) -> List[Tuple[str, str, float]]:
        """
        Returns the delayed feedback mappings.  These are given as a list of triples :math:`(x, y, \\tau)`,
        to indicate that :math:`y = x(t - \\tau)`.

        :returns: A list of triples.

        Example::

            def delayed_feedback(self):
                fb1 = ['x', 'y', 0.1]
                fb2 = ['x', 'z', 0.2]
                return [fb1, fb2]

        """
        return []

    @property
    def ensemble_size(self) -> int:
        """
        The number of ensemble members.
        """
        return 1

    def ensemble_member_probability(self, ensemble_member: int) -> float:
        """
        The probability of an ensemble member occurring.

        :param ensemble_member: The ensemble member index.

        :returns: The probability of an ensemble member occurring.

        :raises: IndexError
        """
        return 1.0

    def parameters(self, ensemble_member: int) -> AliasDict[str, Union[bool, int, float, ca.MX]]:
        """
        Returns a dictionary of parameters.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of parameter names and values.
        """
        return AliasDict(self.alias_relation)

    def string_parameters(self, ensemble_member: int) -> Dict[str, str]:
        """
        Returns a dictionary of string parameters.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of string parameter names and values.
        """
        return {}

    def constant_inputs(self, ensemble_member: int) -> AliasDict[str, Timeseries]:
        """
        Returns a dictionary of constant inputs.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of constant input names and time series.
        """
        return AliasDict(self.alias_relation)

    def lookup_tables(self, ensemble_member: int) -> AliasDict[str, LookupTable]:
        """
        Returns a dictionary of lookup tables.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and lookup tables.
        """
        return AliasDict(self.alias_relation)

    @staticmethod
    def merge_bounds(a: Tuple[BT, BT], b: Tuple[BT, BT]) -> Tuple[BT, BT]:
        """
        Returns a pair of bounds which is the intersection of the two pairs of
        bounds given as input.

        :param a: First pair ``(upper, lower)`` bounds
        :param b: Second pair ``(upper, lower)`` bounds

        :returns: A pair of ``(upper, lower)`` bounds which is the
                  intersection of the two input bounds.
        """
        a, A = a
        b, B = b

        # Make sure we are dealing with the correct types
        if __debug__:
            for v in (a, A, b, B):
                if isinstance(v, np.ndarray):
                    assert v.ndim == 1
                    assert np.issubdtype(v.dtype, np.number)
                else:
                    assert isinstance(v, (float, int, Timeseries))

        all_bounds = [a, A, b, B]

        # First make sure that we treat single element vectors as scalars
        for i, v in enumerate(all_bounds):
            if isinstance(v, np.ndarray) and np.prod(v.shape) == 1:
                all_bounds[i] = v.item()

        # Upcast lower bounds to be of equal type, and upper bounds as well.
        for i, j in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            v1 = all_bounds[i]
            v2 = all_bounds[j]

            # We only check for v1 being of a "smaller" type than v2, as we
            # know we will encounter the reverse as well.
            if isinstance(v1, type(v2)):
                # Same type, nothing to do.
                continue
            elif isinstance(v1, (int, float)) and isinstance(v2, Timeseries):
                all_bounds[i] = Timeseries(v2.times, np.full_like(v2.values, v1))
            elif isinstance(v1, np.ndarray) and isinstance(v2, Timeseries):
                if v2.values.ndim != 2 or len(v1) != v2.values.shape[1]:
                    raise Exception(
                        "Mismatching vector size when upcasting to Timeseries, {} vs. {}.".format(v1, v2))
                all_bounds[i] = Timeseries(v2.times, np.broadcast_to(v1, v2.values.shape))
            elif isinstance(v1, (int, float)) and isinstance(v2, np.ndarray):
                all_bounds[i] = np.full_like(v2, v1)

        a, A, b, B = all_bounds

        assert isinstance(a, type(b))
        assert isinstance(A, type(B))

        # Merge the bounds
        m, M = None, None

        if isinstance(a, np.ndarray):
            if not a.shape == b.shape:
                raise Exception("Cannot merge vector minimum bounds of non-equal size")
            m = np.maximum(a, b)
        elif isinstance(a, Timeseries):
            if len(a.times) != len(b.times):
                raise Exception("Cannot merge Timeseries minimum bounds with different lengths")
            elif not np.all(a.times == b.times):
                raise Exception("Cannot merge Timeseries minimum bounds with non-equal times")
            elif not a.values.shape == b.values.shape:
                raise Exception("Cannot merge vector Timeseries minimum bounds of non-equal size")
            m = Timeseries(a.times, np.maximum(a.values, b.values))
        else:
            m = max(a, b)

        if isinstance(A, np.ndarray):
            if not A.shape == B.shape:
                raise Exception("Cannot merge vector maximum bounds of non-equal size")
            M = np.minimum(A, B)
        elif isinstance(A, Timeseries):
            if len(A.times) != len(B.times):
                raise Exception("Cannot merge Timeseries maximum bounds with different lengths")
            elif not np.all(A.times == B.times):
                raise Exception("Cannot merge Timeseries maximum bounds with non-equal times")
            elif not A.values.shape == B.values.shape:
                raise Exception("Cannot merge vector Timeseries maximum bounds of non-equal size")
            M = Timeseries(A.times, np.minimum(A.values, B.values))
        else:
            M = min(A, B)

        return m, M

    def bounds(self) -> AliasDict[str, Tuple[BT, BT]]:
        """
        Returns variable bounds as a dictionary mapping variable names to a pair of bounds.
        A bound may be a constant, or a time series.

        :returns: A dictionary of variable names and ``(upper, lower)`` bound pairs.
                  The bounds may be numbers or :class:`.Timeseries` objects.

        Example::

            def bounds(self):
                return {'x': (1.0, 2.0), 'y': (2.0, 3.0)}

        """
        return AliasDict(self.alias_relation)

    def history(self, ensemble_member: int) -> AliasDict[str, Timeseries]:
        """
        Returns the state history.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and historical time series (up to and including t0).
        """
        return AliasDict(self.alias_relation)

    def variable_is_discrete(self, variable: str) -> bool:
        """
        Returns ``True`` if the provided variable is discrete.

        :param variable: Variable name.

        :returns: ``True`` if variable is discrete (integer).
        """
        return False

    def variable_nominal(self, variable: str) -> Union[float, np.ndarray]:
        """
        Returns the nominal value of the variable.  Variables are scaled by replacing them with
        their nominal value multiplied by the new variable.

        :param variable: Variable name.

        :returns: The nominal value of the variable.
        """
        return 1

    @property
    def initial_time(self) -> float:
        """
        The initial time in seconds.
        """
        return self.times()[0]

    @property
    def initial_residual(self) -> ca.MX:
        """
        The initial equation residual.

        Initial equations are used to find consistent initial conditions.

        :returns: An :class:`MX` object representing F in the initial equation F = 0.
        """
        return ca.MX(0)

    def seed(self, ensemble_member: int) -> AliasDict[str, Union[float, Timeseries]]:
        """
        Seeding data.  The optimization algorithm is seeded with the data returned by this method.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of variable names and seed time series.
        """
        return AliasDict(self.alias_relation)

    def objective(self, ensemble_member: int) -> ca.MX:
        """
        The objective function for the given ensemble member.

        Call :func:`OptimizationProblem.state_at` to return a symbol representing a model variable at a given time.

        :param ensemble_member: The ensemble member index.

        :returns: An :class:`MX` object representing the objective function.

        Example::

            def objective(self, ensemble_member):
                # Return value of state 'x' at final time:
                times = self.times()
                return self.state_at('x', times[-1], ensemble_member)

        """
        return ca.MX(0)

    def path_objective(self, ensemble_member: int) -> ca.MX:
        """
        Returns a path objective the given ensemble member.

        Path objectives apply to all times and ensemble members simultaneously.
        Call :func:`OptimizationProblem.state` to return a time- and ensemble-member-independent
        symbol representing a model variable.

        :param ensemble_member: The ensemble member index. This index is currently unused,
                                and here for future use only.

        :returns: A :class:`MX` object representing the path objective.

        Example::

            def path_objective(self, ensemble_member):
                # Minimize x(t) for all t
                return self.state('x')

        """
        return ca.MX(0)

    def constraints(self, ensemble_member: int) -> List[
            Tuple[ca.MX, Union[float, np.ndarray], Union[float, np.ndarray]]]:
        """
        Returns a list of constraints for the given ensemble member.

        Call :func:`OptimizationProblem.state_at` to return a symbol representing a model variable at a given time.

        :param ensemble_member: The ensemble member index.

        :returns: A list of triples ``(f, m, M)``, with an :class:`MX` object representing
                  the constraint function ``f``, lower bound ``m``, and upper bound ``M``.
                  The bounds must be numbers.

        Example::

            def constraints(self, ensemble_member):
                t = 1.0
                constraint1 = (
                    2 * self.state_at('x', t, ensemble_member),
                    2.0, 4.0)
                constraint2 = (
                    self.state_at('x', t, ensemble_member) + self.state_at('y', t, ensemble_member),
                    2.0, 3.0)
                return [constraint1, constraint2]

        """
        return []

    def path_constraints(self, ensemble_member: int) -> List[
            Tuple[ca.MX, Union[float, np.ndarray], Union[float, np.ndarray]]]:
        """
        Returns a list of path constraints.

        Path constraints apply to all times and ensemble members simultaneously.
        Call :func:`OptimizationProblem.state` to return a time- and ensemble-member-independent
        symbol representing a model variable.

        :param ensemble_member: The ensemble member index. This index may only
                                be used to supply member-dependent bounds.

        :returns: A list of triples ``(f, m, M)``, with an :class:`MX` object representing
                  the path constraint function ``f``, lower bound ``m``, and upper bound ``M``.
                  The bounds may be numbers or :class:`.Timeseries` objects.

        Example::

            def path_constraints(self, ensemble_member):
                # 2 * x must lie between 2 and 4 for every time instance.
                path_constraint1 = (2 * self.state('x'), 2.0, 4.0)
                # x + y must lie between 2 and 3 for every time instance
                path_constraint2 = (self.state('x') + self.state('y'), 2.0, 3.0)
                return [path_constraint1, path_constraint2]

        """
        return []

    def post(self) -> None:
        """
        Postprocessing logic is performed here.
        """
        pass

    @property
    def equidistant(self) -> bool:
        """
        ``True`` if all time series are equidistant.
        """
        return False

    INTERPOLATION_LINEAR = 0
    INTERPOLATION_PIECEWISE_CONSTANT_FORWARD = 1
    INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD = 2

    def interpolate(
            self,
            t: Union[float, np.ndarray],
            ts: np.ndarray,
            fs: np.ndarray,
            f_left: float = np.nan,
            f_right: float = np.nan,
            mode: int = INTERPOLATION_LINEAR) -> Union[float, np.ndarray]:
        """
        Linear interpolation over time.

        :param t:       Time at which to evaluate the interpolant.
        :type t:        float or vector of floats
        :param ts:      Time stamps.
        :type ts:       numpy array
        :param fs:      Function values at time stamps ts.
        :param f_left:  Function value left of leftmost time stamp.
        :param f_right: Function value right of rightmost time stamp.
        :param mode:    Interpolation mode.

        :returns: The interpolated value.
        """

        if isinstance(fs, np.ndarray) and fs.ndim == 2:
            # 2-D array of values. Interpolate each column separately.
            if len(t) == len(ts) and np.all(t == ts):
                # Early termination; nothing to interpolate
                return fs.copy()

            fs_int = [self.interpolate(t, ts, fs[:, i], f_left, f_right, mode) for i in range(fs.shape[1])]
            return np.stack(fs_int, axis=1)
        elif hasattr(t, '__iter__'):
            if len(t) == len(ts) and np.all(t == ts):
                # Early termination; nothing to interpolate
                return fs.copy()

            return self.__interpolate(t, ts, fs, f_left, f_right, mode)
        else:
            if ts[0] == t:
                # Early termination; nothing to interpolate
                return fs[0]

            return self.__interpolate(t, ts, fs, f_left, f_right, mode)

    def __interpolate(self, t, ts, fs, f_left=np.nan, f_right=np.nan, mode=INTERPOLATION_LINEAR):
        """
        Linear interpolation over time.

        :param t:       Time at which to evaluate the interpolant.
        :type t:        float or vector of floats
        :param ts:      Time stamps.
        :type ts:       numpy array
        :param fs:      Function values at time stamps ts.
        :param f_left:  Function value left of leftmost time stamp.
        :param f_right: Function value right of rightmost time stamp.
        :param mode:    Interpolation mode.

        Note that it is assumed that `ts` is sorted. No such assumption is made for `t`
.
        :returns: The interpolated value.
        """

        if f_left is None:
            if (min(t) if hasattr(t, '__iter__') else t) < ts[0]:
                raise Exception("Interpolation: Point {} left of range".format(t))

        if f_right is None:
            if (max(t) if hasattr(t, '__iter__') else t) > ts[-1]:
                raise Exception("Interpolation: Point {} right of range".format(t))

        if mode == self.INTERPOLATION_LINEAR:
            # No need to handle f_left / f_right; NumPy already does this for us
            return np.interp(t, ts, fs, f_left, f_right)
        elif mode == self.INTERPOLATION_PIECEWISE_CONSTANT_FORWARD:
            v = fs[np.maximum(np.searchsorted(ts, t, side='right') - 1, 0)]
        elif mode == self.INTERPOLATION_PIECEWISE_CONSTANT_BACKWARD:
            v = fs[np.minimum(np.searchsorted(ts, t, side='left'), len(ts) - 1)]
        else:
            raise NotImplementedError

        # Handle f_left / f_right:
        if hasattr(t, "__iter__"):
            v[t < ts[0]] = f_left
            v[t > ts[-1]] = f_right
        else:
            if t < ts[0]:
                v = f_left
            elif t > ts[-1]:
                v = f_right

        return v

    @abstractproperty
    def controls(self) -> List[str]:
        """
        List of names of the control variables (excluding aliases).
        """
        pass

    @abstractmethod
    def discretize_controls(self, resolved_bounds: AliasDict) -> Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs the discretization of the control inputs, filling lower and upper
        bound vectors for the resulting optimization variables, as well as an initial guess.

        :param resolved_bounds: :class:`AliasDict` of numerical bound values. This is the
                                same dictionary as returned by :func:`bounds`, but with all
                                parameter symbols replaced with their numerical values.

        :returns: The number of control variables in the optimization problem, a lower
                  bound vector, an upper bound vector, a seed vector, and a dictionary
                  of offset values.
        """
        pass

    def dynamic_parameters(self) -> List[ca.MX]:
        """
        Returns a list of parameter symbols that may vary from run to run.  The values
        of these parameters are not cached.

        :returns: A list of parameter symbols.
        """
        return []

    @abstractmethod
    def extract_controls(self, ensemble_member: int = 0) -> Dict[str, np.ndarray]:
        """
        Extracts state time series from optimizer results.

        Must return a dictionary of result time series.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of control input time series.
        """
        pass

    def control_vector(self, variable: str, ensemble_member: int = 0) -> Union[ca.MX, List[ca.MX]]:
        """
        Return the optimization variables for the entire time horizon of the given state.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: A vector of control input symbols for the entire time horizon.

        :raises: KeyError
        """
        return self.state_vector(variable, ensemble_member)

    def control(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the given control input, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given control input.

        :raises: KeyError
        """
        return self.variable(variable)

    @abstractmethod
    def control_at(self, variable: str, t: float, ensemble_member: int = 0, scaled: bool = False) -> ca.MX:
        """
        Returns an :class:`MX` symbol representing the given control input at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.
        :param scaled:          True to return the scaled variable.

        :returns: :class:`MX` symbol representing the control input at the given time.

        :raises: KeyError
        """
        pass

    @abstractproperty
    def differentiated_states(self) -> List[str]:
        """
        List of names of the differentiated state variables (excluding aliases).
        """
        pass

    @abstractproperty
    def algebraic_states(self) -> List[str]:
        """
        List of names of the algebraic state variables (excluding aliases).
        """
        pass

    @abstractmethod
    def discretize_states(self, resolved_bounds: AliasDict) -> Tuple[
            int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform the discretization of the states.

        Fills lower and upper bound vectors for the resulting optimization
        variables, as well as an initial guess.

        :param resolved_bounds: :class:`AliasDict` of numerical bound values.
            This is the same dictionary as returned by :func:`bounds`, but
            with all parameter symbols replaced with their numerical values.

        :returns: The number of control variables in the optimization problem,
                  a lower bound vector, an upper bound vector, a seed vector,
                  and a dictionary of vector offset values.
        """
        pass

    @abstractmethod
    def extract_states(self, ensemble_member: int = 0) -> Dict[str, np.ndarray]:
        """
        Extracts state time series from optimizer results.

        Must return a dictionary of result time series.

        :param ensemble_member: The ensemble member index.

        :returns: A dictionary of state time series.
        """
        pass

    @abstractmethod
    def state_vector(self, variable: str, ensemble_member: int = 0) -> Union[ca.MX, List[ca.MX]]:
        """
        Return the optimization variables for the entire time horizon of the given state.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: A vector of state symbols for the entire time horizon.

        :raises: KeyError
        """
        pass

    def state(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the given state, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given state.

        :raises: KeyError
        """
        return self.variable(variable)

    @abstractmethod
    def state_at(self, variable: str, t: float, ensemble_member: int = 0, scaled: bool = False) -> ca.MX:
        """
        Returns an :class:`MX` symbol representing the given variable at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.
        :param scaled:          True to return the scaled variable.

        :returns: :class:`MX` symbol representing the state at the given time.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def extra_variable(self, variable: str, ensemble_member: int = 0) -> ca.MX:
        """
        Returns an :class:`MX` symbol representing the extra variable inside the state vector.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` symbol representing the extra variable.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def states_in(self, variable: str, t0: float = None, tf: float = None, ensemble_member: int = 0) -> Iterator[ca.MX]:
        """
        Iterates over symbols for states in the interval [t0, tf].

        :param variable:        Variable name.
        :param t0:              Left bound of interval.  If equal to None, the initial time is used.
        :param tf:              Right bound of interval.  If equal to None, the final time is used.
        :param ensemble_member: The ensemble member index.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def integral(self, variable: str, t0: float = None, tf: float = None, ensemble_member: int = 0) -> ca.MX:
        """
        Returns an expression for the integral over the interval [t0, tf].

        :param variable:        Variable name.
        :param t0:              Left bound of interval.  If equal to None, the initial time is used.
        :param tf:              Right bound of interval.  If equal to None, the final time is used.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` object representing the integral.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def der(self, variable: str) -> ca.MX:
        """
        Returns an :class:`MX` symbol for the time derivative given state, not bound to any time.

        :param variable: Variable name.

        :returns: :class:`MX` symbol for given state.

        :raises: KeyError
        """
        pass

    @abstractmethod
    def der_at(self, variable: str, t: float, ensemble_member: int = 0) -> ca.MX:
        """
        Returns an expression for the time derivative of the specified variable at time t.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.

        :returns: :class:`MX` object representing the derivative.

        :raises: KeyError
        """
        pass

    def get_timeseries(self, variable: str, ensemble_member: int = 0) -> Timeseries:
        """
        Looks up a timeseries from the internal data store.

        :param variable:        Variable name.
        :param ensemble_member: The ensemble member index.

        :returns: The requested time series.
        :rtype: :class:`.Timeseries`

        :raises: KeyError
        """
        raise NotImplementedError

    def set_timeseries(
            self,
            variable: str,
            timeseries: Timeseries,
            ensemble_member: int = 0,
            output: bool = True,
            check_consistency: bool = True) -> None:
        """
        Sets a timeseries in the internal data store.

        :param variable:          Variable name.
        :param timeseries:        Time series data.
        :type timeseries:         iterable of floats, or :class:`.Timeseries`
        :param ensemble_member:   The ensemble member index.
        :param output:            Whether to include this time series in output data files.
        :param check_consistency: Whether to check consistency between the time stamps on
                                  the new timeseries object and any existing time stamps.
        """
        raise NotImplementedError

    def timeseries_at(self, variable: str, t: float, ensemble_member: int = 0) -> float:
        """
        Return the value of a time series at the given time.

        :param variable:        Variable name.
        :param t:               Time.
        :param ensemble_member: The ensemble member index.

        :returns: The interpolated value of the time series.

        :raises: KeyError
        """
        raise NotImplementedError

    def map_path_expression(self, expr: ca.MX, ensemble_member: int) -> ca.MX:
        """
        Maps the path expression `expr` over the entire time horizon of the optimization problem.

        :param expr: An :class:`MX` path expression.

        :returns: An :class:`MX` expression evaluating `expr` over the entire time horizon.
        """
        raise NotImplementedError

    @debug_check(DebugLevel.HIGH)
    def __debug_check_linearity_constraints(self, nlp):
        x = nlp['x']
        f = nlp['f']
        g = nlp['g']

        expand_f_g = ca.Function('f_g', [x], [f, g]).expand()
        X_sx = ca.SX.sym('X', *x.shape)
        f_sx, g_sx = expand_f_g(X_sx)

        jac = ca.Function('j', [X_sx], [ca.jacobian(g_sx, X_sx)]).expand()
        if jac(np.nan).is_regular():
            logger.info("The constraints are linear")
        else:
            hes = ca.Function('j', [X_sx], [ca.jacobian(ca.jacobian(g_sx, X_sx), X_sx)]).expand()
            if hes(np.nan).is_regular():
                logger.info("The constraints are quadratic")
            else:
                logger.info("The constraints are nonlinear")

    @debug_check(DebugLevel.VERYHIGH)
    def __debug_check_linear_independence(self, lbx, ubx, lbg, ubg, nlp):
        x = nlp['x']
        f = nlp['f']
        g = nlp['g']

        expand_f_g = ca.Function('f_g', [x], [f, g]).expand()
        x_sx = ca.SX.sym('X', *x.shape)
        f_sx, g_sx = expand_f_g(x_sx)

        x, f, g = x_sx, f_sx, g_sx

        lbg = np.array(ca.vertsplit(ca.veccat(*lbg))).ravel()
        ubg = np.array(ca.vertsplit(ca.veccat(*ubg))).ravel()

        # Find the linear constraints
        g_sjac = ca.Function('Af', [x], [ca.jtimes(g, x, x.ones(*x.shape))])

        res = g_sjac(np.nan)
        res = np.array(res).ravel()
        g_is_linear = ~np.isnan(res)

        # Find the rows in the jacobian with only a single entry
        g_jac_csr = ca.DM(ca.Function('tmp', [x], [g]).sparsity_jac(0, 0)).tocsc().tocsr()
        g_single_variable = (np.diff(g_jac_csr.indptr) == 1)

        # Find the rows which are equality constraints
        g_eq_constraint = (lbg == ubg)

        # The intersection of all selections are constraints like we want
        g_constant_assignment = g_is_linear & g_single_variable & g_eq_constraint

        # Map of variable (index) to constraints/row numbers
        var_index_assignment = {}
        for i in range(g.size1()):
            if g_constant_assignment[i]:
                var_ind = g_jac_csr.getrow(i).indices[0]
                var_index_assignment.setdefault(var_ind, []).append(i)

        var_names, named_x, named_f, named_g = self._debug_get_named_nlp(nlp)

        for vi, g_inds in var_index_assignment.items():
            if len(g_inds) > 1:
                logger.info("Variable '{}' has duplicate constraints setting its value:".format(var_names[vi]))
                for g_i in g_inds:
                    logger.info("row {}: {} = {}".format(g_i, named_g[g_i], lbg[g_i]))

        # Find variables for which the bounds are equal, but also an equality
        # constraint is set. This would result in a constraint `1 = 1` with
        # the default IPOPT option `fixed_variable_treatment = make_parameter`
        x_inds = np.flatnonzero(lbx == ubx)

        for vi in x_inds:
            if vi in var_index_assignment:
                logger.info("Variable '{}' has equal bounds (value = {}), but also the following equality constraints:"
                            .format(var_names[vi], lbx[vi]))
                for g_i in var_index_assignment[vi]:
                    logger.info("row {}: {} = {}".format(g_i, named_g[g_i], lbg[g_i]))
