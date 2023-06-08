import copy
import logging

import casadi as ca

import numpy as np

import pandas as pd

from rtctools.diagnostics_utils import (
    casadi_to_lp,
    convert_constraints,
    get_systems_of_equations,
    get_varnames,
)

logger = logging.getLogger("rtctools")


def get_constraints(casadi_equations):
    expand_f_g = casadi_equations["func"]
    in_var = ca.SX.sym("X", expand_f_g.nnz_in())
    bf = ca.Function("bf", [in_var], [expand_f_g(in_var)[1]])
    b = bf(0)
    b = ca.sparsify(b)
    b = np.array(b)[:, 0]
    constraints = casadi_to_lp(casadi_equations)
    return constraints


def evaluate_constraints(results, nlp, casadi_equations):
    # Evaluate the constraints wrt to the optimized solution
    x_optimized = np.array(results["x"]).ravel()
    X_sx = ca.SX.sym("X", *nlp["x"].shape)
    f_sx, g_sx = casadi_equations["func"](X_sx)
    eval_g = ca.Function("g_eval", [X_sx], [g_sx]).expand()
    evaluated_g = [x[0] for x in np.array(eval_g(x_optimized))]
    lam_g = [x[0] for x in np.array(results["lam_g"])]
    lam_x = [x[0] for x in np.array(results["lam_x"])]

    return evaluated_g, lam_g, lam_x


def extract_var_name_timestep(variable):
    """Split the variable name into its original name and its timestep"""
    var_name, _, timestep_str = variable.partition("__")
    return var_name, int(timestep_str)


def add_to_dict(new_dict, var_name, timestep, sign="+"):
    """Add variable to dict grouped by variable names"""
    if var_name not in new_dict:
        new_dict[var_name] = {"timesteps": [timestep], "effect_direction": sign}
    else:
        new_dict[var_name]["timesteps"].append(timestep)
    return new_dict


def convert_to_dict_per_var(constrain_list):
    """Convert list of ungrouped variables to a dict per variable name,
    with as values the time-indices where the variable was active"""
    new_dict = {}
    for constrain in constrain_list:
        if isinstance(constrain, list):
            for i, variable in enumerate(constrain[2::3]):
                var_name, timestep = extract_var_name_timestep(variable)
                add_to_dict(new_dict, var_name, timestep, constrain[i * 3])
        else:
            var_name, timestep = extract_var_name_timestep(constrain)
            add_to_dict(new_dict, var_name, timestep)
    # Sort values and remove duplicates
    for var_name in new_dict:
        new_dict[var_name]["timesteps"] = sorted(set(new_dict[var_name]["timesteps"]))
    return new_dict

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


def check_lambda_exceedence(lam, tol):
    return [x > tol for x in lam], [x < -tol for x in lam]


def find_variable_hits(
    exceedance_list, lowers, uppers, variable_names, variable_values, lam
):
    variable_hits = []
    for i, hit in enumerate(exceedance_list):
        if hit:
            logger.debug(
                "Bound for variable {}={} was hit! Lam={}".format(
                    variable_names[i], variable_values[i], lam
                )
            )
            logger.debug(
                "{} < {} < {}".format(lowers[i], variable_values[i], uppers[i])
            )
            variable_hits.append(variable_names[i])
    return variable_hits


# part 1 functions
def conversion_step_one(results, nlp, casadi_equations, lam_tol):
    constraints = get_constraints(casadi_equations)
    lbx, ubx, lbg, ubg, _x0 = casadi_equations["other"]
    variable_names = get_varnames(casadi_equations)

    evaluated_g, lam_g, lam_x = evaluate_constraints(results, nlp, casadi_equations)

    # Upper and lower bounds
    lam_x_larger_than_zero, lam_x_smaller_than_zero = check_lambda_exceedence(
        lam_x, lam_tol
    )
    upper_bound_variable_hits = find_variable_hits(
        lam_x_larger_than_zero,
        lbx,
        ubx,
        variable_names,
        np.array(results["x"]).ravel(),
        lam_x,
    )
    lower_bound_variable_hits = find_variable_hits(
        lam_x_smaller_than_zero,
        lbx,
        ubx,
        variable_names,
        np.array(results["x"]).ravel(),
        lam_x,
    )
    upper_bound_dict = convert_to_dict_per_var(upper_bound_variable_hits)
    lower_bound_dict = convert_to_dict_per_var(lower_bound_variable_hits)

    # Upper and lower constraints
    lam_g_larger_than_zero, lam_g_smaller_than_zero = check_lambda_exceedence(
        lam_g, lam_tol
    )
    upper_constraint_variable_hits = find_variable_hits(
        lam_g_larger_than_zero, lbg, ubg, constraints, evaluated_g, lam_g
    )
    lower_constraint_variable_hits = find_variable_hits(
        lam_g_smaller_than_zero, lbg, ubg, constraints, evaluated_g, lam_g
    )
    upper_constraint_dict = convert_to_dict_per_var(upper_constraint_variable_hits)
    lower_constraint_dict = convert_to_dict_per_var(lower_constraint_variable_hits)

    return (
        upper_bound_dict,
        lower_bound_dict,
        upper_constraint_dict,
        lower_constraint_dict,
    )


def get_all_active_constraints(results, nlp, casadi_equations, lam_tol=0.1, n_dec=4):
    constraints = get_constraints(casadi_equations)
    _lbx, _ubx, lbg, ubg, _x0 = casadi_equations["other"]
    eq_systems = get_systems_of_equations(casadi_equations)
    _A, b = eq_systems["constraints"]
    converted_constraints = convert_constraints(constraints, lbg, ubg, b, n_dec)
    _evaluated_g, lam_g, _lam_x = evaluate_constraints(results, nlp, casadi_equations)
    lam_g_larger_than_zero, lam_g_smaller_than_zero = check_lambda_exceedence(
        lam_g, lam_tol
    )
    active_upper_constraints = [
        constraint
        for i, constraint in enumerate(converted_constraints)
        if lam_g_larger_than_zero[i]
    ]
    active_lower_constraints = [
        constraint
        for i, constraint in enumerate(converted_constraints)
        if lam_g_smaller_than_zero[i]
    ]
    return active_lower_constraints, active_upper_constraints


def list_to_ranges(lst):
    if not lst:
        return []
    ranges = []
    start = end = lst[0]
    for i in range(1, len(lst)):
        if lst[i] == end + 1:
            end = lst[i]
        else:
            ranges.append((start, end))
            start = end = lst[i]
    ranges.append((start, end))
    return ranges


def convert_lists_in_dict(dic):
    new_dic = copy.deepcopy(dic)
    for key, val in dic.items():
        new_dic[key]["timesteps"] = list_to_ranges(val["timesteps"])
    return new_dic


def strip_timestep(s):
    parts = []
    for part in s.split():
        if "__" in part:
            name, _ = part.split("__")
            name += "__"
            parts.append(name)
        else:
            parts.append(part)
    return " ".join(parts)


def add_symbol_before_line(lines, symbol):
    return "\n".join([f"{symbol} {line}" for line in lines.split("\n")])


def add_blockquote(lines):
    return add_symbol_before_line(lines, ">")


def group_variables(equations):
    unique_equations = {}
    for equation in equations:
        variables = {}
        # Get all variables in equation
        for var in equation.split():
            if "__" in var:
                var_name, var_suffix = var.split("__")
                if var_name in variables:
                    if variables[var_name] != var_suffix:
                        variables[var_name] = None
                else:
                    variables[var_name] = var_suffix
        variables = {k: v for k, v in variables.items() if v is not None}
        key = strip_timestep(equation)

        # Add equation to dict of unique equations
        if key in unique_equations:
            unique_suffixes = unique_equations[key]
            for var_suffix in variables.values():
                if var_suffix not in unique_suffixes:
                    unique_suffixes.append(var_suffix)
        else:
            unique_equations[key] = [list(variables.values())[0]]

    return unique_equations


# end part 2 functions
def get_debug_markdown_per_prio(
    lowerconstr_range_dict,
    upperconstr_range_dict,
    lowerbound_range_dict,
    upperbound_range_dict,
    active_lower_constraints,
    active_upper_constraints,
    priority="unknown",
):
    upper_constraints_df = pd.DataFrame.from_dict(
        upperconstr_range_dict, orient="index"
    )
    lower_constraints_df = pd.DataFrame.from_dict(
        lowerconstr_range_dict, orient="index"
    )
    lowerbounds_df = pd.DataFrame.from_dict(lowerbound_range_dict, orient="index")
    upperbounds_df = pd.DataFrame.from_dict(upperbound_range_dict, orient="index")
    result_text = "\n# Priority {}\n".format(priority)
    result_text += "## Lower constraints:\n"
    if len(lower_constraints_df):
        result_text += ">### Active variables:\n"
        result_text += add_blockquote(lower_constraints_df.to_markdown()) + "\n"
        result_text += ">### from active constraints:\n"
        for eq, timesteps in group_variables(active_lower_constraints).items():
            result_text += f">- `{eq}`: {timesteps}\n"
    else:
        result_text += ">No active lower constraints\n"

    result_text += "\n## Upper constraints:\n"
    if len(upper_constraints_df):
        result_text += ">### Active variables:\n"
        result_text += add_blockquote(upper_constraints_df.to_markdown()) + "\n"
        result_text += ">### from active constraints:\n"
        for eq, timesteps in group_variables(active_upper_constraints).items():
            result_text += f">- `{eq}`: {timesteps}\n"
    else:
        result_text += ">No active upper constraints\n"

    result_text += "\n ## Lower bounds:\n"
    if len(lowerbounds_df):
        result_text += add_blockquote(lowerbounds_df.to_markdown()) + "\n"
    else:
        result_text += ">No active lower bounds\n"
    result_text += "\n ## Upper bounds:\n"
    if len(upperbounds_df):
        result_text += add_blockquote(upperbounds_df.to_markdown()) + "\n"
    else:
        result_text += ">No active upper bounds\n"
    return result_text


class GetLinearProblem:
    lam_tol = 0.1
    manual_expansion = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.problem_and_results_list = []

    def problem_and_results(self, results, nlp, lbx, ubx, lbg, ubg, x0):
        super().problem_and_results(nlp, results, lbx, ubx, lbg, ubg, x0)
        expand_f_g = ca.Function("f_g", [nlp["x"]], [nlp["f"], nlp["g"]]).expand()
        casadi_equations = {}
        casadi_equations[
            "indices"
        ] = self._CollocatedIntegratedOptimizationProblem__indices
        casadi_equations["func"] = expand_f_g
        casadi_equations["other"] = (lbx, ubx, lbg, ubg, x0)
        self.problem_and_results_list.append((results, nlp, casadi_equations))

    def post(self):
        super().post()

        result_text = ""
        if len(self.problem_and_results_list) == 0:
            result_text += "No completed priorities... Is the problem infeasible?"

        for problem_and_results in self.problem_and_results_list:
            results, nlp, casadi_equations = problem_and_results
            (
                upper_bound_dict,
                lower_bound_dict,
                upper_constraint_dict,
                lower_constraint_dict,
            ) = conversion_step_one(results, nlp, casadi_equations, self.lam_tol)
            upperconstr_range_dict = convert_lists_in_dict(upper_constraint_dict)
            lowerconstr_range_dict = convert_lists_in_dict(lower_constraint_dict)
            lowerbound_range_dict = convert_lists_in_dict(upper_bound_dict)
            upperbound_range_dict = convert_lists_in_dict(lower_bound_dict)

            (
                active_lower_constraints,
                active_upper_constraints,
            ) = get_all_active_constraints(results, nlp, casadi_equations, self.lam_tol)

            priority = "unknown"
            result_text += get_debug_markdown_per_prio(
                lowerconstr_range_dict,
                upperconstr_range_dict,
                lowerbound_range_dict,
                upperbound_range_dict,
                active_lower_constraints,
                active_upper_constraints,
                priority=priority,
            )

        with open("active_constraints_per_priority.md", "w") as f:
            f.write(result_text)
