import copy
import logging
import os

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
    """ Get constraints in human-readable format"""
    return casadi_to_lp(casadi_equations)


def evaluate_constraints(results, nlp):
    """ Evaluate the constraints wrt to the optimized solution """
    x_optimized = np.array(results["x"]).ravel()
    X_sx = ca.SX.sym("X", *nlp["x"].shape)
    expand_fg = ca.Function("f_g", [nlp["x"]], [nlp["f"], nlp["g"]]).expand()
    _f_sx, g_sx = expand_fg(X_sx)
    eval_g = ca.Function("g_eval", [X_sx], [g_sx]).expand()
    evaluated_g = [x[0] for x in np.array(eval_g(x_optimized))]
    return evaluated_g


def print_evaluated_constraints(results, nlp, casadi_equations):
    """ Print the actual constraints, showing the actual value of each
    variable between brackets."""


def get_lagrange_mult(results):
    """Get the lagrange multipliers for the constraints (g) and bounds (x)"""
    lam_g = [x[0] for x in np.array(results["lam_g"])]
    lam_x = [x[0] for x in np.array(results["lam_x"])]
    return lam_g, lam_x


def extract_var_name_timestep(variable):
    """Split the variable name into its original name and its timestep"""
    var_name, _, timestep_str = variable.partition("__")
    return var_name, int(timestep_str.split("_")[-1])


def add_to_dict(new_dict, var_name, timestep, sign="+"):
    """Add variable to dict grouped by variable names"""
    if var_name not in new_dict:
        new_dict[var_name] = {"timesteps": [timestep], "effect_direction": sign}
    else:
        if new_dict[var_name]["effect_direction"] == "sign":
            new_dict[var_name]["timesteps"].append(timestep)
        else:
            new_dict[var_name+sign] = {"timesteps": [timestep], "effect_direction": sign}
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


def get_tol_exceedance(in_list, tolerance):
    return [x > tolerance for x in in_list], [x < -tolerance for x in in_list]


def find_variable_hits(
    exceedance_list, lowers, uppers, variable_names, variable_values, lam
):
    """ Returns the elements of variable_names corresponding to the elements
    that are True in the exceedance list. """
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


def get_variables_in_active_constr(results, nlp, casadi_equations, lam_tol):
    """"
    This function determines all active constraints/bounds and extracts the
    variables that are in those active constraints/bounds. It returns dictionaries
    with keys indicating the active variable and with values the timestep(s) at which
    that variable is active.
    """
    constraints = get_constraints(casadi_equations)
    lbx, ubx, lbg, ubg, _x0 = casadi_equations["other"]
    variable_names = get_varnames(casadi_equations)

    lam_g, lam_x = get_lagrange_mult(results)

    # Upper and lower bounds
    lam_x_larger_than_zero, lam_x_smaller_than_zero = get_tol_exceedance(
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
    lam_g_larger_than_zero, lam_g_smaller_than_zero = get_tol_exceedance(
        lam_g, lam_tol
    )

    evaluated_g = evaluate_constraints(results, nlp)
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


def get_active_constraints(results, casadi_equations, lam_tol=0.1, n_dec=4):
    """Get all constraints that are active in a human-radable format."""
    constraints = get_constraints(casadi_equations)
    _lbx, _ubx, lbg, ubg, _x0 = casadi_equations["other"]
    eq_systems = get_systems_of_equations(casadi_equations)
    _A, b = eq_systems["constraints"]
    converted_constraints = convert_constraints(constraints, lbg, ubg, b, n_dec)
    lam_g, _lam_x = get_lagrange_mult(results)
    lam_g_larger_than_zero, lam_g_smaller_than_zero = get_tol_exceedance(
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
    """Given a list with integers, returns a list of closed ranges
    present in the input-list."""
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
    """ Converts all lists in a dictionairy to lists of ranges.
    See list_to_ranges. """
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
    """For markdown formatting"""
    return "\n".join([f"{symbol} {line}" for line in lines.split("\n")])


def add_blockquote(lines):
    """For markdown formatting"""
    return add_symbol_before_line(lines, ">")


def group_equations(equations):
    """ Group identical equations for different timesteps. """
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
        variables_equal = {k: v for k, v in variables.items() if v is not None}
        variables_none = {k: v for k, v in variables.items() if v is None}
        timesteps = list(variables_equal.values())
        if len(variables_none) > 0 or not all(x == timesteps[0] for x in timesteps):
            unique_equations[equation] = "Equation depends on >1 timestep"
        else:
            key = strip_timestep(equation)
            # Add equation to dict of unique equations
            if key in unique_equations:
                unique_suffixes = unique_equations[key]
                for var_suffix in variables_equal.values():
                    if var_suffix not in unique_suffixes:
                        unique_suffixes.append(var_suffix)
            else:
                if len(variables_equal.values()) > 0:
                    unique_equations[key] = [list(variables_equal.values())[0]]

    return unique_equations


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
        for eq, timesteps in group_equations(active_lower_constraints).items():
            result_text += f">- `{eq}`: {timesteps}\n"
    else:
        result_text += ">No active lower constraints\n"

    result_text += "\n## Upper constraints:\n"
    if len(upper_constraints_df):
        result_text += ">### Active variables:\n"
        result_text += add_blockquote(upper_constraints_df.to_markdown()) + "\n"
        result_text += ">### from active constraints:\n"
        for eq, timesteps in group_equations(active_upper_constraints).items():
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

    def problem_and_results(self, results, nlp, lbx, ubx, lbg, ubg, x0, priority=None):
        super().problem_and_results(nlp, results, lbx, ubx, lbg, ubg, x0, priority)
        expand_f_g = ca.Function("f_g", [nlp["x"]], [nlp["f"], nlp["g"]]).expand()
        casadi_equations = {}
        casadi_equations[
            "indices"
        ] = self._CollocatedIntegratedOptimizationProblem__indices
        casadi_equations["func"] = expand_f_g
        casadi_equations["other"] = (lbx, ubx, lbg, ubg, x0)
        self.problem_and_results_list.append((priority, results, nlp, casadi_equations))

    def post(self):
        super().post()

        result_text = ""
        if len(self.problem_and_results_list) == 0:
            result_text += "No completed priorities... Is the problem infeasible?"

        for problem_and_results in self.problem_and_results_list:
            priority, results, nlp, casadi_equations = problem_and_results
            (
                upper_bound_dict,
                lower_bound_dict,
                upper_constraint_dict,
                lower_constraint_dict,
            ) = get_variables_in_active_constr(results, nlp, casadi_equations, self.lam_tol)
            upperconstr_range_dict = convert_lists_in_dict(upper_constraint_dict)
            lowerconstr_range_dict = convert_lists_in_dict(lower_constraint_dict)
            lowerbound_range_dict = convert_lists_in_dict(upper_bound_dict)
            upperbound_range_dict = convert_lists_in_dict(lower_bound_dict)

            (
                active_lower_constraints,
                active_upper_constraints,
            ) = get_active_constraints(results, casadi_equations, self.lam_tol)

            result_text += get_debug_markdown_per_prio(
                lowerconstr_range_dict,
                upperconstr_range_dict,
                lowerbound_range_dict,
                upperbound_range_dict,
                active_lower_constraints,
                active_upper_constraints,
                priority=priority,
            )

        with open(os.path.join(self._output_folder, "active_constraints.md"), "w") as f:
            f.write(result_text)
