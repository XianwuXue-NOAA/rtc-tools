import copy
import logging

import pandas as pd


logger = logging.getLogger("rtctools")


class GetLinearProblem:
    store_intermediate_lp_info = True
    lam_tol = 0.1

    def pre(self):
        super().pre()
        self.intermediate_lp_info = []

    def priority_completed(self, priority: int) -> None:
        to_store = {
            "priority": priority,
            "upper_constraint_dict": self.upper_constraint_dict,
            "lower_constraint_dict": self.lower_constraint_dict,
            "upper_bound_dict": self.upper_bound_dict,
            "lower_bound_dict": self.lower_bound_dict,
            "active_lower_constraints": self.active_lower_constraints,
            "active_upper_constraints": self.active_upper_constraints,
        }
        self.intermediate_lp_info.append(to_store)
        super().priority_completed(priority)

    def post(self):
        super().post()

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

        result_text = ""
        if len(self.intermediate_lp_info) == 0:
            result_text += "No completed priorities... Is the problem infeasible?"
        for intermediate_result_prev, intermediate_result in zip(
            [None] + self.intermediate_lp_info[:-1], self.intermediate_lp_info
        ):
            priority = intermediate_result["priority"]
            result_text += "\n# Priority {}\n".format(priority)
            upperconstr_range_dict = convert_lists_in_dict(
                intermediate_result["upper_constraint_dict"]
            )
            lowerconstr_range_dict = convert_lists_in_dict(
                intermediate_result["lower_constraint_dict"]
            )
            upper_constraints_df = pd.DataFrame.from_dict(upperconstr_range_dict, orient="index")
            lower_constraints_df = pd.DataFrame.from_dict(lowerconstr_range_dict, orient="index")
            result_text += "## Lower constraints:\n"
            if len(lower_constraints_df):
                result_text += ">### Active variables:\n"
                result_text += add_blockquote(lower_constraints_df.to_markdown()) + "\n"
                result_text += ">### from active constraints:\n"
                for eq, timesteps in group_variables(
                    intermediate_result["active_lower_constraints"]
                ).items():
                    result_text += f">- `{eq}`: {timesteps}\n"
            else:
                result_text += ">No active lower constraints\n"

            result_text += "\n## Upper constraints:\n"
            if len(upper_constraints_df):
                result_text += ">### Active variables:\n"
                result_text += add_blockquote(upper_constraints_df.to_markdown()) + "\n"
                result_text += ">### from active constraints:\n"
                for eq, timesteps in group_variables(
                    intermediate_result["active_upper_constraints"]
                ).items():
                    result_text += f">- `{eq}`: {timesteps}\n"
            else:
                result_text += ">No active upper constraints\n"

            lowerbound_range_dict = convert_lists_in_dict(intermediate_result["upper_bound_dict"])
            upperbound_range_dict = convert_lists_in_dict(intermediate_result["lower_bound_dict"])
            lowerbounds_df = pd.DataFrame.from_dict(lowerbound_range_dict, orient="index")
            upperbounds_df = pd.DataFrame.from_dict(upperbound_range_dict, orient="index")
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

        with open("active_constraints_per_priority.md", "w") as f:
            f.write(result_text)
