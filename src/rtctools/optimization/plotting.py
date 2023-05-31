import copy
import logging
import math
import os

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


logger = logging.getLogger("rtctools")


class PlotGoals:
    plotting_and_active_constraints = True
    lam_tol = 0.1

    def pre(self):
        super().pre()
        self.intermediate_results = []

    def plot_goal_results_from_dict(self, result_dict, results_dict_prev=None):
        self.plot_goals_results(result_dict, results_dict_prev)
    def plot_goal_results_from_dict(self, result_dict, results_dict_prev=None):
        self.plot_goals_results(result_dict, results_dict_prev)

    def plot_goal_results_from_self(self, priority=None):
        result_dict = {
            "timeseries_import_times": self.timeseries_import.times,
            "extract_result": self.extract_results(),
            "min_q_goals": self.min_q_goals,
            "range_goals": self.range_goals,
            "priority": priority,
        }
        self.plot_goals_results(result_dict)

    def plot_goals_results(self, result_dict, results_dict_prev=None):
        timeseries_import_times = result_dict["timeseries_import_times"]
        extract_result = result_dict["extract_result"]
        range_goals = result_dict["range_goals"]
        min_q_goals = result_dict["min_q_goals"]
        priority = result_dict["priority"]
        result_dict = {
            "timeseries_import_times": self.timeseries_import.times,
            "extract_result": self.extract_results(),
            "min_q_goals": self.min_q_goals,
            "range_goals": self.range_goals,
            "priority": priority,
        }
        self.plot_goals_results(result_dict)

    def plot_goals_results(self, result_dict, results_dict_prev=None):
        timeseries_import_times = result_dict["timeseries_import_times"]
        extract_result = result_dict["extract_result"]
        range_goals = result_dict["range_goals"]
        min_q_goals = result_dict["min_q_goals"]
        priority = result_dict["priority"]

        t = self.times()
        t_datetime = np.array(timeseries_import_times)
        results = extract_result

        # TODO: consider making labels prettier, though for debugging this is fine

        # Prepare the plot
        n_plots = len(range_goals + min_q_goals)
        n_cols = math.ceil(n_plots / self.plot_max_rows)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False
        )
        fig.suptitle("Results after optimizing until priority {}".format(priority), fontsize=14)
        fig, axs = plt.subplots(
            nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False
        )
        fig.suptitle("Results after optimizing until priority {}".format(priority), fontsize=14)
        i_plot = -1

        # Function to apply the general settings used by all goal types
        # existing_labels = []
        def apply_general_settings():
            i_c = math.ceil((i_plot + 1) / n_rows) - 1
            i_r = i_plot - i_c * n_rows

            goal_variable = g[0]
            axs[i_r, i_c].plot(t_datetime, results[goal_variable], label=goal_variable)

            if results_dict_prev:
                results_prev = results_dict_prev["extract_result"]
                axs[i_r, i_c].plot(
                    t_datetime,
                    results_prev[goal_variable],
                    label=goal_variable + " at previous priority optimization",
                    color="gray",
                    linestyle="dotted",
                )

            if results_dict_prev:
                results_prev = results_dict_prev["extract_result"]
                axs[i_r, i_c].plot(
                    t_datetime,
                    results_prev[goal_variable],
                    label=goal_variable + " at previous priority optimization",
                    color="gray",
                    linestyle="dotted",
                )

            prio = result_dict["priority"]

            def add_variable_effects(constraints):
                if goal_variable in constraints:
                    for xr in constraints[goal_variable]["timesteps"]:
                        if constraints[goal_variable]["effect_direction"] == "+":
                            modification = "Increase"
                            marker_type = matplotlib.markers.CARETUPBASE
                            marker_color = "g"

                        else:
                            modification = "Decrease"
                            marker_type = matplotlib.markers.CARETDOWNBASE
                            marker_color = "r"

                        label = "{} {} to improve {}".format(modification, goal_variable, prio)
                        if label in axs[i_r, i_c].get_legend_handles_labels()[1]:
                            label = "_nolegend_"
                        axs[i_r, i_c].plot(
                            t_datetime[int(xr)],
                            results[goal_variable][int(xr)],
                            marker=marker_type,
                            color=marker_color,
                            label=label,
                            markersize=5,
                            alpha=0.6,
                        )

            upper_constraints = {
                name.replace(".", "_"): value
                for name, value in result_dict["upper_constraint_dict"].items()
                name.replace(".", "_"): value
                for name, value in result_dict["upper_constraint_dict"].items()
            }
            lower_constraints = {
                name.replace(".", "_"): value
                for name, value in result_dict["lower_constraint_dict"].items()
                name.replace(".", "_"): value
                for name, value in result_dict["lower_constraint_dict"].items()
            }
            # add_variable_effects(upper_constraints)
            # add_variable_effects(lower_constraints)

            return i_c, i_r

        def apply_additional_settings(goal_settings):
            add_settings = goal_settings[-1]

            for var in add_settings[1]:
                axs[i_row, i_col].plot(t_datetime, results[var], label=var)
            for var in add_settings[2]:
                axs[i_row, i_col].plot(
                    t_datetime, results[var], linestyle="solid", linewidth="0.5", label=var
                )
                axs[i_row, i_col].plot(
                    t_datetime, results[var], linestyle="solid", linewidth="0.5", label=var
                )
            axs[i_row, i_col].set_ylabel(add_settings[0])
            axs[i_row, i_col].legend()
            if len(goal_settings)==8:
                axs[i_row, i_col].set_title(
                    "Goal for {} (active from priority {})".format(goal_settings[0], goal_settings[4])
                )
            else:
                axs[i_row, i_col].set_title(
                    "Goal for {} (active from priority {})".format(goal_settings[0], goal_settings[1])
                )

            dateFormat = mdates.DateFormatter("%d%b%H")
            axs[i_row, i_col].xaxis.set_major_formatter(dateFormat)
            axs[i_row, i_col].grid(which="both", axis="x")

        # Add plots needed for range goals
        for g in sorted(self.range_goals, key=lambda goal: goal[4]):
            i_plot += 1

            i_col, i_row = apply_general_settings()

            if g[1] == "parameter":
                target_min = np.full_like(t, 1) * self.parameters(0)[g[2]]
                target_max = np.full_like(t, 1) * self.parameters(0)[g[3]]
            elif g[1] == "timeseries":
                target_min = self.get_timeseries(g[2]).values
                target_max = self.get_timeseries(g[3]).values
            else:
                logger.error("Target type {} not known.".format(g[1]))
                raise

            if np.array_equal(target_min, target_max, equal_nan=True):
                axs[i_row, i_col].plot(t_datetime, target_min, "r--", label="Target")
            else:
                axs[i_row, i_col].plot(t_datetime, target_min, "r--", label="Target min")
                axs[i_row, i_col].plot(t_datetime, target_max, "r--", label="Target max")

            apply_additional_settings(g)

        # Add plots needed for minimization of discharge
        for g in min_q_goals:
            i_plot += 1

            i_col, i_row = apply_general_settings()

            apply_additional_settings(g)

        # TODO: this should be expanded when there are more columns
        for i in range(0, n_cols):
            axs[n_rows - 1, i].set_xlabel("Time")
        os.makedirs("goal_figures", exist_ok=True)
        fig.tight_layout()
        fig.savefig("goal_figures/after_priority_{}.png".format(priority))
        # plt.show()

    def priority_completed(self, priority: int) -> None:
        # Store results required for plotting
        to_store = {
            "extract_result": self.extract_results(),
            "range_goals": self.range_goals,
            "min_q_goals": self.min_q_goals,
            "timeseries_import_times": self.timeseries_import.times,
            "priority": priority,
            "upper_constraint_dict": self.upper_constraint_dict,
            "lower_constraint_dict": self.lower_constraint_dict,
            "upper_bound_dict": self.upper_bound_dict,
            "lower_bound_dict": self.lower_bound_dict,
            "active_lower_constraints": self.active_lower_constraints,
            "active_upper_constraints": self.active_upper_constraints,
        }
        self.intermediate_results.append(to_store)
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
        if len(self.intermediate_results) == 0:
            result_text += "No completed priorities... Is the problem infeasible?"
        for intermediate_result_prev, intermediate_result in zip(
            [None] + self.intermediate_results[:-1], self.intermediate_results
        ):
            self.plot_goal_results_from_dict(intermediate_result, intermediate_result_prev)
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
