import math
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import logging
import matplotlib.dates as mdates
import os

logger = logging.getLogger("rtctools")


class Plotting:
    def pre(self):
        super().pre()
        self.intermediate_results = []

    def plot_goal_results_from_dict(self, result_dict, priority=None):
        self.plot_goals_results(
            result_dict["timeseries_import_times"],
            result_dict["extract_result"],
            result_dict["range_goals"],
            result_dict["min_q_goals"],
            result_dict["priority"],
            result_dict,
        )

    def plot_goal_results_from_self(self, priority=None):
        self.plot_goals_results(
            self.timeseries_import.times,
            self.extract_results(),
            self.min_q_goals,
            self.range_goals,
            priority,
        )

    def plot_goals_results(
        self,
        timeseries_import_times,
        extract_result,
        range_goals,
        min_q_goals,
        priority=None,
        result_dict=None,
    ):
        t = self.times()
        t_datetime = np.array(timeseries_import_times)
        results = extract_result
        result_dict = result_dict

        # TODO: consider making labels prettier, though for debugging this is fine

        # Prepare the plot
        n_plots = len(range_goals + min_q_goals)
        n_cols = math.ceil(n_plots / self.plot_max_rows)
        n_rows = math.ceil(n_plots / n_cols)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 9, n_rows * 3), dpi=80, squeeze=False)
        i_plot = -1

        # Function to apply the general settings used by all goal types
        # existing_labels = []
        def apply_general_settings():
            i_c = math.ceil((i_plot + 1) / n_rows) - 1
            i_r = i_plot - i_c * n_rows

            goal_variable = g[0]
            axs[i_r, i_c].plot(t_datetime, results[goal_variable], label=goal_variable)

            result_dict["positive_effect_dict"].keys()
            prio = result_dict["priority"]
            positive_dict = {
                name.replace(".", "_"): value for name, value in result_dict["positive_effect_dict"].items()
            }
            negative_dict = {
                name.replace(".", "_"): value for name, value in result_dict["negative_effect_dict"].items()
            }
            if goal_variable in positive_dict:
                bounded_at = positive_dict[goal_variable]
                for xr in positive_dict[goal_variable]:
                    # axs[i_r, i_c].vlines(x=t_datetime[int(xr)], color='r', linestyle ='--',ymin= axs[i_r, i_c].get_ylim()[0],ymax= axs[i_r, i_c].get_ylim()[1])
                    label = f"Increase to improve {prio}"
                    if label in axs[i_r, i_c].get_legend_handles_labels()[1]:
                        label = "_nolegend_"
                    # else:
                    #     existing_labels.append(label)
                    axs[i_r, i_c].plot(
                        t_datetime[int(xr)],
                        results[goal_variable][int(xr)],
                        marker=matplotlib.markers.CARETUPBASE,
                        color="r",
                        label=label,
                    )
            if goal_variable in negative_dict:
                for xr in negative_dict[goal_variable]:
                    label = f"Decrease to improve {prio}"
                    if label in axs[i_r, i_c].get_legend_handles_labels()[1]:
                        label = "_nolegend_"
                    # else:
                    #     existing_labels.append(label)
                    axs[i_r, i_c].plot(
                        t_datetime[int(xr)],
                        results[goal_variable][int(xr)],
                        marker=matplotlib.markers.CARETDOWNBASE,
                        color="b",
                        label=label,
                    )
                    # axs[i_r, i_c].vlines(x=t_datetime[int(xr)], color='r', linestyle ='--',ymin= axs[i_r, i_c].get_ylim()[0],ymax= axs[i_r, i_c].get_ylim()[1])
            return i_c, i_r

        def apply_additional_settings(goal_settings):
            add_settings = goal_settings[-1]

            for var in add_settings[1]:
                axs[i_row, i_col].plot(t_datetime, results[var], label=var)
            for var in add_settings[2]:
                axs[i_row, i_col].plot(t_datetime, results[var], linestyle="solid", linewidth="0.5", label=var)
            axs[i_row, i_col].set_ylabel(add_settings[0])
            axs[i_row, i_col].legend()
            axs[i_row, i_col].set_title(f"Priority {goal_settings[4]}")
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
        fig.savefig(f"goal_figures/after_priority_{priority}.png")
        # plt.show()

    def priority_completed(self, priority: int) -> None:
        # Store results required for plotting
        to_store = {
            "extract_result": self.extract_results(),
            "range_goals": self.range_goals,
            "min_q_goals": self.min_q_goals,
            "timeseries_import_times": self.timeseries_import.times,
            "priority": priority,
            "activated_lower_bounds": self.activated_lower_bounds,
            "activated_upper_bounds": self.activated_upper_bounds,
            "textual_constraints": self._textual_constraints,
            "positive_effect_dict": self.positive_effect_dict,
            "negative_effect_dict": self.negative_effect_dict,
            "upper_bound_variable_hits": self.upper_bound_variable_hits,
            "lower_bound_variable_hits": self.lower_bound_variable_hits,
            "new": 'hey!',
            "upper_constraint_dict": self.upper_constraint_dict,
            "lower_constraint_dict": self.lower_constraint_dict,
            "upper_bound_dict": self.upper_bound_dict,
            "lower_bound_dict": self.lower_bound_dict,
        }
        self.intermediate_results.append(to_store)

        # plot current results
        # self.plot_goal_results_from_self(priority)
        super().priority_completed(priority)

    def post(self):
        super().post()

        def list_to_ranges(lst):
            ranges = []
            start = None
            if len(lst) == 1:
                return [(int(lst[0]),int(lst[0]))]
            for i in range(len(lst)):
                if start is None:
                    start = lst[i]
                elif i == len(lst) - 1 or int(lst[i]) + 1 != int(lst[i + 1]):
                    ranges.append((int(start), int(lst[i])))
                    start = None
            return ranges

        def convert_lists_in_dict(dic):
            new_dic = {}
            for key, val in dic.items():
                new_dic[key] = list_to_ranges(val)
            return new_dic

        # Plot all intermediate results

        # Convert effect dicts to ranges
        for intermediate_result in self.intermediate_results:
            self.plot_goal_results_from_dict(intermediate_result)
            priority = intermediate_result["priority"]
            print(f"\nRESULTS FOR PRIORITY {priority}")
            # for i in range(len(intermediate_result['textual_constraints'])):
            #     if intermediate_result['activated_lower_bounds'][i]:
            #         print("hit lower bound for:")
            #         print(intermediate_result['textual_constraints'][i])
            #     if intermediate_result['activated_upper_bounds'][i]:
            #         print("hit upper bound for:")
            #         print(intermediate_result['textual_constraints'][i])
            import pprint

            pos_eff_range_dict = convert_lists_in_dict(intermediate_result["upper_constraint_dict"])
            neg_eff_range_dict = convert_lists_in_dict(intermediate_result["lower_constraint_dict"])
            print("\nVariables :")
            pprint.pprint(pos_eff_range_dict)
            print("\nNegative effect dict constraints:")
            pprint.pprint(neg_eff_range_dict)

            lowerbound_range_dict = convert_lists_in_dict(intermediate_result["upper_bound_dict"])
            upperbound_range_dict = convert_lists_in_dict(intermediate_result["lower_bound_dict"])
            print("\nLowerbounds:")
            pprint.pprint(lowerbound_range_dict)
            print("\nUpperbounds")
            pprint.pprint(upperbound_range_dict)
