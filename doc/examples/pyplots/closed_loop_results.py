from datetime import datetime

import matplotlib.pyplot as plt

import numpy as np

# If False, the input data will be cropped such that the timepoints
# equal those in the results.
plot_full_input = False

# Import Data
input_path = "../../../examples/closed_loop/input/timeseries_import.csv"
input_data = np.recfromcsv(input_path, encoding=None)
results_path = "../../../examples/closed_loop/output/timeseries_export-closedloop.csv"
results = np.recfromcsv(results_path, encoding=None)

# Get times as datetime objects
times = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), results["time"]))
input_times = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), input_data["utc"]))

if not plot_full_input:
    # Crop the input_data such that input times overlap with the times in the results
    input_data = input_data[[time in times for time in input_times]]
    input_times = times

# Generate Plot
n_subplots = 2
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
axarr[0].set_title("Volume and flow rate")

# Upper subplot (volume)
axarr[0].set_ylabel("Volume in reservoir [m³]")
# flood control volume targets copied from the example.py file
Vlimits_c = (505066826.8963750, 715419465.7775500)
# volume targets for recreational boating, copied from the example.py file
Vlimits_b = (616740918.8, 629075737.1)
len_itimes = len(input_times)

axarr[0].plot(input_times, [Vlimits_c[0]]*len_itimes, label="$V_{min}$ (flood control)",
              linewidth=2, linestyle='--', color="m")
axarr[0].plot(input_times, [Vlimits_c[1]]*len_itimes, label="$V_{max}$ (flood control)",
              linewidth=2, linestyle='--', color="m")
axarr[0].plot(input_times, [Vlimits_b[0]]*len_itimes, label="$V_{min}$ (boating)",
              linewidth=2, linestyle=':', color="g")
axarr[0].plot(input_times, [Vlimits_b[1]]*len_itimes, label="$V_{max}$ (boating)",
              linewidth=2, linestyle=':', color="g")
axarr[0].plot(times, results["troutlake_v"], label="$V$ (reservoir)", linewidth=2, marker='.', color="b")

# Lower Subplot
axarr[1].set_ylabel("Flow Rate [m³/s]")
# flow rate targets copied from the example.py file
Qlimits = (7.044311236, 23.48103745)
axarr[1].plot(input_times, [Qlimits[0]]*len_itimes, label="$Q_{min}$ (ecology)",
              linestyle='--', linewidth=2, color="m")
axarr[1].plot(input_times, [Qlimits[1]]*len_itimes, label="$Q_{max}$ (ecology)",
              linestyle='--', linewidth=2, color="m")
axarr[1].plot(input_times, input_data["rafting_qmin"], label="$Q_{min}$ (rafting)",
              linestyle=':', linewidth=2, color="g")
axarr[1].plot(input_times, input_data["rafting_qmax"], label="$Q_{max}$ (rafting)",
              linestyle=':', linewidth=2, color="g")
axarr[1].plot(times, results["rivercity_q"], label="$Q$ (River City)", linewidth=2, marker='.', color="orange")
axarr[1].plot(times, results["troutlake_q_out"], label="$Q$ (Trout Lake)", linestyle='-.',
              linewidth=2, marker='.', color="c")

# Shrink margins
fig.tight_layout()


# Use a function to list the last 1 and 2 entries first in the legend
def legor(lenh):
    return ([lenh - 1] + list(range(lenh - 1)),
            list(range(lenh - 1, lenh - 3, -1)) + list(range(lenh - 2)))


# Shrink each axis and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.75, box.height])
    handles, labels = axarr[i].get_legend_handles_labels()
    legend_order = legor(len(handles))[i]
    axarr[i].legend([handles[idx] for idx in legend_order], [labels[idx] for idx in legend_order],
                    loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

plt.autoscale(enable=True, axis="x", tight=True)
# Output Plot
plt.show()
