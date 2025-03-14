from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

# Import Data
data_path = "../../../examples/lookup_table/reference_output/timeseries_export.csv"
results = np.genfromtxt(
    data_path, delimiter=",", encoding=None, dtype=None, names=True, case_sensitive="lower"
)

# Get times as datetime objects
times = [datetime.strptime(x, "%Y-%m-%d %H:%M:%S") for x in results["time"]]

# Generate Plot
n_subplots = 2
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(8, 3 * n_subplots))
axarr[0].set_title("Water Volume and Discharge")

# Upper subplot
axarr[0].set_ylabel("Water Volume [m³]")
axarr[0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
axarr[0].plot(times, results["storage_v"], label="Storage", linewidth=2, color="b")
axarr[0].plot(times, results["v_max"], label="Storage Max", linewidth=2, color="r", linestyle="--")
axarr[0].plot(times, results["v_min"], label="Storage Min", linewidth=2, color="g", linestyle="--")

# Lower Subplot
axarr[1].set_ylabel("Flow Rate [m³/s]")
axarr[1].scatter(times, results["q_in"], linewidth=1, color="g")
axarr[1].scatter(times, results["q_release"], linewidth=1, color="r")
# add horizontal lines to the left of these dots, to indicate that the value is attained over an
# entire timestep:
axarr[1].step(times, results["q_in"], linewidth=2, where="pre", label="Inflow", color="g")
axarr[1].step(times, results["q_release"], linewidth=1, where="pre", label="Release", color="r")

# Format bottom axis label
axarr[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

# Shrink margins
fig.tight_layout()

# Shrink each axis and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    axarr[i].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

plt.autoscale(enable=True, axis="x", tight=True)

# Output Plot
plt.show()
