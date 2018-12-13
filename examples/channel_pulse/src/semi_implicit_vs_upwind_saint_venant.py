"""Compare semi implicit inertial wave with full, upwind saint-venant"""
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import numpy as np

# Map root_dir
root_dir = Path(__file__).parents[1].resolve()

# Import Data
thetas = [0.0, 0.5, 1.0]
formulations = [
    "inertial_wave",
    "inertial_wave_semi_implicit",
    "saint_venant",
    "saint_venant_upwind",
]
outputs = {
    f: {
        t: np.recfromcsv(
            root_dir / f"output.{t}/timeseries_export_{f}.csv", encoding=None
        )
        for t in thetas
    }
    for f in formulations
}

# Get times as datetime objects
times = list(
    map(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S"),
        outputs["inertial_wave"][1.0]["time"],
    )
)
# Generate Plot
n_subplots = 2
fig, axarr = plt.subplots(n_subplots, sharex=True, figsize=(10, 3 * n_subplots))
axarr[0].set_title(
    "Comparison of semi-implicit inertial wave equations with full, upwind saint-venant"
)

# Upper subplot
axarr[0].set_ylabel("Flow Rate [m³/s]")
axarr[0].plot(
    times,
    outputs["inertial_wave_semi_implicit"][1.0]["channel_q_dn"],
    label="Downstream\n(semi-implicit inertial wave)",
)
axarr[0].plot(
    times,
    outputs["saint_venant_upwind"][1.0]["channel_q_dn"],
    label="Downstream\n(full, upwind saint-venant)",
)
axarr[0].plot(
    times,
    outputs["inertial_wave"][1.0]["channel_q_up"],
    label="Upstream",
    linestyle="--",
    color="grey",
)

# Lower subplot
axarr[1].set_ylabel("Water Level [m]")
axarr[1].plot(
    times,
    outputs["inertial_wave_semi_implicit"][1.0]["channel_h_up"],
    label="Upstream\n(semi-implicit inertial wave)",
)
axarr[1].plot(
    times,
    outputs["saint_venant_upwind"][1.0]["channel_h_up"],
    label="Upstream\n(full, upwind saint-venant)",
)
axarr[1].plot(
    times,
    outputs["inertial_wave"][1.0]["channel_h_dn"],
    label="Downstream",
    linestyle="--",
    color="grey",
)

# Format bottom axis label
axarr[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

# Shrink margins
fig.tight_layout()

# Shrink each axis and put a legend to the right of the axis
for i in range(n_subplots):
    box = axarr[i].get_position()
    axarr[i].set_position([box.x0, box.y0, box.width * 0.75, box.height])
    axarr[i].legend(
        loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, prop={"size": 8}
    )

plt.autoscale(enable=True, axis="x", tight=True)

# Output Plot
plt.savefig(Path(__file__).with_suffix(".svg"))
