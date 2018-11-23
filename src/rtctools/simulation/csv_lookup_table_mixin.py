from typing import Dict

from rtctools._internal.csv_lookup_table_provider import CSVLookupTableProvider
from rtctools._internal.lookup_table import LookupTable

from .simulation_problem import SimulationProblem


class CSVLookupTableMixin(CSVLookupTableProvider, SimulationProblem):
    """Adds lookup tables to your SimulationProblem.

    During preprocessing, the CSV files located inside the ``lookup_tables``
    subfolder are read. In every CSV file, the first column contains the output
    of the lookup table.  Subsequent columns contain the input variables.

    Cubic B-Splines are used to turn the data points into continuous lookup
    tables.

    Optionally, a file ``curvefit_options.ini`` may be included inside the
    ``lookup_tables`` folder. This file contains, grouped per lookup table, the
    following options:

    * monotonicity:
        * is an integer, magnitude is ignored
        * if positive, causes spline to be monotonically increasing
        * if negative, causes spline to be monotonically decreasing
        * if 0, leaves spline monotonicity unconstrained

    * curvature:
        * is an integer, magnitude is ignored
        * if positive, causes spline curvature to be positive (convex)
        * if negative, causes spline curvature to be negative (concave)
        * if 0, leaves spline curvature unconstrained

    .. note::

        Currently only one-dimensional lookup tables are fully supported.
        Support for two-dimensional lookup tables is experimental.

    :cvar csv_delimiter:
        Column delimiter used in CSV files.  Default is ``,``.
    :cvar csv_lookup_table_debug:
        Whether to generate plots of the spline fits.  Default is ``false``.
    :cvar csv_lookup_table_debug_points:
        Number of evaluation points for plots.  Default is ``100``.
    """

    def lookup_tables(self) -> Dict[str, LookupTable]:
        """Get a dict of LookupTables found by CSVLookupTableMixin"""
        return self._provided_lookup_tables
