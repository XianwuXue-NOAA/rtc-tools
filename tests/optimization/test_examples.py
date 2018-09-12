import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

import pandas as pd


examples_folder = (Path(__file__) / "../../../examples").resolve()


def run_example(script_path, *output_paths, check_results=False):

    # Read and store the reference output
    reference_results = {}

    if check_results:
        for o in output_paths:

            # Check that the reference outputs exist
            assert o.exists()

            ref = o.parent / (o.name + "_reference")
            o.replace(ref)

            reference_results[o] = ref

    p = subprocess.call([sys.executable, str(script_path)])
    assert p == 0

    if check_results:
        for o in output_paths:
            try:
                df = pd.read_csv(o, index_col=0, parse_dates=True, dayfirst=True)
                ref = pd.read_csv(reference_results[o], index_col=0, parse_dates=True, dayfirst=True)

                assert np.all(df.columns == ref.columns)
                assert np.all(df.index == ref.index)
                assert np.allclose(df.values, ref.values, rtol=0.05, atol=1e-06)
            except Exception:
                # Restore the reference, and rename the current output to "failed"
                failed = o.parent / (o.stem + "_failed" + o.suffix)
                o.replace(failed)
                reference_results[o].rename(o)
                raise
            else:
                # Clean up
                os.remove(o)
                reference_results[o].rename(o)


class TestExamples(unittest.TestCase):

    # NOTE: Most examples have many possible solutions with the same objective
    # value. Checking results on those examples makes little sense. Even when
    # comparing results, we do so while allow a large relative error.

    def test_example_basic(self):
        root = examples_folder / "basic"
        run_example(root / "src" / "example.py",
                    root / "output" / "timeseries_export.csv",
                    check_results=False)

    def test_example_cascading_channels(self):
        root = examples_folder / "cascading_channels"
        run_example(root / "src" / "example.py",
                    root / "output" / "timeseries_export.csv",
                    check_results=False)

    def test_example_ensemble(self):
        root = examples_folder / "ensemble"
        run_example(root / "src" / "example.py",
                    root / "output" / "forecast1" / "timeseries_export.csv",
                    root / "output" / "forecast2" / "timeseries_export.csv",
                    check_results=False)

    def test_example_goal_programming(self):
        root = examples_folder / "goal_programming"
        run_example(root / "src" / "example.py",
                    root / "output" / "timeseries_export.csv")

    def test_example_lookup_table(self):
        root = examples_folder / "lookup_table"
        run_example(root / "src" / "example.py",
                    root / "output" / "timeseries_export.csv")

    def test_example_mixed_integer(self):
        root = examples_folder / "mixed_integer"
        run_example(root / "src" / "example.py",
                    root / "output" / "timeseries_export.csv")

    def test_example_simulation(self):
        root = examples_folder / "simulation"
        run_example(root / "src" / "example.py",
                    root / "output" / "timeseries_export.csv")
