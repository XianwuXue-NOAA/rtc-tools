import os
import runpy
from datetime import datetime, timedelta

from blue_river import BlueRiver

from load_MPC_settings import load_MPC_settings

import numpy as np

import pandas as pd

from rtctools._internal.alias_tools import AliasDict
from rtctools.optimization.timeseries import Timeseries
from rtctools.util import run_optimization_problem

# copies necessary files from the BlueRiver example if set to True
copy_from_BlueRiver_example = True

# hardcoded IO paths:
dirpath = os.path.dirname(os.path.realpath(__file__))
outputfolder = os.path.join(dirpath, '..', 'output')


class BlueRiver_closed_loop(BlueRiver):

    # The modelica file in this example uses the name BlueRiver (not BlueRiverClosedLoop)
    model_name = 'BlueRiver'

    # There will be no available history in the first optimization.
    overrule_history = None

    def __init__(self, *args,
                 len_MPC_timestep=None, MPC_timestep_num=None,
                 horizon=None, history=None, **kwargs):
        super().__init__(*args, **kwargs)
        # check availability of required values
        availability = (x is not None for x in
                        (MPC_timestep_num, len_MPC_timestep, horizon))
        assert all(availability), \
            '''An MPC timestep number is required together with its
               length and the length of the prediction horizon.'''
        if history:
            # Store the old history when available
            # (in the first optimization, this won't be the case).
            self.overrule_history = history
        self.len_MPC_timestep = len_MPC_timestep
        self.MPC_timestep_num = MPC_timestep_num
        self.horizon = horizon

    def times(self, variable=None):
        # Change the timepoints used based on the current MPC_timestep_num
        # and the length of the used prediction horizon (based on the used
        # time resolution used).
        times = super().times(variable)
        ind_start = int(self.MPC_timestep_num * self.len_MPC_timestep)
        ind_end = int((self.MPC_timestep_num+1) * self.horizon)+1
        newtimes = times[ind_start:ind_end]
        return newtimes

    def history(self, ensemble_member):
        # Loads the history in case it won't be overruled.
        history = super().history(ensemble_member)
        if self.overrule_history is not None:
            history = AliasDict(self.alias_relation)
            for variable in self.differentiated_states:
                ts = self.overrule_history[variable]
                # Use the results  of the last computed timestep
                # as initial state.
                history[variable] = Timeseries(ts.times[-1:], ts.values[-1:])
        return history


def run_closed_loop_implementation(len_forecastperiod, len_MPC_timestep, len_MPC_horizon):
    # runs the closed loop imlementation of the BlueRiver system.

    # compute the number of optimisation which can be run with the used settings
    num_MPC_timestep_nums = np.floor(
        (len_forecastperiod - len_MPC_horizon + len_MPC_timestep) / len_MPC_timestep)
    MPC_timestep_nums = np.arange(0, num_MPC_timestep_nums, 1)

    # Initialize history and pandas dataframe.
    history = None
    df = pd.DataFrame()

    # Run an optimization problem for each of the parts of the original timeseries.
    for MPC_timestep_num in MPC_timestep_nums:
        print("starting with optimisation in step: {} of {}".format(
            int(MPC_timestep_num + 1), int(num_MPC_timestep_nums)))
        problem = run_optimization_problem(BlueRiver_closed_loop,
                                           len_MPC_timestep=len_MPC_timestep,
                                           MPC_timestep_num=MPC_timestep_num,
                                           horizon=len_MPC_horizon, history=history)
        print("ready with optimisation in step {} of {}".format(
            int(MPC_timestep_num + 1), int(num_MPC_timestep_nums)))
        # get the results and times of the latest optimization problem
        results = problem.extract_results(0)
        collocation_times = problem.times()

        history = problem.history(0)
        # initial values are stored in history
        for k, v in results.items():
            # Initial derivative results are skipped (length of 1)
            if k not in history and len(v) > 1:
                history[k] = Timeseries(np.array([problem.initial_time]), v[:1])

        # For each variable, the results over a timespan corresponding to
        # len_MPC_timestep (in months) is appended to history
        for k in history.keys():
            history[k] = Timeseries(np.hstack((history[k].times,
                                               collocation_times[1:len_MPC_timestep+1])),
                                    np.hstack((history[k].values,
                                               results[k][1:len_MPC_timestep + 1])))

        # Store all data in a pandas data frame
        datetimestart = datetime(2000, 1, 1)
        times_sec = collocation_times.copy()
        times = [datetimestart + timedelta(seconds=s)
                 for s in times_sec[:len_MPC_timestep+1]]

        for k, v in results.items():
            df['time'] = pd.Series(times)
            # skip results for derivatives and variables which are internally used for
            # optimization by RTC-tools
            if ("_eps" in k) or ("_der" in k):
                continue
            k_name = k
            # rename the variables to a convenient format
            if "." in k:
                for alias in results._AliasDict__relation._aliases[k]:
                    if "." not in alias:
                        k_name = alias
            df[k_name] = pd.Series(v[:len_MPC_timestep+1])

        if int(MPC_timestep_num) == 0:
            dfout = pd.DataFrame()
            dfout = dfout.append(df[0:], sort=False, ignore_index=True)
        else:
            dfout = dfout.append(df[1:], sort=False, ignore_index=True)
        print("ready with step {} of {}".format(
            int(MPC_timestep_num + 1), int(num_MPC_timestep_nums)))
    # compute Qout from Q_spill and Q_turbine
    dfout['TroutLake_Q_out'] = dfout['TroutLake_Q_spill'] + dfout['TroutLake_Q_turbine']
    # Save the accumulated results in outputfileto.
    outputfileto = 'timeseries_export-closedloop.csv'
    dfout.to_csv(os.path.join(outputfolder, outputfileto),
                 index=False, date_format='%Y-%m-%d %H:%M:%S')

    # A final notification that all steps have been completed.
    print("closed loop optimization completed.")


if __name__ == "__main__":

    cwd = os.getcwd()
    os.chdir(dirpath)
    if copy_from_BlueRiver_example:
        # copy BlueRiver files when True (and necessary)
        runpy.run_path('copy_BlueRiver.py')
    # load Model Predictive Control settings from file
    (len_forecastperiod, len_MPC_timestep, len_MPC_horizon) = \
        load_MPC_settings()
    os.chdir(cwd)
    run_closed_loop_implementation(
        len_forecastperiod, len_MPC_timestep, len_MPC_horizon)
