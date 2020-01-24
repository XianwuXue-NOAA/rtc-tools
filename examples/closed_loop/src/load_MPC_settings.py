
import os

import pandas as pd


def load_MPC_settings():
    # Load model predictive control settings from a file.

    dirpath = os.path.dirname(os.path.realpath(__file__))
    MPC_settings_path = os.path.join(dirpath, 'MPC_settings.csv')

    # Measurements are not equidistant in time but are available every month,
    # so the index in the timeseries is used in manipulation of the forecasts.
    MPC_settings = pd.read_csv(MPC_settings_path, sep=';')
    # setting for MPC contain:
    # the length of the forecast period in computing time steps (here: months)
    len_forecastperiod = 60  # known from the input series used in this example
    # the length of the forecasting time step in computing time steps
    len_MPC_timestep = MPC_settings.loc[0]['len_MPC_timestep']
    # the length of the receding horizon in computing time steps
    len_MPC_horizon = MPC_settings.loc[0]['len_MPC_horizon']

    return (len_forecastperiod, len_MPC_timestep, len_MPC_horizon)
