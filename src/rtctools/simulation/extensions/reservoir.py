from typing import Union

import numpy as np

import pandas as pd


# assume csv separated by semi colon, make this generic so we can also have commas.
def readReservoirData(
    reservoirs_csv_path,
    volume_level_csv_path,
    spillwaydischarge_csv_path,
    volume_area_csv_path
):
    r"""
    This function reads the CSV files provided as input and converts the reservoir data, volume-level tables and
    volume-area tables and optionally the spill-way-discharge table to dataFrames.

    Parameters
    ----------
    reservoirs_csv_path :
        Path to csv file that contains the columns Name, surcharge, fullsupply, crestheight, volume_min, volume_max,
        q_turbine_max, q_spill_max, lowflowlocation

    volume_level_csv_path :
        Path to csv file that contains the columns ReservoirName, Storage_m3, Elevation_m

    spillwaydischarge_csv_path :
        Path to csv file that contains the columns ReservoirName, Elevation_m, Discharge_m3s

    volume_area_csv_path :
        Path to csv file that contains the columns ReservoirName, Storage_m3, Area_m2

    Returns
    -------
    reservoirs :
        Returns a dictionary of lookup tables for the reservoir
    """
    res_df = pd.read_csv(reservoirs_csv_path, sep=",", index_col=0)
    vh_data_df = pd.read_csv(volume_level_csv_path, sep=";", index_col=0)
    va_data_df = pd.read_csv(volume_area_csv_path, sep=";", index_col=0)

    # do we always need spillway discharge? make this generic

    spillwaydischarge_df = pd.read_csv(spillwaydischarge_csv_path, sep=",", index_col=0)
    # compute setpoints as volumes, using the vh_data_df
    reservoirs = {}
    for index, row in res_df.iterrows():
        reservoirs[index] = Reservoir(
            index,
            vh_data_df.loc[index],
            va_data_df.loc[index],
            spillwaydischarge_df.loc[index], row)
        # if user gives set point
        #     for names
        #         reservoirs[index].set_Vsetpoints(name)
        reservoirs[index].set_Vsetpoints()
    return reservoirs


class Reservoir():

    def __init__(self, name, vh_data, va_data, spillwaydischargedata, reservoir_properties):
        self.__vh_lookup = vh_data
        self.__va_lookup = va_data

        self.__spillwaydischargelookup = spillwaydischargedata
        self.name = name
        self.properties = reservoir_properties

    def level_to_volume(self, levels: Union[float, np.ndarray]):
        r'''
        Returns the water levels in the reservoir for elevation (m) and storage (m$^3$)
        by one-dimensional linear interpolation for given volume and storage volume-level table.

        Parameters
        ----------

        Returns
        -------
        levels :
            Water level [m]
        '''
        return np.interp(levels, self.__vh_lookup['Elevation_m'], self.__vh_lookup['Storage_m3'])

    def volume_to_level(self, volumes: Union[float, np.ndarray]):
        return np.interp(volumes, self.__vh_lookup['Storage_m3'], self.__vh_lookup['Elevation_m'])

    def volume_to_area(self, volumes: Union[float, np.ndarray]):
        return np.interp(volumes, self.__va_lookup['Storage_m3'], self.__va_lookup['Area_m2'])

    def level_to_area(self, levels: Union[float, np.ndarray]):
        volume_interp = np.interp(levels, self.__vh_lookup['Elevation_m'], self.__vh_lookup['Storage_m3'])
        return np.interp(volume_interp, self.__va_lookup['Storage_m3'], self.__va_lookup['Area_m2'])

    def volume_to_spillwaydischarge(self, volumes: Union[float, np.ndarray]):
        levels = self.volume_to_level(volumes)
        return np.interp(
            levels,
            self.__spillwaydischargelookup['Elevation_m'],
            self.__spillwaydischargelookup['Discharge_m3s'])

    def set_Vsetpoints(self):
        # define interpolated volume setpoints corresponding to heights used in goals:
        # make this generic, call function three times with name as argument.
        self.Vsetpoints = {}
        self.Vsetpoints['surcharge'] = self.level_to_volume(self.properties['surcharge'])
        self.Vsetpoints['fullsupply'] = self.level_to_volume(self.properties['fullsupply'])
        self.Vsetpoints['crestheight'] = self.level_to_volume(self.properties['crestheight'])
