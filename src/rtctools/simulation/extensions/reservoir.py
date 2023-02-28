from typing import Union

import numpy as np

import pandas as pd


# assume csv separated by semi colon, make this generic so we can also have commas.
def readReservoirData(
    reservoirs_csv_path,
    volume_level_csv_paths,
    spillwaydischarge_csv_path,
    volume_area_csv_path
):
    res_df = pd.read_csv(reservoirs_csv_path, sep=",", index_col=0)
    vh_data_df = pd.read_csv(volume_level_csv_paths, sep=";", index_col=0)  # should be csv_path not paths for generic
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
