from pathlib import Path

from rtctools.simulation.extensions.reservoir_csv import ReadReservoirData

from test_case import TestCase

DATA_DIR = Path(__file__).resolve().parent / "data" / "reservoir"


class TestReservoir(TestCase):
    def setUp(self):
        self.reservoirs_csv_path = DATA_DIR / "reservoirs.csv"
        self.volume_level_csv_path = DATA_DIR / "volumelevel.csv"
        self.spillwaydischarge_csv_path = DATA_DIR / "spillwaydischarge.csv"
        self.volume_area_csv_path = DATA_DIR / "VolumeArea.csv"

    def get_reservoir_data(self):
        return ReadReservoirData(
            reservoirs_csv_path=self.reservoirs_csv_path,
            volume_area_csv_path=self.volume_area_csv_path,
            spillwaydischarge_csv_path=self.spillwaydischarge_csv_path,
            volume_level_csv_path=self.volume_level_csv_path)

    def test_read_reservoir_data(self):
        reservoir_data = self.get_reservoir_data()
        name = reservoir_data['FrenchMeadowsReservoir'].name
        self.assertEqual(name, "FrenchMeadowsReservoir")
