import glob
import os
import shutil

import numpy as np


def copy_files_in_folder(path_to_file, new_folder):
    """Expects 'path_to_file' to be a path with filename, but
    without extension. The function then copies all files with that
    name to the new_folder."""
    for output_file in glob.glob(path_to_file + ".*"):
        output_file_name = os.path.basename(output_file)
        shutil.copyfile(output_file, os.path.join(new_folder, output_file_name))


class ExportResultsEachPriority:
    def priority_completed(self, priority):
        super().priority_completed(priority)

        self.write()
        # Move all output files to a priority-specific folder
        num_len = 3
        subfolder_name = "priority_{:0{}}".format(priority, num_len)
        if self.csv_ensemble_mode:
            ensemble = np.genfromtxt(
                os.path.join(self._input_folder, self.csv_ensemble_basename + ".csv"),
                delimiter=",",
                deletechars="",
                dtype=None,
                names=True,
                encoding=None,
            )
            for ensemble_member in ensemble["name"]:
                new_output_folder = os.path.join(
                    self._output_folder, ensemble_member, subfolder_name
                )
                os.makedirs(new_output_folder, exist_ok=True)
                file_to_copy_stem = os.path.join(
                    self._output_folder, ensemble_member, self.timeseries_export_basename
                )
                copy_files_in_folder(file_to_copy_stem, new_output_folder)
        else:
            new_output_folder = os.path.join(self._output_folder, subfolder_name)
            os.makedirs(new_output_folder, exist_ok=True)
            file_to_copy_stem = os.path.join(self._output_folder, self.timeseries_export_basename)
            copy_files_in_folder(file_to_copy_stem, new_output_folder)
