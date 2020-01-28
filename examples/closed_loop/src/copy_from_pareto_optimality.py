'''
This script can be used to copy content from the pareto_optimality
example to this example if so desired.
More specifically, it will copy (and rename) the input timeseries,
the intial state, the modelica file and the python script.
'''
import os
from copy import copy
from shutil import copyfile

# only run this script when set to True
copy_from_pareto_optimality_example = True

# path to the main folder of open loop implementation
cl_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# path to the main folder of the closed loop implementation
ol_path = os.path.join(cl_path, '..', 'pareto_optimality')
# potential files to copy
files_in = ['BlueRiver.mo', 'timeseries_import.csv',
            'initial_state.csv', 'blue_river_example.py']
# modify the name of the python file in the closed loop folder
files_out = copy(files_in)
files_out[-1] = 'blue_river.py'
# names of the subdirectories containing the files to be copied
folders = ['model', 'input', 'input', 'src']

if copy_from_pareto_optimality_example:
    for n, path_parts in enumerate(zip(folders, files_in, files_out)):
        path_in = os.path.join(ol_path, path_parts[0],  path_parts[1])
        path_out = os.path.join(cl_path, path_parts[0],  path_parts[2])
        if not os.path.isfile(path_out):
            copyfile(path_in, path_out)
            print('copied {} from the open loop implementation.'.format(
                files_in[n]))
