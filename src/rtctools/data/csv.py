import csv
import logging
import sys
from datetime import datetime

import numpy as np

logger = logging.getLogger("rtctools")

csv.field_size_limit(1000000)


def load(fname, delimiter=',', with_time=False):
    """
    Check delimiter of csv and read contents to an array. Assumes no date-time conversion needed.

    :param fname:     Filename.
    :param delimiter: CSV column delimiter.
    :param with_time: Whether the first column is expected to contain time stamps.

    :returns: A named numpy array with the contents of the file.
    """

    # Check delimiter of csv file. If semicolon, check if decimal separator is
    # a comma.
    comma_decimal = False
    if delimiter == ';':
        with open(fname, 'rb') as csvfile:
            # Read the first line, this should be a header.
            sample_csvfile = csvfile.readline()
            # We actually only need one number to evaluate if commas are used as decimal separator, but
            # certain csv writers don't use a decimal when the value has no meaningful decimal
            # (e.g. 12.0 becomes 12) so we read the next 1024 bytes to make sure we catch a number.
            sample_csvfile = csvfile.read(1024)
            # Count the commas
            comma_decimal = sample_csvfile.count(b',') > 0

    # Read the csv file and convert to array
    try:
        # We do not use NumPy's genfromtxt, as it is difficult to read in data
        # with different dtype (str, float). Using dtype=None does not work,
        # as that will result in empty columns being read in as boolean.
        with open(fname, 'r', encoding='utf-8') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=delimiter, quotechar='#')

            column_names = next(csvreader)

            data = []
            for r, row in enumerate(csvreader):
                if not len(row) == len(column_names):
                    raise ValueError("Row {} has {} values, but header has {} names".format(
                        r + 1, len(row), len(column_names)))

                # Convert elements to correct type
                if with_time:
                    row[0] = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')

                for i in range(int(with_time), len(row)):
                    val = row[i].strip()

                    if comma_decimal:
                        val = val.replace(',', '.')

                    row[i] = np.nan if not val else float(val)

                data.append(tuple(row))

        dtypes = ['f8'] * len(column_names)
        if with_time:
            dtypes[0] = 'O'

        ret = np.array(data, dtype=list(zip(column_names, dtypes)))

        # Make sure we are compatible in return type with np.genfromtxt. That
        # is, we do not return an array when there is only a single row.
        if len(data) == 1:
            return ret[0]
        else:
            return ret

    except ValueError as e:
        type, value, traceback = sys.exc_info()
        logger.error(
            'CSVMixin: converter of csv reader failed on {}: {}'.format(fname, value))
        raise ValueError("Error converting value or datetime in {}".format(fname)) from e


def save(fname, data, delimiter=',', with_time=False):
    """
    Write the contents of an array to a csv file.

    :param fname:     Filename.
    :param data:      A named numpy array with the data to write.
    :param delimiter: CSV column delimiter.
    :param with_time: Whether to output the first column of the data as time stamps.
    """
    if with_time:
        data['time'] = [t.strftime("%Y-%m-%d %H:%M:%S") for t in data['time']]
        fmt = ['%s'] + (len(data.dtype.names) - 1) * ['%f']
    else:
        fmt = len(data.dtype.names) * ['%f']

    np.savetxt(fname, data, delimiter=delimiter, header=delimiter.join(
        data.dtype.names), fmt=fmt, comments='')
