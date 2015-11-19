""" Holds any run parameters and in future will be able to load new ones from
 a par file
"""

import os
import os.path

import numpy as np

_rootdir = os.path.dirname(__file__)  # get current directory location
_data_dir = os.path.join(_rootdir, 'data')

class WayneDataError(Exception):
    pass

# darks, flats etc
try:
    _calb_dir = os.path.join(os.environ['WAYNE_DATA'], 'calb', 'wfc3')
except KeyError:
    _calb_dir = os.path.join(os.path.expanduser('~'), '.wayne')

    if not os.path.exists(_calb_dir):
        raise WayneDataError('Calibration files not given. Please add them to {} or set '
            'WAYNE_DATA (see https://github.com/ucl-exoplanets/Wayne)'.format( _calb_dir))

seed = np.random.get_state()[1][0]