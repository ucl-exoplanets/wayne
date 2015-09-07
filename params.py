""" Holds any run parameters and in future will be able to load new ones from
 a par file
"""

import os
import os.path

import numpy as np

_rootdir = os.path.dirname(__file__)  # get current directory location
_data_dir = os.path.join(_rootdir, 'data')

# darks, flats etc
_calb_dir = os.path.join(os.environ['PYSYN_CDBS'], 'calb', 'wfc3')

seed = np.random.get_state()[1][0]