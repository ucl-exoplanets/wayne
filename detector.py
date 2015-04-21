""" The detector class should map any fields from the telescope to a pixel map on the detector.
"""

import numpy as np
from astropy import units as u
import pandas as pd
import os.path

import params


class WFC3_IR(object):

    def __init__(self):

        # Start with a single 1024x1024 array, add complexity in when we need it.
        self._pixel_array = np.zeros((1024, 1024))
        self.pixel_unit = u.Unit('WFC3IR_Pix', 18*u.micron, doc='Pixel size for the HST WFC3 IR detector')
        self.telescope_area = np.pi * (2.4*u.m/2.)**2

        # General Info
        self.telescope = 'HST'
        self.instrument = 'WFC3'
        self.detector_type = 'IR'

        # Init
        self.modes_table = self._get_modes()

    def exptime(self, NSAMP, SUBARRAY, SAMPSEQ):
        """ Retrieves the total exposure time for the modes given
        :return:
        """

        mt = self.modes_table

        exptime_table = mt.ix[(mt['SAMPSEQ'] == SAMPSEQ) & (mt['NSAMP'] == NSAMP) & (mt['SUBARRAY'] == SUBARRAY)]

        try:
            exptime = exptime_table.TIME.values[0]  # 0 as we want a single value not a single value in an array
        except IndexError:  # empty list
            raise WFC3SimSampleModeError("SAMPSEQ = {}, NSAMP={}, SUBARRAY={} is not a permitted combination"
                                         "".format(SAMPSEQ, NSAMP, SUBARRAY))

        return exptime * u.s

    def gen_pixel_array(self):
        # this could return subbary types etc, lets just keep it out of class for now
        return self._pixel_array.copy()

    def _get_modes(self):
        """ Retrieves table of exposure time for each NSAMP, SAMPSEQ and SUBARRAY type
        :return:
        """

        modes_table = pd.read_csv(os.path.join(params._data_dir, 'wfc3_ir_mode_exptime.csv'), skiprows=1, dtype={
            'SUBARRAY': np.int64, 'SAMPSEQ': np.object, 'NSAMP': np.int64, 'TIME':np.float},
                                  thousands=',')

        return modes_table

    def num_exp_per_buffer(self, NSAMP, SUBARRAY):
        """ max number of exposures that can be taken before buffer dumping
        :return:
        """

        hard_limit = 304  # headers pg 208

        headers_per_exp = NSAMP + 1  # + 1 for zero read

        # 2 full frame (1024) 16 sample exposures
        total_allowed_reads = 2*16*(1024/SUBARRAY)

        if total_allowed_reads > hard_limit:
            total_allowed_reads = hard_limit

        num_exp = int(np.floor(total_allowed_reads / headers_per_exp))

        return num_exp


class WFC3SimException(BaseException):
    pass


class WFC3SimSampleModeError(WFC3SimException):
    pass