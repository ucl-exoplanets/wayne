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

        if exptime_table.empty:
            raise WFC3SimSampleModeError("SAMPSEQ = {}, NSAMP={}, SUBARRAY={} is not a permitted combination"
                                         "".format(SAMPSEQ, NSAMP, SUBARRAY))
        else:
            exptime = exptime_table.TIME.values[0]
            return exptime * u.s

    def gen_pixel_array(self, light_sensitive=True):
        """ returns the pixel array as an array of zeroes

        this could return subbary types etc, but lets just keep it out of class for now

        :param light_sensitive: only return the light sensitive parts (neglecting 5 pixel border)
        """

        if light_sensitive:
            return np.zeros((1024-10, 1024-10))
        else:
            return np.zeros((1024, 1024))

    def add_bias_pixels(self, pixel_array):
        """ converts a light sensitive array to one with the 5 pixel border. In future will simulate the function
        of bias pixels but for now returns zeroes.

        input must be a full frame (for now)

        :return:
        """

        full_array = np.zeros((1024, 1024))
        full_array[5:-5, 5:-5] = pixel_array

        return full_array

    def get_read_times(self, NSAMP, SUBARRAY, SAMPSEQ):
        """ Retrieves the time of each sample up the ramp for the mode given
        :param NSAMP:
        :param SUBARRAY:
        :param SAMPSEQ:
        :return:
        """

        if not 1 <= NSAMP <= 15:
            raise WFC3SimSampleModeError("NSAMP must be an integer between 1 and 15, got {}".format(NSAMP))

        mt = self.modes_table

        exptime_table = mt.ix[(mt['SAMPSEQ'] == SAMPSEQ) & (mt['NSAMP'] <= NSAMP) & (mt['SUBARRAY'] == SUBARRAY)]['TIME']

        if exptime_table.empty:
            raise WFC3SimSampleModeError("SAMPSEQ = {}, NSAMP={}, SUBARRAY={} is not a permitted combination"
                                         "".format(SAMPSEQ, NSAMP, SUBARRAY))

        return np.array(exptime_table) * u.s

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