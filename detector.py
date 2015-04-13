""" The detector class should map any fields from the telescope to a pixel map on the detector.
"""

import numpy as np
from astropy import units as u
import pandas as pd


class Detector(object):
    def __init__(self):
        # some defaults to show what we need
        self.pixel_array = np.zeros((1024, 1024))
        self.pixel_unit = u.Unit('WFC3IR_Pix', 18*u.micron, doc='Pixel size for the HST WFC3 IR detector')
        self.telescope_area = np.pi * (2.4/2.)**2


class WFC3_IR(Detector):

    def __init__(self):
        Detector.__init__(self)
        # Start with a single 1024x1024 array, add complexity in when we need it.
        self.pixel_array = np.zeros((1024, 1024))
        self.pixel_unit = u.Unit('WFC3IR_Pix', 18*u.micron, doc='Pixel size for the HST WFC3 IR detector')
        self.telescope_area = np.pi * (2.4*u.m/2.)**2

        self.modes_table = self._get_modes()

    def _get_modes(self):
        """ Retrieves table of exposure time for each NSAMP, SAMPSEQ and SUBARRAY type
        :return:
        """
        return pd.read_csv('wfc3_ir_mode_exptime.csv', skiprows=1)
