""" The detector class should map any fields from the telescope to a pixel map on the detector.
"""

import numpy as np
import quantities as pq


class Detector(object):
    def __init__(self):
        # some defaults to show what we need
        self.pixel_array = np.zeros((1024, 1024))
        self.pixel_unit = 15*pq.micron


class WFC3_IR(Detector):

    def __init__(self):
        Detector.__init__(self)
        # Start with a single 1024x1024 array, add complexity in when we need it.
        self.pixel_array = np.zeros((1024, 1024))
        self.pixel_unit = pq.length.UnitLength('WFC3 IR Pixel', 18*pq.micron, 'WFC3IR_Pix',
                                               doc='Pixel size for the HST WFC3 IR detector')