""" The detector class should map any fields from the telescope to a pixel map on the detector.
"""

import numpy as np
import quantities as pq


class Detector(object):
    pass  # build this after WFC3_IR i.e. generalising when i know more about whats involved
    #
    # def __init__(self):
    #
    #     # Start with a single 256x256 array, add complexity in when we need it.
    #     self.pixel_array = np.zeros((256, 256))


class WFC3_IR(Detector):

    def __init__(self):
        # Start with a single 256x256 array, add complexity in when we need it.
        self.pixel_array = np.zeros((256, 256))
        self.pixel_unit = pq.length.UnitLength('WFC3 IR Pixel', 18*pq.micron, 'WFC3IR_Pix',
                                               doc='Pixel size for the HST WFC3 IR detector')