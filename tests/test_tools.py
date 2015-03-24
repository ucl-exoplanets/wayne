import unittest

import numpy as np
import numpy.testing
from .. import tools


class Test_crop_spectrum(unittest.TestCase):

    def test_crop_works_exact_limits(self):

        wl = np.arange(10.)
        flux = np.arange(10.)*2

        crop_wl, crop_flux = tools.crop_spectrum(1, 8, wl, flux)

        answer = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        numpy.testing.assert_array_equal(crop_wl, answer)
        numpy.testing.assert_array_equal(crop_flux, answer*2)

    def test_crop_works_near_limits(self):

        wl = np.arange(10.)
        flux = np.arange(10.)*2

        crop_wl, crop_flux = tools.crop_spectrum(0.99, 8.99, wl, flux)

        answer = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        numpy.testing.assert_array_equal(crop_wl, answer)
        numpy.testing.assert_array_equal(crop_flux, answer*2)

    def test_crop_works_mid_points(self):

        wl = np.arange(10.)
        flux = np.arange(10.)*2

        crop_wl, crop_flux = tools.crop_spectrum(1.5, 7.5, wl, flux)

        answer = np.array([2, 3, 4, 5, 6, 7])
        numpy.testing.assert_array_equal(crop_wl, answer)
        numpy.testing.assert_array_equal(crop_flux, answer*2)