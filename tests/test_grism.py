import unittest

import numpy as np
import numpy.testing
import astropy.units as u

from .. import grism
from unittest_tools import assertArrayAlmostEqual


class Test_G141_Grism(unittest.TestCase):
    def setUp(self):
        self.g141_grism = grism.G141()

    def test__init__(self):
        grism.G141()  # pass if no exceptions

    def test_flux_to_psf_raises_TypeError_given_no_units(self):
        with self.assertRaises(TypeError):
            self.g141_grism.flux_to_psf(1.2, 1.)

    def test_flux_to_psf_at_sample_points(self):
        # Verified using MCMC
        psf1 = self.g141_grism.flux_to_psf(1.2 * u.micron, 1.)
        self.assertAlmostEqual(psf1.flux, 1., 9)
        self.assertAlmostEqual(psf1.amplitude, 0.9033, 4)
        self.assertAlmostEqual(psf1.stddev, 0.4416, 4)

        psf2 = self.g141_grism.flux_to_psf(1.7 * u.micron, 1.5)

        self.assertAlmostEqual(psf2.flux, 1.5, 9)
        self.assertAlmostEqual(psf2.amplitude, 1.1560, 4)
        self.assertAlmostEqual(psf2.stddev, 0.5177, 4)

    def test_flux_to_psf_between_sample_points_interpolation(self):
        psf1 = self.g141_grism.flux_to_psf(1.15 * u.micron, 1.)
        self.assertAlmostEqual(psf1.flux, 1., 9)
        self.assertAlmostEqual(psf1.amplitude, 0.9125, 4)
        self.assertAlmostEqual(psf1.stddev, 0.4372, 4)

        psf2 = self.g141_grism.flux_to_psf(1.65 * u.micron, 1.5)
        self.assertAlmostEqual(psf2.flux, 1.5, 9)
        self.assertAlmostEqual(psf2.amplitude, 1.1767, 4)
        self.assertAlmostEqual(psf2.stddev, 0.5085, 4)

    def test_get_pixel_wl(self):
        self.assertAlmostEqual(self.g141_grism.get_pixel_wl(50, 50, 100, 50), 11222.2, 1)
        self.assertAlmostEqual(self.g141_grism.get_pixel_wl(50, 50, 200, 50), 15748.6, 1)
        self.assertAlmostEqual(self.g141_grism.get_pixel_wl(50, 50, 100, 51), 11222.7, 1)
        self.assertAlmostEqual(self.g141_grism.get_pixel_wl(50, 60, 100, 50), 11218.8, 1)
        self.assertAlmostEqual(self.g141_grism.get_pixel_wl(60, 50, 100, 50), 10770.6, 1)

    # There is no code to stop this, but perhaps in future there should be
    # Going beyond the detector in ref or normal still gives a value as the calculations are polynomial based
    # def test_get_pixel_wl_beyond_limits(self):
    # self.assertAlmostEqual(self.g141_grism.get_pixel_wl(2000, 2000, 100, 50), 11218.8, 1)
    # self.assertAlmostEqual(self.g141_grism.get_pixel_wl(60, 50, 2000, 2000), 10770.4, 1)

    def test_get_pixel_wl_per_row(self):
        # TODO (ryan) should this be 1024 or 1014 in length?
        wl = self.g141_grism.get_pixel_wl_per_row(50, 50, np.arange(1024))

        self.assertEqual(len(wl), 1024)
        self.assertAlmostEqual(wl.mean(), 29961.2, 1)
        self.assertAlmostEqual(wl.min(), 8959., 1)
        self.assertAlmostEqual(wl.max(), 53001.1, 1)

    def test_get_pixel_wl_per_row_x_values(self):
        wl = self.g141_grism.get_pixel_wl_per_row(50, 50, np.array([100, 110, 120, 150, 200]))

        np.testing.assert_array_almost_equal(wl, [11222.2, 11674.8, 12127.5, 13485.4, 15748.6], 1)

    def test_get_pixel_wl_per_row_y_value(self):
        wl = self.g141_grism.get_pixel_wl_per_row(50, 50, np.array([100, 110, 120, 150, 200]), 51)

        np.testing.assert_array_almost_equal(wl, [11222.7, 11675.3, 12127.9, 13485.9, 15749.1], 1)

    def test_get_pixel_edges_wl_per_row(self):
        wl = self.g141_grism.get_pixel_edges_wl_per_row(50, 50, np.array([100, 110, 120, 130]), None, 10)

        np.testing.assert_array_almost_equal(wl, [10995.9, 11448.5, 11901.2, 12353.8, 12806.5], 1)

    def test_bin_centers_to_limits(self):
        centers = np.array([-1, 0, 1])
        limits = self.g141_grism._bin_centers_to_limits(centers, 1)

        numpy.testing.assert_array_equal(limits, np.arange(-1.5, 2.))


class Test_SpectrumTrace(unittest.TestCase):
    def test__init__(self):
        grism._SpectrumTrace(50, 50, np.zeros(9), np.zeros(9))  # pass if no exceptions

class Test_G141_Trace(unittest.TestCase):

    def test__get_wavelength_calibration_coeffs_50_50(self):
        # This test is failing yet the result given as not equal isequal :-S
        g141_trace = grism.G141_Trace(50, 50)

        # This step is normally done at initialisation
        trace_50_50 = np.array(g141_trace._get_wavelength_calibration_coeffs(50, 50))
        assertArrayAlmostEqual(trace_50_50,
                               np.array([9.94400817e-03, 1.87673580e+00, 4.52664778e+01, 8.95898963e+03]), 4)

        trace_100_50 = np.array(g141_trace._get_wavelength_calibration_coeffs(100, 50))
        assertArrayAlmostEqual(trace_100_50,
                               np.array([9.5914e-03, 1.8813e+00, 4.5278e+01, 8.9637e+03]), 4)

        trace_50_100 = np.array(g141_trace._get_wavelength_calibration_coeffs(50, 100))
        assertArrayAlmostEqual(trace_50_100,
                               np.array([9.85778097e-03, 1.78010580e+00, 4.53781960e+01, 8.95898963e+03]), 4)