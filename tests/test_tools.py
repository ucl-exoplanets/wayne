import unittest

import numpy as np
import numpy.testing
import astropy.units as u

from wayne import tools


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


class Test_bin_centers_to_edges(unittest.TestCase):

    def test_works_uniform_bins(self):
        centers = np.array([1, 2, 3, 4])
        bin_edges = tools.bin_centers_to_edges(centers)

        answer = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

        numpy.testing.assert_array_equal(answer, bin_edges)

    def test_works_non_uniform_bins(self):
        centers = np.array([1, 2, 4, 5.4])
        bin_edges = tools.bin_centers_to_edges(centers)

        answer = np.array([0.5, 1.5, 3, 4.7, 6.1])

        numpy.testing.assert_array_almost_equal(answer, bin_edges, 6)


class Test_bin_centers_to_width(unittest.TestCase):

    def test_works_uniform_bins(self):
        centers = np.array([1, 2, 3, 4])
        bin_width = tools.bin_centers_to_widths(centers)

        answer = np.array([1, 1, 1, 1])

        numpy.testing.assert_array_equal(answer, bin_width)

    def test_works_non_uniform_bins(self):
        centers = np.array([1, 2, 4, 5.4])
        bin_width = tools.bin_centers_to_widths(centers)

        answer = np.array([1, 1.5, 1.7, 1.4])

        numpy.testing.assert_array_almost_equal(answer, bin_width, 6)


class Test_detect_orbits(unittest.TestCase):
    def test_detect_orbits(self):
        exp_start_times = [1.001, 1.002, 1.032] * u.day

        orbit_intervals = tools.detect_orbits(exp_start_times)

        self.assertEqual(orbit_intervals, [0, 2])