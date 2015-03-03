import unittest

import numpy as np
import quantities as pq

from .. import grism


class Test_Grism(unittest.TestCase):

    def setUp(self):

        self.flux = np.linspace(0.9, 1.2, 10)
        self.wavelength = np.linspace(0.9, 1.2, 10)
        self.g141_grism = grism.Grism()

    def test__init__(self):
        g141_grism = grism.Grism()  # pass if no exceptions

    def test_flux_to_psf_raises_TypeError_given_no_units(self):
        with self.assertRaises(TypeError):
            self.g141_grism.flux_to_psf(1.2, 1.)

    def test_flux_to_psf_at_sample_points(self):
        # Verified using MCMC
        psf1 = self.g141_grism.flux_to_psf(1.2 * pq.micron, 1.)
        self.assertAlmostEqual(psf1.flux, 1., 9)
        self.assertAlmostEqual(psf1.amplitude, 0.9033, 4)
        self.assertAlmostEqual(psf1.stddev, 0.4416, 4)

        psf2 = self.g141_grism.flux_to_psf(1.7 * pq.micron, 1.5)

        self.assertAlmostEqual(psf2.flux, 1.5, 9)
        self.assertAlmostEqual(psf2.amplitude, 1.1560, 4)
        self.assertAlmostEqual(psf2.stddev, 0.5177, 4)

    def test_flux_to_psf_between_sample_points_interpolation(self):

        psf1 = self.g141_grism.flux_to_psf(1.15 * pq.micron, 1.)
        self.assertAlmostEqual(psf1.flux, 1., 9)
        self.assertAlmostEqual(psf1.amplitude, 0.9125, 4)
        self.assertAlmostEqual(psf1.stddev, 0.4372, 4)

        psf2 = self.g141_grism.flux_to_psf(1.65 * pq.micron, 1.5)
        self.assertAlmostEqual(psf2.flux, 1.5, 9)
        self.assertAlmostEqual(psf2.amplitude, 1.1767, 4)
        self.assertAlmostEqual(psf2.stddev, 0.5085, 4)