import unittest

import numpy as np

from .. import grism


class Test_Grism(unittest.TestCase):

    def setUp(self):

        self.flux = np.linspace(0.9, 1.2, 10)
        self.wavelength = np.linspace(0.9, 1.2, 10)
        self.g141_grism = grism.Grism(self.flux, self.wavelength)

    def test_psf_at_sample_points(self):

        self.g141_grism.psf(1.2)
        self.g141_grism.psf(1.7)

        assert False

    def test_psf_between_sample_points_interpolation(self):

        self.g141_grism.psf(1.15)
        self.g141_grism.psf(1.65)

        assert False