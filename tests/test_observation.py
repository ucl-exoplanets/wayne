import unittest

import numpy as np
from exodata.example import genExamplePlanet
from astropy.analytic_functions import blackbody_lambda
import astropy.units as u

from .. import observation
from .. import exposure
from .. import detector
from .. import grism


class TestExposureGenerator(unittest.TestCase):

    def setUp(self):
        planet = genExamplePlanet()
        det = detector.WFC3_IR()
        g141 = grism.Grism()

        self.wl = np.linspace(0.7, 1.8, 100) * u.micron
        self.star_flux = blackbody_lambda(self.wl, 3000) * u.sr
        self.planet_depth = np.ones(len(self.wl))


        self.expgen = observation.ExposureGenerator(det, g141, 2, 'RAPID', 256, planet)

    def test_staring_frame(self):  # just testing it generates
        exp = self.expgen.staring_frame(500, 500, self.wl, self.star_flux, self.planet_depth, 3)
        self.assertIsInstance(exp, exposure.Exposure)

    def test_scanning_frame(self):  # just testing it generates
        exp = self.expgen.scanning_frame(500, 500, self.wl, self.star_flux,
                                         self.planet_depth, 1*u.pixel/u.s, 0.2*u.s, 3)
        self.assertIsInstance(exp, exposure.Exposure)