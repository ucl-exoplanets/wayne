import unittest

import numpy as np
import numpy.testing
from exodata.example import genExamplePlanet
from astropy.analytic_functions import blackbody_lambda
import astropy.units as u

from .. import observation
from .. import exposure
from .. import detector
from .. import grism


class TestExposureGenerator(unittest.TestCase):
    def setUp(self):
        self.planet = genExamplePlanet()
        self.det = detector.WFC3_IR()
        self.g141 = grism.Grism()

        self.wl = np.linspace(0.7, 1.8, 100) * u.micron
        self.star_flux = blackbody_lambda(self.wl, 3000) * u.sr
        self.planet_depth = np.ones(len(self.wl))

        self.expgen = observation.ExposureGenerator(self.det, self.g141, 2, 'RAPID', 256, self.planet)

    def test_staring_frame(self):  # just testing it generates
        exp = self.expgen.staring_frame(500, 500, self.wl, self.star_flux, self.planet_depth, 3)
        self.assertIsInstance(exp, exposure.Exposure)

    def test_scanning_frame(self):  # just testing it generates
        exp = self.expgen.scanning_frame(500, 500, self.wl, self.star_flux,
                                         self.planet_depth, 1 * u.pixel / u.s, 0.2 * u.s, 3)
        self.assertIsInstance(exp, exposure.Exposure)

    def test__gen_scanning_sample_times_works(self):
        expgen = observation.ExposureGenerator(self.det, self.g141, 3, 'SPARS10', 1024, self.planet)
        sample_starts, sample_mid_points, sample_durations = expgen._gen_scanning_sample_times(1 * u.s)

        # read times 2.932, 12.933, 22.934
        # this code has been manually reviewed to produce the desired result
        starts = np.array([0., 1., 2., 2.932, 3.932, 4.932, 5.932, 6.932, 7.932, 8.932,
                  9.932, 10.932, 11.932, 12.932, 12.933, 13.933, 14.933, 15.933, 16.933, 17.933, 18.933,
                  19.933, 20.933, 21.933, 22.933])

        _ends = np.roll(starts, -1)
        durations = _ends - starts
        durations[-1] = 22.934 - starts[-1]

        mid_points = starts + (durations/2)

        numpy.testing.assert_array_almost_equal(starts, sample_starts.to(u.s).value, 3)
        numpy.testing.assert_array_almost_equal(durations, sample_durations.to(u.s).value, 3)
        numpy.testing.assert_array_almost_equal(mid_points, sample_mid_points.to(u.s).value, 3)