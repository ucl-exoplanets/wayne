import unittest

import numpy as np
from .. import models


class Test_GaussianModel1D(unittest.TestCase):

    # Start by testing initialisation conditions and that my changes to the class do not break the astropy model
    def test_init_with_stddev_mean_amplitude(self):
        gauss = models.GaussianModel1D(amplitude=1., mean=1.5, stddev=0.5)

        # check variables are correct
        self.assertEqual(gauss.amplitude, 1.)
        self.assertEqual(gauss.mean, 1.5)
        self.assertEqual(gauss.stddev, 0.5)

        # check the model works on a couple of points (evaluating astropy code here)
        self.assertEqual(gauss(1.5), 1.)
        self.assertAlmostEqual(gauss(1.75), 0.8825, 4)

    def test_init_with_stddev_mean_flux(self):
        gauss = models.GaussianModel1D(flux=1., mean=1.5, stddev=0.5)

        # check variables are correct
        self.assertEqual(gauss.flux, 1.)
        self.assertAlmostEqual(gauss.amplitude, 0.7979, 4)
        self.assertEqual(gauss.mean, 1.5)
        self.assertEqual(gauss.stddev, 0.5)

        # check the model works on a couple of points (evaluating astropy code here)
        self.assertAlmostEqual(gauss(1.5), 0.7979, 4)
        self.assertAlmostEqual(gauss(1.75), 0.7041, 4)

    def test_init_with_fwhm_mean_amplitude(self):
        gauss = models.GaussianModel1D(amplitude=1., mean=1.5, fwhm=0.5)

        # check variables are correct
        self.assertEqual(gauss.amplitude, 1.)
        self.assertEqual(gauss.mean, 1.5)
        self.assertEqual(gauss.fwhm, 0.5)
        self.assertAlmostEqual(gauss.stddev, 0.2123, 4)

        # check the model works on a couple of points (evaluating astropy code here)
        self.assertEqual(gauss(1.5), 1.)
        self.assertEqual(gauss(1.75), 0.5)

    def test_init_with_fwhm_mean_flux(self):
        gauss = models.GaussianModel1D(flux=1., mean=1.5, fwhm=0.5)

        # check variables are correct
        self.assertEqual(gauss.flux, 1.)
        self.assertAlmostEqual(gauss.amplitude, 1.879, 3)
        self.assertEqual(gauss.mean, 1.5)
        self.assertEqual(gauss.fwhm, 0.5)
        self.assertAlmostEqual(gauss.stddev, 0.2123, 4)

        # check the model works on a couple of points (evaluating astropy code here)
        self.assertAlmostEqual(gauss(1.5), 1.879, 3)
        self.assertAlmostEqual(gauss(1.75), 0.9394, 4)

    def test_init_raises_Valueerror_with_stdev_fwhm_mean_amplitude(self):
        with self.assertRaises(ValueError):
            models.GaussianModel1D(amplitude=1., mean=1.5, fwhm=0.5, stddev=0.5)

    def test_init_fails_with_stdev_mean_amplitude_flux(self):
        with self.assertRaises(ValueError):
            models.GaussianModel1D(amplitude=1., mean=1.5, flux=1., stddev=0.5)

    # These are currently done partly in __init__ tests. But they should probably be done explicitly here
    # def test_amplitude_to_flux_calculation(self):
    #     assert False
    #
    # def test_flux_to_amplitude_calculation(self):
    #     assert False
    #
    # def test_fwhm_to_stddev_calculation(self):
    #     assert False
    #
    # def test_stdev_to_fwhm_calculation(self):
    #     assert False

    @unittest.skip("Fails as this is not coded behavior, but perhaps it should be")
    def test_changing_stddev_recalculates_amp_from_flux(self):
        """ The reason for this test is if you initialise a gaussian with flux, then change the stddev after you may
        expect flux to be preserved. It is not (as amplitude is calculated from flux when set, not on the fly). Im
        unsure if i should change this, raise a warning or leave it. For now im writing this test that will fail. And
        raising an issue on github (issue #1)
        """

        gauss = models.GaussianModel1D(mean=1.5, flux=1., stddev=0.5)

        gauss.stddev = 0.6

        self.assertEqual(gauss.flux, 1.)

    # Integration values verified by MCMC
    def test_integrate_dist1(self):
        gauss = models.GaussianModel1D(mean=0, flux=1, stddev=1)

        self.assertEqual(gauss.integrate(0, 0.5), gauss.integrate(-0.5, 0))

        self.assertAlmostEqual(gauss.integrate(0, 0.5), 0.1915, 4)
        self.assertAlmostEqual(gauss.integrate(-0.5, 0.5), 0.3829, 4)

        self.assertAlmostEqual(gauss.integrate(0.3, 0.7), 0.1401, 4)
        self.assertAlmostEqual(gauss.integrate(-0.1, 0.4), 0.1952, 4)

        self.assertAlmostEqual(gauss.integrate(-4, 4), 1., 3)

    def test_integrate_dist2(self):
        gauss = models.GaussianModel1D(mean=5, flux=3, stddev=0.5)

        self.assertEqual(gauss.integrate(5, 6), gauss.integrate(4, 5))

        self.assertAlmostEqual(gauss.integrate(4, 5), 1.43174, 4)
        self.assertAlmostEqual(gauss.integrate(4, 6), 1.43174*2, 4)

        self.assertAlmostEqual(gauss.integrate(4.3, 4.7), 0.5805, 4)

        self.assertAlmostEqual(gauss.integrate(3, 7), 3., 3)