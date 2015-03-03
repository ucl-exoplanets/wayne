""" Simulation for a grism, including templates for WFC3's on board G102 and G141 grisms. These classes take a raw
spectrum (or in future through the instrument) and simulates its passing through the grism as a field. The detector class
then maps the field to pixels.
"""

import os.path

import numpy as np
import quantities as pq

from models import GaussianModel1D
from detector import WFC3_IR
import params

# Strictly the grism class should be non specfic and the G141 var / function should create one with g141 params.


class Grism(object):
    """ Handles a grism object and can be used to peform calculation on it such as the psf given a flux and wavelength

    This class should not include observation data and instead be used to turn observational dat into observed. An
    observation / combined detector grism class can do this. calling each component as its needed. You are not then doing

    x = grism(observation, g141)
    x.psf(wl)

    but x.psf(wl, flux), psf should include all the factors, such as throughput
    """
    def __init__(self):
        """ In future will take the vars that define a unique grism, for now this is kept WFC3 G141 specific, with all
        params defined in __init__. this is mostly because i dont know what unique set of params will be required in the
        end for each grism.

        self.detector = WFC3_IR()  # this should be called and set by an observation / instrument class
        Remeber many parameters like dispersion here are intrinsically linked to the detector, so this is really
        WFC3 G141 not G141 applicable to any instrument.
        """

        # Detector Values
        # ---------------
        # Note that most det values are being pulled directly from the detector class. Given that these two classes are
        # intrinsically linked this is probably ok, but can be changed if needed. (i.e. self.detector.pixel_unit)
        self.detector = WFC3_IR()

        # Grism Values
        # ------------
        self.min_lambda = 1075 * pq.nm
        self.max_lambda = 1700 * pq.nm

        # self.dispersion = 4.65 * pq.nm - The dispersion is actually dependant on x and y and not constant

        ## PSF Information
        self._psf_file = np.loadtxt(os.path.join(params._data_dir, 'wfc3-g141-fwhm.dat'))
        # in future add option to set unit of data file but we work internally in microns
        self.psf_wavelength = (self._psf_file[:, 0] * pq.nm).rescale(pq.micron)
        self.psf_fwhm = self._psf_file[:, 1] * self.detector.pixel_unit

        # Non Grism Specific Constants
        self._FWHM_to_StDev = 1./(2*np.sqrt(2*np.log(2)))

    def throughput(self):
        """ should be done before the binning, the convert each flux and $\lambda$ to the correct throughput

        :return: factor to scale the flux by
        """
        pass

    def flux_to_psf(self, wavelength, flux):
        """
        Given a wavelength and flux this function returns the gaussian function for the observation at the wl given
         (linearly interpolated using numpy.interp)

        The FWHM at each wavelength should be defined in a textfile loaded by self._psf_file

        We assume the psf can be represented by a gaussian, this is true for WFC3 G141 (*WFC3 inst handbook (cycle 23)
         - sec 7.6.1 - page 140*).

        :param wavelength: wavelength to sample (quantites unit length)
        :param flux: The flux at the wavelength (i.e the area under the gaussian)

        :return:
        """

        if not type(wavelength) is pq.quantity.Quantity:
            raise TypeError("Wavelength must be given as a quantities unit i.e. 1 * pq.micron got {} type {}".format(
                wavelength, type(wavelength)
            ))

        mean = 0.  # this is the center, make sense that we zero it for now

        # linear interpolation of the FWHM given in self._psf_file TODO quadratic interp / fit a function?
        FWHM = np.interp(wavelength.rescale(pq.micron), self.psf_wavelength, self.psf_fwhm, left=0., right=0.)

        gaussian_model = GaussianModel1D(mean=mean, fwhm=FWHM, flux=flux)

        return gaussian_model
