""" Simulation for a grism, including templates for WFC3's on board G102 and G141 grisms. These classes take a raw
spectrum (or in future through the instrument) and simulates its passing through the grism as a field. The detector class
then maps the field to pixels.
"""

import os.path

import numpy as np
import quantities as pq
from astropy.modeling import models

from detector import WFC3_IR
import params

# Strictly the grism class should be non specfic and the G141 var / function should create one with g141 params.


class Grism(object):

    def __init__(self, flux, wavelength):
        """ Takes input of a spectrum of flux and wavelength, it can then rebin

        :param flux: list of fluxes of the raw spectrum
        :type flux: list / numpy.ndarray
        :param wavelength: list of wavelengths of the raw spectrum
        :type wavelength: list / numpy.ndarray
        """
        self.raw_flux = flux
        self.raw_wavelength = wavelength

        self.detector = WFC3_IR()  # this should be called and set by an observation / instrument class

        # Grism Values
        self.min_lambda = 1075 * pq.nm
        self.max_lambda = 1700 * pq.nm

        self.dispersion = 4.65 * pq.nm

        self._psf_file = np.loadtxt(os.path.join(params._data_dir, 'wfc3-g141-fwhm.dat'))
        # in future add option to set unit of data file but we work internally in microns
        self.psf_wavelength = (self._psf_file[:, 0] * pq.nm).rescale(pq.micron)
        self.psf_fwhm = self._psf_file[:, 0] * self.detector.pixel_unit

        # Some constants
        self._FWHM_to_StDev = 1./(2*np.sqrt(2*np.log(2)))

    def rebin(self):
        """ Rebins the raw spectrum to the resolution of the grism.

        .. Note:
            * due to the way the grism is mapped on the detector, a closer relationship between grism and detector
            may be needed and this function changed when angled.

        :return: rebinned spectrum (pixel units)
        """

        # TODO rebin using pysynphot

    def throughput(self):
        """ should be done before the binning, the convert each flux and $\lambda$ to the correct throughput
        :return:
        """
        pass

    def psf(self, wavelength, flux):
        """ Given a wavelength and flux this function returns the gaussian function for the observation at the wl
        given (interpolated)

        :param wavelength: wavelength to sample in microns
        :param flux: The flux at the wavelength (i.e the area under the gaussian)

        :return:
        """

        amplitude = 0  # To be fitted
        mean = 0  # To be Fitted

        FWHM = self.psf_fwhm
        stdev = FWHM * self._FWHM_to_StDev

        gauss = models.Gaussian1D(amplitude, mean, stdev)

        # TODO fix the stdev and vary mean and amplitude to match flux, read a little more to see if we can
        # find a relation
