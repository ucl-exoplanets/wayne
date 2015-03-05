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

    def _get_wavelength_calibration_coeffs(self, x_ref, y_ref):
        """ Returns the coefficients to compute the spectrum trace and the wavelength calibration as defined in the Axe
        Software manual and determined by (Knutschner 2009, calibration of the G141 Grism).

        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame

        :return: a_t, b_t, a_w, b_w
        """

        # Spectrum Trace
        a_t = 1.04275e-2 + (-7.96978e-6)*x_ref + (-2.49607e-6)*y_ref + (1.45963e-9)*x_ref**2 + (1.39757e-8)*x_ref*y_ref \
                                                                                            + (4.84940e-10)*y_ref**2
        b_t = 1.96882 + (9.09159e-5)*x_ref + (-1.93260e-3)*y_ref

        # Wavelength Solution
        a_w = 4.51423e1 + (3.17239e-4)*x_ref + (2.17055e-3)*y_ref + (-7.42504e-7)*x_ref**2 + (3.48639e-7)*x_ref*y_ref \
                                                                                            + (3.09213e-7)*y_ref**2
        b_w = 8.95431e3 + (9.35925e-2)*x_ref # + (0)*y_ref

        return a_t, b_t, a_w, b_w

    def get_pixel_wl(self, x_ref, y_ref, x_1, y_1):
        """ gives the wavelength of pixel x1, y1 given reference pixel x_ref, y_ref.

        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame
        :param x_1:
        :param y_1:

        :notes: takes around 5.6 seconds to compute each pixel separately on a 1024x1024 grid

        :return: the wavelength of pixel (x_1, y_1)
        """

        a_t, b_t, a_w, b_w = self._get_wavelength_calibration_coeffs(x_ref, y_ref)
        a_t_i = 1/a_t  # the inverse

        # Distance between the reference and required point on the trace
        d = np.sqrt((y_ref-y_1+a_t_i*x_ref-a_t_i*x_1)**2/(a_t_i**2+1))

        # Determination of wavelength
        wl = a_w * d + b_w

        return wl

    def get_pixel_wl_per_row(self, x_ref, y_ref, x_values=None, y_value=None):
        """ Given the star position (x_ref, y_ref) this function will return the wavelengths for the pixels in the row
        y_ref. If x_values is given, this function will return the values for x_values rather than the whole row

        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame
        :param x_values: list of pixel numbers to obtain wavelengths for, can be fractional
        :param y_value: y value of the row, if None will use y_ref

        :notes: takes around 0.33 seconds to compute each row, on 1024 grid. ~170x faster than each pixel individually

        :return: wavelengths for the row at y_ref for all pixels or all values of x_values
        """

        if x_values is None:
            x_values = np.arange(1024)  # TODO set pixel numbers in detector class
        else:
            x_values = np.array(x_values)

        if y_value is None:
            y_value = y_ref

        a_t, b_t, a_w, b_w = self._get_wavelength_calibration_coeffs(x_ref, y_ref)
        a_t_i = 1/a_t  # the inverse

        d_values = np.sqrt((y_ref-y_value+a_t_i*x_ref-a_t_i*x_values)**2/(a_t_i**2+1))

        wl_values = a_w * d_values + b_w

        return wl_values

    def get_pixel_edges_wl_per_row(self, x_ref, y_ref, x_centers=None, y_value=None, pixel_size=1.):
        """ Calculates the wavelength inbetween each pixel defined in x_centers. For example x_centers = 1,2,3 and the
        return will give the wavelengths for 0.5, 1.5, 2.5, 3.5. This is the same as

            self.get_pixel_row_wl(x_ref, y_ref, x_values=self._bin_centers_to_limits(x_centers, pixel_size), y_value)

        And is used to get the limits to pass to a binning function to reduce the input spectra to pixels

        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame
        :param x_centers: list of pixel numbers to be turned into edges
        :param y_value: y value of the row, if None will use y_ref
        :param pixel_size: size of pixel (same units as centers)

        :notes: takes around 0.33 seconds to compute each row, on 1024 grid. ~170x faster than each pixel individually

        :return: wavelengths for the row at y_ref for all pixels or all values of x_values
        """

        x_values = self._bin_centers_to_limits(x_centers, pixel_size)
        wl_values = self.get_pixel_wl_per_row(x_ref, y_ref, x_values, y_value)

        return wl_values

    def _bin_centers_to_limits(self, centers, bin_size=1.):
        """ Converts a list of bin centers into limits (edges) given a (constant) bin size. Can be fractional.

        This is used to convert raw pixel numbers into limits
            i.e 1, 2, 3 -> 0.5, 1.5, 2.5, 3.5
        in order to then determine the spectral contribution in that pixel

        No effort is made to see if you gave a constant bin, you will just get an incorrect result.
            i.e. 1, 5, 7 -> 0.5, 4.5, 6.5, 7.5

        :param centers: array of bin centers
        :param bin_size: size of bin (same units as centers)

        :return: array of bin edges of length len(centers)+1
        """

        centers = np.array(centers)
        half_bin = bin_size/2.

        bin_edges = np.append(centers-half_bin, centers[-1]+half_bin)

        return bin_edges