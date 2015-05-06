""" Simulation for a grism, including templates for WFC3's on board G102 and G141 grisms. These classes take a raw
spectrum (or in future through the instrument) and simulates its passing through the grism as a field. The detector class
then maps the field to pixels.
"""

import os.path

import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits

from models import GaussianModel1D, _gauss_StDev_to_FWHM
from detector import WFC3_IR
import params

# Strictly the grism class should be non specfic and the G141 var / function should create one with g141 params.


class Grism(object):
    """ Handles a grism object and can be used to perform calculation on it such as the psf given a flux and wavelength

    This class should not include observation data and instead be used to turn observational data into observed. An
    observation / combined detector grism class can do this. calling each component as its needed.
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
        self.name = 'G141'

        # Grism Values
        # ------------
        self.min_lambda = 1.075 * u.micron
        self.max_lambda = 1.700 * u.micron

        # self.dispersion = 4.65 * pq.nm - The dispersion is actually dependant on x and y and not constant

        ## PSF Information
        self._psf_file = np.loadtxt(os.path.join(params._data_dir, 'wfc3-g141-fwhm.dat'))
        # in future add option to set unit of data file but we work internally in microns
        self.psf_wavelength = (self._psf_file[:, 0] * u.nm).to(u.micron)
        self.psf_fwhm = self._psf_file[:, 1] * self.detector.pixel_unit

        # we crop the input spectrum using this, the limiting factor here is psf
        self.wl_limits = (self.psf_wavelength[0], self.psf_wavelength[-1])

        # Throughput
        self.throughput_file = os.path.join(params._data_dir, 'wfc3_ir_g141_src_004_syn.fits')
        with fits.open(self.throughput_file) as f:
            tbl = f[1].data  # the table is in the data of the second HDU
            self.throughput_wl = (tbl.field('WAVELENGTH') * u.angstrom).to(u.micron)
            self.throughput_val = tbl.field('THROUGHPUT')

        # Non Grism Specific Constants
        self._FWHM_to_StDev = 1./(2*np.sqrt(2*np.log(2)))

    def flux_to_psf(self, wavelength, flux, y_pos=0.):
        """
        Given a wavelength and flux this function returns the gaussian function for the observation at the wl given
         (linearly interpolated using numpy.interp)

        The FWHM at each wavelength should be defined in a textfile loaded by self._psf_file

        We assume the psf can be represented by a gaussian, this is true for WFC3 G141 (*WFC3 inst handbook (cycle 23)
         - sec 7.6.1 - page 140*).

        :param wavelength: wavelength to sample (unit length) singular
        :type wavelength: astropy.units.quantity.Quantity
        :param flux: The flux at the wavelength (i.e the area under the gaussian)
        :type flux: float
        :param y_pos: optionally set ypos. this means you can integrate over the same values as pixels set mean=2.1 and
          integrate from 2 to 3
        :type y_pos: float

        :return:
        """

        if not isinstance(wavelength, u.Quantity):
            raise TypeError("Wavelength must be given in units, got {}".format(type(wavelength)))

        mean = y_pos  # this is the center of the guassian

        # linear interpolation of the FWHM given in self._psf_file TODO quadratic interp / fit a function?
        FWHM = np.interp(wavelength.to(u.micron).value, self.psf_wavelength.to(u.micron).value,
                         self.psf_fwhm, left=0., right=0.)

        gaussian_model = GaussianModel1D(mean=mean, fwhm=FWHM, flux=flux)

        return gaussian_model

    def wl_to_psf_std(self, wavelengths):
        """ This is an optimised function to return the standard deviations of the gaussians at each wavelength for
        each flux. looks up FWHM then converts to stddev.

        :param wavelengths:

        :return: standard deviations of the gaussian for each lambda
        """

        FWHM = np.interp(wavelengths.to(u.micron).value, self.psf_wavelength.to(u.micron).value,
                         self.psf_fwhm, left=0., right=0.)

        std = FWHM / _gauss_StDev_to_FWHM

        return std


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
        b_w = 8.95431e3 + (9.35925e-2)*x_ref  # + (0)*y_ref

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

    def throughput(self, wl, flux):
        """ Converts measurements of wavelength, flux to post grism values by multiplying by the throughput. A linear
        interpolation (numpy.interp) is used to generate values between sample points.

        :param wl:
        :type wl: numpy.ndarray
        :param flux:
        :type flux: numpy.ndarray

        :return: flux, throughput corrected
        :rtype: numpy.ndarray
        """

        throughput_vales = np.interp(wl, self.throughput_wl, self.throughput_val, 0., 0.)

        return flux*throughput_vales

    def plot_throughput(self):
        """ Plots the throughput curve for the grism
        """

        plt.figure()
        plt.plot(self.throughput_wl, self.throughput_val)
        plt.title("Grism Throughput")
        plt.ylabel("Throughput")
        plt.xlabel("$\lambda (\mu m)$")

    def plot_spectrum_with_throughput(self, wl, flux):
        """ Plots the spectrum before and after the throughput corrections

        :param wl:
        :type wl: numpy.ndarray
        :param flux:
        :type flux: numpy.ndarray
        """

        plt.figure()
        plt.plot(wl, flux, label="Input Spectra")
        plt.plot(wl, self.throughput(wl, flux), label="Throughput Corrected")

        plt.title("Input Spectrum through the grism")
        plt.ylabel("Flux")  # really its normally transit depth
        plt.xlabel("$\lambda (\mu m)$")

        plt.legend(loc="lower center")

    def get_trace(self, x_ref, y_ref):
        """ Get the spectrum trace, in order to convert between x, y and wl.

        This function should setup the trace class wth the x_ref and y_ref values given along with the calibration
        coefficients

        :return: SpectrumTrace class
        """

        return SpectrumTrace(x_ref, y_ref)


class SpectrumTrace(object):
    """ Calculates the equation of the spectrum trace given a source position. These are defined in the hubble documents
    and turn pixel number into wavelength (and vise versa).

    This is in effect a special type of polynomial class which converts between the x and y pixel positions
    (called x and y) herein and the wavelength of these positions. This class only operates on the trace, and for
    $x>x_ref, y > y_ref$ and as such is not suitable for multi-object simulations
    """

    def __init__(self, x_ref, y_ref):
        """
        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame
        """

        self.x_ref = x_ref
        self.y_ref = y_ref

        self.m_t, self.c_t, self.m_w, self.c_w = self._get_wavelength_calibration_coeffs(x_ref, y_ref)
        self.m_wl, self.c_wl = self._get_x_to_wl_poly_coeffs(x_ref, y_ref)

    def _get_wavelength_calibration_coeffs(self, x_ref, y_ref):
        """ Returns the coefficients to compute the spectrum trace and the wavelength calibration as defined in the Axe
        Software manual and determined by (Knutschner 2009, calibration of the G141 Grism).

        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame

        :return: a_t, b_t, a_w, b_w
        """

        # TODO, make these coefficiants part of the input to set this class (so we can translate to the other grisms)
        # Spectrum Trace
        m_t = np.array(1.04275e-2 + (-7.96978e-6)*x_ref + (-2.49607e-6)*y_ref + (1.45963e-9)*x_ref**2 + (1.39757e-8)*x_ref*y_ref \
                                                                                            + (4.84940e-10)*y_ref**2)
        c_t = np.array(1.96882 + (9.09159e-5)*x_ref + (-1.93260e-3)*y_ref)

        # Wavelength Solution
        m_w = np.array(4.51423e1 + (3.17239e-4)*x_ref + (2.17055e-3)*y_ref + (-7.42504e-7)*x_ref**2 + (3.48639e-7)*x_ref*y_ref \
                                                                                            + (3.09213e-7)*y_ref**2)
        c_w = np.array(8.95431e3 + (9.35925e-2)*x_ref)  # + (0)*y_ref

        return m_t, c_t, m_w, c_w

    def x_to_y(self, x):
        """ array of x values to convert to y (from the trace)

        :param x: numpy array of x values
        :type x: numpy.ndarray

        $y=m_t*(x-x_ref)+c_t+y_ref$

        :return: y values
        :rtype: numpy.ndarray
        """

        return self.m_t * (x-self.x_ref) + self.c_t + self.y_ref

    def y_to_x(self, y):
        """ array of y values to convert to x (from the trace)

        :param y: numpy array of y values
        :type y: numpy.ndarray

        $x = \frac{y-y_ref-c_t}{m_t}+x_ref$

        :return: x values
        :rtype: numpy.ndarray
        """

        return ((y-self.y_ref-self.c_t)/self.m_t)+self.x_ref

    def _get_x_to_wl_poly_coeffs(self, x_ref, y_ref):
        """ This function finds the polynomial that maps x position to the wavelength. Note this only valid for
        $x > x_ref and y > y_ref$ although for performance we don't check that here as we are only interested in the
        produced spectrum.

        This function works by plotting two points at x_ref+10 and x_ref+20, calculating y using `self.x_to_y` and then
        converting the result to $\lambda$. (10 and 20 could be any combinations of positive numbers). d is the distance
        between $(x_*, y_*)$ and (x, y) on the trace.

        $d = \sqrt{(y-y_*)^2+(x-x_*)^2}$

        we can then get the wl (a factor of 10,000 is used to convert the result to microns)

        $\lambda = (a_w d + b_w) / 10000$

        And then finding the equation of the line given ($\lambda_{10}, $\\x_{10}) and ($\lambda_{20}, $\\x_{20})$$

        $y_{\lambda} = m_{\lambda}x+c_{\lambda}$

        $m_{\lambda} = \frac{y_{20} - y{10}}{x_{20} - x{10}}$

        $c{\lambda} = y_{10} - my_\lambda x_{10}$

        The actual solution is a v shape, increasing, not decreasing for $x < x_ref$ and as such this solution is only
        valid for $x>x_ref$ but the class makes no attempt to check inputs for this

        :param x_ref: The x position of the star on the reference frame
        :param y_ref: The y position of the star on the reference frame

        :return: m_wl and c_wl, the coefficients of the line mapping x to wl
        :rtype: numpy.ndarray
        """

        x = np.array([x_ref+10, x_ref+20])
        y = self.x_to_y(x)

        d = np.sqrt((y - y_ref)**2 + (x-x_ref)**2)

        wl = (self.m_w * d + self.c_w)*u.angstrom
        wl = wl.to(u.micron)

        m_wl = (wl[1] - wl[0]) / (x[1] - x[0])
        c_wl = wl[0] - m_wl*x[0]

        return m_wl, c_wl

    def x_to_wl(self, x):
        """ Converts x to wavelength using coeffs from `self._get_x_to_wl_poly_coeffs`, only valid for $x > x_ref$

        $\lambda = m_{\lambda}x + c_{\lambda}$

        :param x: numpy array of x values
        :type x: numpy.ndarray

        :return: wl values
        :rtype: numpy.ndarray
        """

        return self.m_wl*x + self.c_wl

    def y_to_wl(self, y):
        """ Converts y to wavelength using `self.y_to_x` then `self.x_to_wl`. only valid for $y > y_ref$

        $\lambda = m_{\lambda}x + c_{\lambda}$

        :param y: numpy array of y values
        :type y: numpy.ndarray

        :return: wl values
        :rtype: numpy.ndarray
        """

        x = self.y_to_x(y)
        return self.x_to_wl(x)

    def wl_to_x(self, wl):
        """ Converts wavelength to x using coeffs from `self._get_x_to_wl_poly_coeffs`, this is more dangerous as it is
         still only valid for $x > x_ref$ which equates to $\lambda > 1. \mu m$ but should only be considered valid in
         the grisms wavelength range, as outside this it has no real meaning.

        $x = \frac{\lambda - c_{\lambda}}{m_{\lambda}}$

        :param wl: numpy array of wl values
        :type wl: numpy.ndarray

        :return: x values
        :rtype: numpy.ndarray
        """

        return (wl-self.c_wl)/self.m_wl

    def wl_to_y(self, wl):
        """ Converts wavelength to y using coeffs from `self._get_x_to_wl_poly_coeffs` and then `self.x_to_y`. This is
        more dangerous as it is still only valid for $y > y_ref$ which equates to $\lambda > 1. \mu m$ but should only
        be considered valid in the grisms wavelength range, as outside this it has no real meaning.

        :param wl: numpy array of wl values
        :type wl: numpy.ndarray

        :return: y values
        :rtype: numpy.ndarray
        """

        x = (wl-self.c_wl)/self.m_wl

        return self.x_to_y(x)

    def psf_line(self, wl):
        """ Gives the intercept points and the coefficients of the psf lines at the wavelengths wl

        :param wl: numpy array of wl values
        :type wl: numpy.ndarray

        :return:
        """

        x = self.wl_to_x(wl)
        y = self.wl_to_y(wl)

        m = -np.array(1.)/self.m_t  # so m is an array
        c = y - m*x

        return x, y, m, c

    def plot_trace(self):
        """ plots the trace line on the detector grid with points for the source position and the start and end of the
         spectrum.

        :param fig: figure object
        :return:
        """

        fig = plt.figure(figsize=(6, 6))
        plt.title("WFC3 IR G141 Calibration")
        plt.xlim(0, 1024)
        plt.ylim(0, 1024)
        plt.xlabel("x pixel number")
        plt.ylabel("y pixel number")

        x = np.arange(1024)
        y = self.x_to_y(x)
        plt.plot(x, y, c='b', label="trace line", lw=2)

        plt.scatter(self.x_ref, self.y_ref, c='r', s=60, label="Source Position", zorder=10)

        spec_wl = np.array([1.075, 1.7]) * u.micron
        spec_x = self.wl_to_x(spec_wl)
        spec_y = self.wl_to_y(spec_wl)
        plt.scatter(spec_x, spec_y, c='g', s=60, label="Spectrum Position", zorder=10)
        plt.legend()

        return fig

    def xangle(self):
        """ get the angle from the x axis to the trace line using triganometry.

        :return: angle in radians
        """

        x = np.array([1., 2.])
        y = self.x_to_y(x)
        a1 = x[1] - x[0]
        o1 = y[1] - y[0]

        theta = np.arctan(o1/a1)

        return theta

    def psf_length_per_pixel(self):
        """ Calculates the length of the line perpendicular to the trace that will fit inside 1 pixel. This is used
        to calculate how much flux should be binned per pixel.

        To do this we draw triangle in with the psf line being h, adjacent on the y axis (and equal 1) and opposite on
        the x axis. The angle is the same at xangle

        :return:
        """

        theta = self.xangle()

        h = 1/np.cos(theta)

        return h