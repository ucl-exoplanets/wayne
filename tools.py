""" Contains functions useful to the program
"""

import numpy as np
import pysynphot
import astropy.io.fits as fits
from scipy.interpolate import interp1d
import pylightcurve.fcmodel as pylc


def crop_spectrum(min_wl, max_wl, wl, flux):
    """ crops the spectrum strictly between the limits provided, only works
     if wl is ordered

    :param min_wl: lower limit to crop
    :type: numpy.ndarray
    :param max_wl: upper limit to crop
    :type: numpy.ndarray
    :param wl: array of spectrum wl
    :type wl: numpy.ndarray
    :param flux: array of spectrum flux
    :type flux: numpy.ndarray

    :return: wl, flux (cropped)
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    wl_min_nearest = wl-min_wl
    # any negative values are < min_wl and are excluded by assigning them the
    #  highest value in the array
    wl_min_nearest[wl_min_nearest < 0] = wl_min_nearest.max()
    imin = wl_min_nearest.argmin()

    wl_max_nearest = wl-max_wl

    # any positive values are > min_wl and are excluded
    wl_max_nearest[wl_max_nearest > 0] = wl_max_nearest.min()

    # plus 1 because we want this value included in the slice
    imax = wl_max_nearest.argmax() + 1

    return wl[imin:imax], flux[imin:imax]


def bin_centers_to_edges(centers):
    """ Converts bin centers to edges, handling uneven bins. Bins are assumed to
    be between the midpoint of the surrounding centers, the edges are assumed to
    extend out by the midpoint to the next/previous point.

    :param centers: array of bin centers
    :type centers: numpy.ndarray

    :return: bin edges
    :rtype: numpy.ndarray
    """

    bin_range = (centers-np.roll(centers, 1))/2.
    # Handle the start point (the roll means [0] will be centers[-1]
    bin_range[0] = bin_range[1]

    # len(edges) = len(centers)+1
    bin_edges = np.zeros(len(centers)+1)
    bin_edges[:-1] = centers - bin_range

    # now handle end point
    bin_edges[-1] = centers[-1] + bin_range[-1]

    return bin_edges


def bin_centers_to_widths(centers):
    """ Converts bin centers to the width per bin

    :param centers: array of bin centers
    :type centers: numpy.ndarray

    :return: bin widths
    :rtype: numpy.ndarray
    """

    bin_range = (centers-np.roll(centers, 1))/2.
    # Handle the start point (the roll means [0] will be centers[-1]
    bin_range[0] = bin_range[1]

    # Now we need to to add each gap to the following gap. so we need to roll
    bin_range_roll = np.roll(bin_range, -1)
    # handle the end this time
    bin_range_roll[-1] = bin_range[-1]

    # len(edges) = len(centers)
    bin_width = bin_range + bin_range_roll

    return bin_width


def rebin_spec(wavelength, spectrum, new_wavelength):
    """ Takes an input spectrum and rebins to new given wavelengths whilst
    preserving flux

    This function was modified from the original by Jessica Lu and John Johnson, taken from
    http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/

    :param wave: original wavelength
    :param specin: original spectrum
    :param wavnew: new wavlengths to bin to
    :return: rebinned spectrum
    """

    spec = pysynphot.spectrum.ArraySourceSpectrum(wave=wavelength, flux=spectrum)
    f = np.ones(len(wavelength))
    filt = pysynphot.spectrum.ArraySpectralElement(wavelength, f, waveunits='angstrom')
    obs = pysynphot.observation.Observation(spec, filt, binset=new_wavelength, force='taper')

    return obs.binflux


def load_pheonix_stellar_grid_fits(fits_file):
    """ loads a phenoix stellar model precomputed wavelength grid in fits
    format.

    :param fits_file:
    :return: wavelength, flux
    """

    with fits.open(fits_file) as f:
        tab = f[1]
        star_wl, star_flux = order_flux_grid(
            tab.data['Wavelength'], tab.data['Flux'])

        # Remove duplicate wl values
        idx = np.nonzero(np.diff(star_wl))
        star_wl = star_wl[idx]
        star_flux = star_flux[idx]

    return star_wl, star_flux


def order_flux_grid(wavelength, spectrum):
    """ Given a wavelength and spectrum, will sort the wavelength in increasing
    order. this is necessary for models like pheonix that have been computed
    on clusters.

    :param wavelength:
    :param spectrum:
    :return:
    """

    sorted_flux = np.array(sorted(zip(wavelength, spectrum)))
    wl = sorted_flux[:, 0]
    flux = sorted_flux[:, 1]

    return wl, flux


def gaussian_smoothing(wavelength, flux):
    """
    Gaussian smoothing for a spectrum written by Angelos. Meant to simulate
     the psf in the spectral direction.
    
    Distributes the flux of each spectral element to the rest of the wavelength
    grid. Uses a gaussian distribution centered at the spectral element with a
    FWHM given by an interpolation function created from the given information
    about the instrument. Because the instrumental psf is given in pixels we
    assume a ratio of 4.5 nm / pixel.  The contribution on each spectral
    element is calculated with numerical integration using the rectangle rule
    between the half-distance wavelengths from the previous and the next
    spectral element.

    Parameters
    ----------
    wavelength : array_like
        array containing the wavelength of each spectral point
    
    flux : array_like
        array containing the flux of each spectral point

    Returns
    -------
    wavelength, smoothed_flux : array_like, array_like 
        arrays containing the wavelength and the smoothed flux for each spectral point

    """
    
    # TODO (ryan) replace with astropy gaussian kernal
    # TODO (ryan) or ideally, simulate a 2d gaussian in generation instead
    
    wavelength = np.array(wavelength)
    flux = np.array(flux)

    def gauss(x, mean, sigma):
        return (1.0/(sigma*np.sqrt(2.0*np.pi))) * np.exp(-(x - mean) ** 2 / (2 * sigma ** 2))

    # TODO (ryan) pull this info from grism
    psfx = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7])
    psfy = np.array([0.971, 0.986, 1.001, 1.019, 1.040, 1.067, 1.100, 1.136, 1.176, 1.219])
    psf = interp1d(psfx, psfy/(2.0*np.sqrt(2.0*np.log(2.0))), kind='cubic')
    
    smoothed_flux = np.zeros_like(flux)[1:-1]
    
    for i in range(len(wavelength)):
        wl = wavelength[i]
        fl = flux[i]
        lim1 = 0.5 * (wavelength[1:-1] + wavelength[:-2])
        lim2 = 0.5 * (wavelength[1:-1] + wavelength[2:])
        smoothed_flux = smoothed_flux + fl*((lim2-lim1)*0.5*(gauss(lim1,wl,psf(wl)*0.0045)+gauss(lim2,wl,psf(wl)*0.0045)))
    
    smoothed_flux = np.insert(smoothed_flux,0,0)
    smoothed_flux = np.append(smoothed_flux,0)
    
    return np.array(smoothed_flux)


def make_nonlinear(frame, non_linear_coeffs_fits):  # Angelos' code
    """ Takes a HST WFC3 style fits file containing non-linearity coeffs and
    scales the frame appropriately. Importantly this makes a linear frame
    non-linear

    :param frame: the frame to make non-linear
    :param non_linear_coeffs_fits: file containing non-linear coeffs
    :return:
    """

    with fits.open(non_linear_coeffs_fits) as f:
        # cropping scales the non-linear frame to the input frame
        crop1 = len(f[1].data) / 2 - len(frame) / 2
        crop2 = len(f[1].data) / 2 + len(frame) / 2
        c1 = f[1].data[crop1:crop2, crop1:crop2]
        c2 = f[2].data[crop1:crop2, crop1:crop2]
        c3 = f[3].data[crop1:crop2, crop1:crop2]
        c4 = f[4].data[crop1:crop2, crop1:crop2]

    non_linear_frame = np.zeros_like(frame)

    for i in xrange(len(frame)):  # finding roots isn't vectorised
        for j in xrange(len(frame[0])):
            roots = np.real(np.roots([c4[i][j], c3[i][j], c2[i][j],
                                      c1[i][j] + 1, -frame[i][j]]))
            non_linear_frame[i][j] = roots[np.argmin((roots - frame[i][j]) ** 2)]

    return non_linear_frame


def get_limb_darkening_coeffs(star):
    """ fetches the limb darkening paremeters for the star. Currently just
    looks them up with pylc.

    I exported the function here as its called both in observation and in
    exposure (in order to add to the header)

    :param star: exodata star object
    """

    print pylc.ldcoeff(star.Z, float(star.T), star.calcLogg(), 'J')

    return pylc.ldcoeff(star.Z, float(star.T), star.calcLogg(), 'J')