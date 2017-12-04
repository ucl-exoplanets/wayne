""" Contains functions useful to the program
"""

import numpy as np
import pandas as pd
import pyfits as fits
import pysynphot
import ephem

import pylightcurve as pylc


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

def crop_spectrum_ind(min_wl, max_wl, wl):
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

    return imin, imax


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


def load_and_sort_spectrum(file_path):
    """ must be in the format wl, flux / depth
    :param file:
    :return:
    """

    df = pd.read_table(file_path, sep=" ", header=None, names=['wl', 'flux'],
                       dtype={'wl': np.float64, 'flux': np.float64})

    try:
        df.sort('wl', inplace=True)
    except AttributeError:
        df.sort_values('wl', inplace=True)
        
    return df.wl.values, df.flux.values


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


def get_limb_darkening_coeffs(star):
    """ fetches the limb darkening paremeters for the star. Currently just
    looks them up with pylc.

    I exported the function here as its called both in observation and in
    exposure (in order to add to the header)

    :param star: exodata star object
    """

    return pylc.clablimb('claret', star.calcLogg(), float(star.T), star.Z, 'J', stellar_model='ATLAS')


def jd_to_hjd(jd, planet):
    """ converts jd to hjd for a given target

    :param star: exodata star object
    """
    ra_target, dec_target = (planet.system.ra.degree) * np.pi / 180, (planet.system.dec.degree) * np.pi / 180

    try:
        hjd = []

        for julian_date in jd:
            # calculate the position of the sun on the sky for this date

            sun = ephem.Sun()
            sun.compute(ephem.date(julian_date - 2415020))
            ra_sun, dec_sun = float(sun.ra), float(sun.dec)

            # calculate the correction coefficients

            a = 149597870700.0 / ephem.c
            b = np.sin(dec_target) * np.sin(dec_sun)
            c = np.cos(dec_target) * np.cos(dec_sun) * np.cos(ra_target - ra_sun)

            # apply the correction and save the result as the heliocentric_julian_date keyword

            heliocentric_julian_date = julian_date - (a * (b + c)) / (24.0 * 60.0 * 60.0)

            hjd.append(heliocentric_julian_date)

        return np.array(hjd)
    
    except TypeError:

        julian_date = jd

        # calculate the position of the sun on the sky for this date

        sun = ephem.Sun()
        sun.compute(ephem.date(julian_date - 2415020))
        ra_sun, dec_sun = float(sun.ra), float(sun.dec)

        # calculate the correction coefficients

        a = 149597870700.0 / ephem.c
        b = np.sin(dec_target) * np.sin(dec_sun)
        c = np.cos(dec_target) * np.cos(dec_sun) * np.cos(ra_target - ra_sun)

        # apply the correction and save the result as the heliocentric_julian_date keyword

        heliocentric_julian_date = julian_date - (a * (b + c)) / (24.0 * 60.0 * 60.0)

        return heliocentric_julian_date


def detect_orbits(exp_start_times, separation=0.028):
    """ Uses exposure time to detect orbits

    :param separation: minimum exptime seperation to count as a new orbit,
     the default of 0.028 is ~ 40 mins is just under half a HST orbit

    Caveats: only uses start time so may mess up if you do really long exposures

    :return: tuple of cutoff points of orbits. i.e if orbit one is 0-17,
     orbit 2 is 18-25 and orbit 3 is 26-30 you
    will get (0, 18, 26, None)
    :rtype: tuple
    """

    exp_start_times = np.array(exp_start_times)

    lastExpEnd = exp_start_times[0]
    orbitIndex = [0, ]  # Start at 0

    for i, exp_time in enumerate(exp_start_times):
        diff = exp_time - lastExpEnd
        if diff >= separation:
            orbitIndex.append(i)

        lastExpEnd = exp_time

    return orbitIndex


def wl_at_resolution(R, wl_min, wl_max):
    """ Produces a wavelength grid at a resolutionof R between the limits. The
    spacing is based on the mid point of the wl and is even throughout

    :param R: resolution
    :return:
    """

    mid_wl = (wl_max - wl_min)/2 + wl_min
    delta_wl = mid_wl/R

    return np.arange(wl_min, wl_max+delta_wl, delta_wl)


def crop_central_box(array, size):
    """ Crops the central size of pixels, Array must be square, and probably
     even numbered
    """

    index = (len(array) - size) / 2

    return array[index:-index, index:-index]


class WFC3IR_DQFlags(object):
    """ This class analyses a single data quality flag from the WFC3 IR camera.
    This includes general problem checking ie DQFlags.problems() aswell as
    looking for individual problems ie .isBad(), .isHot() etc.

    This class is from the EASE Pipeline

    Flags are WFC3 IR specific from 'WFC3 data handbook 2.1'
    """

    def __init__(self, flag):
        """
        :param flag: DQ flag
        :type flag: int
        """

        self.flag = flag
        self.binary = '{0:015b}'.format(flag)

        self._flagList()  # sets flagLists

    def isBad(self):  # Notes as self.binary is a str then all slices are also str
        """ Checks weather the pixel is bad by seeing if ANY flags are raised
        :return: True or False
        :rtype: bool
        """
        print self.binary, type(self.binary)
        if '1' in self.binary:
            return True
        else:
            return False

    def isRSError(self):
        if self.binary[14] is '1':
            return True
        else:
            return False

    def isFilled(self):
        if self.binary[13] is '1':
            return True
        else:
            return False

    def isBadPix(self):
        if self.binary[12] is '1':
            return True
        else:
            return False

    def isDeviantZero(self):
        if self.binary[11] is '1':
            return True
        else:
            return False

    def isHot(self):
        if self.binary[10] is '1':
            return True
        else:
            return False

    def isUnstable(self):
        if self.binary[9] is '1':
            return True
        else:
            return False

    def isWarm(self):
        if self.binary[8] is '1':
            return True
        else:
            return False

    def isBadRef(self):
        if self.binary[7] is '1':
            return True
        else:
            return False

    def isSaturated(self):
        if self.binary[6] is '1':
            return True
        else:
            return False

    def isBadFlat(self):
        if self.binary[5] is '1':
            return True
        else:
            return False

    def isReserved(self):
        if self.binary[4] is '1':
            return True
        else:
            return False

    def isZeroSignal(self):
        if self.binary[3] is '1':
            return True
        else:
            return False

    def isCosmic(self):
        """ there are two methods for checking cosmics, this returns True if either raise the flag
        :return:
        """
        if self.binary[2] is '1' or self.binary[1] is '1':
            return True
        else:
            return False

    def isCosmic_drizzle(self):
        if self.binary[2] is '1':
            return True
        else:
            return False

    def isCosmic_calwf3(self):
        if self.binary[1] is '1':
            return True
        else:
            return False

    def isGhost(self):
        if self.binary[0] is '1':
            return True
        else:
            return False

    def problems(self):
        """ returns a list of all the problem flags

        :return: list of flag shortnames
        :rtype: list
        """
        problems = []
        for i, num in enumerate(list(self.binary)):
            if num is '1':
                problems.append(self.flagList[i][1])

        return problems

    def _isFlag(self, flag):  # TODO rewrite and test this (currently tested by data)
        """ Takes a flag or tuple/list of flags as input and returns True if any of the flags are raised and false if
        none are

        :return: True / False
        :rtype: bool
        """

        flagList = self.flagList + self.extraFlags

        for flagInfo in flagList:  # Not the most efficient way but good enough
            flagNum = flagInfo[0]
            flagCode = flagInfo[1]
            if flag == flagCode:
                if type(flagNum) is int:
                    if self.binary[flagNum] is '1':
                        return True
                    else:
                        return False
                else:
                    for flagNumber in flagNum:  # In this case there are multiple flagnums in flagnum
                        if self.binary[flagNumber] is '1':
                            return True

                    return False  # if we're here no flags matched in the list


        raise ValueError('{} is not a valid flag'.format(flag))

    def _flagList(self):
        """ just an list of flags in order from binary 16384 to 0 (ie backwards) with flags grouped in extra flags
        function sets them in the class
        """

        self.flagList = [  # Format - position of flag / flag shortname / flag description
            (0, 'ghost', 'ghost/crosstalk'),
            (1, 'cosmic-calwf3', 'cosmic ray (calwf3 up-the-ramp fitting)'),
            (2, 'cosmic-drizzle', 'cosmic ray (MultiDrizzle)'),
            (3, 'zero-signal', 'Signal in zero-read'),
            (4, 'reserved', '(Reserved)'),
            (5, 'badflat', 'Bad or uncertain flat value'),
            (6, 'saturated', 'Full-well saturation'),
            (7, 'badref', 'Bad reference pixel'),
            (8, 'warm', 'Warm pixel'),
            (9, 'unstable', 'Unstable response'),
            (10, 'hot', 'Hot pixel'),
            (11, 'deviantzero', 'Deviant zero-read (bias) value'),
            (12, 'badpix', 'Bad detector pixel'),
            (13, 'filled', 'Data missing and replaced by fill value'),
            (14, 'reedsolomon', 'Reed-Solomon decoding error'),
        ]
        self.extraFlags = [  # combines flags into more general flags
            ((1, 2), 'cosmic', 'Cosmic ray')
        ]
