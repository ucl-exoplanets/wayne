""" An exposure object, this is designed to be a fits file like object, taking input from a generation function and able
to construct the output along with certain visualisation methods.
"""

import datetime
import os.path
import sys

import astropy
import astropy.io.fits as fits
import astropy.units as u
import numpy as np
import pandas
import scipy
import exodata.astroquantities as pq

import params
import tools


class Exposure(object):
    def __init__(self, detector=None, filter=None, planet=None, exp_info=None):
        """ Sets up the exposure class which holds the exposure, its reads, and
         generates the headers and fits files

        :param exp_info: a dictionary with most of the information about the
         exposure. In future could be replaced by
        an exp plan class, which is used for visiting planning
        :type exp_info: dict
        :param detector: initialised detector class
        :type detector: detector.WFC3_I
        :param planet: and exodata planet object being observed
        :type planet: exodata.astroclasses.Planet
        :return:
        """

        self.detector = detector
        self.filter = filter
        self.planet = planet
        self.exp_info = exp_info

        self.SUBARRAY = exp_info['SUBARRAY']
        self.NSAMP = exp_info['NSAMP']
        self.SAMPSEQ = exp_info['SAMPSEQ']

        self.reads = []  # read 0 ->

    def apply_non_linear(self):
        """ Uses the detector apply_non_linearity() method to make each sample
        up the ramp non-linear. This correction is performed here as the
        data is already part of the exposure class.
        """

        for i, (pixel_array, header) in enumerate(self.reads[1:], 1):
            pixel_array_non_linear = self.detector.apply_non_linearity(pixel_array)

            self.reads[i] = (pixel_array_non_linear, header)

    def add_read_noise(self):
        """ Uses the detector add_read_noise() method on each sample
        """

        for i, (pixel_array, header) in enumerate(self.reads):
            pixel_array_rdnoise = self.detector.add_read_noise(pixel_array)

            self.reads[i] = (pixel_array_rdnoise, header)

    def add_dark_current(self):
        """ Adds dark current using the detectors add_dark_current method
        """

        for i, (pixel_array, header) in enumerate(self.reads[1:], 1):
            read_NSAMP = i + 1
            pixel_array_drk = self.detector.add_dark_current(
                pixel_array, read_NSAMP, self.SUBARRAY,
                self.SAMPSEQ)

            self.reads[i] = (pixel_array_drk, header)

    def scale_counts_between_limits(self):
        """ Gets count limits from the detector and scales the flux between
        these limits
        """

        min_count = self.detector.min_counts
        max_count = self.detector.max_counts

        for i, (pixel_array, header) in enumerate(self.reads):
            pixel_array_scaled = np.clip(pixel_array, min_count, max_count)
            self.reads[i] = (pixel_array_scaled, header)

    def add_zero_read(self):
        """ Adds the zero read to every other exposure, as this must be done
        last for other corrections to be valid
        :return:
        """

        zero_read = self.reads[0][0]  # [zero read][pixel array]

        for i, (pixel_array, header) in enumerate(self.reads[1:], 1):
            pixel_array_zero_read = pixel_array + zero_read
            self.reads[i] = (pixel_array_zero_read, header)

    def add_read(self, data, read_info=None):
        """ Adds the read to the exposure. Reads should be added in time order
        from the zero read to the final read

        :param data: an array to add
        :type data: np.ndarray
        :return:
        """

        if read_info is not None:
            header = self.generate_read_header(read_info)
        else:
            header = fits.Header()

        self.reads.append((data, header))

    def reset_reference_pixels(self, value=0.):
        """ Resets the reference pixels (5 pixel border) to zero
        :return:
        """

        for i, (pixel_array, header) in enumerate(self.reads):
            ref_is_true = np.ones_like(pixel_array, dtype='bool_')
            ref_is_true[5:-5, 5:-5] = 0
            pixel_array[ref_is_true] = value
            self.reads[i] = (pixel_array, header)

    def generate_fits(self, out_dir='', filename=None):
        """ Saves the exposure as a HST style fits file.

        :param out_dir: director to save output
        :type out_dir: str
        :param filename: filename to save the fits
        :type filename: str

        :return:
        """

        assert (len(self.reads) == (self.exp_info['NSAMP'])),\
            'Reads {} != NSAMP {}'.format(len(self.reads), self.exp_info['NSAMP'])

        if filename is None:
            filename = self.exp_info['filename']

        out_path = os.path.join(out_dir, filename)

        science_header = self.generate_science_header()

        hdulist = fits.HDUList([science_header])

        compression = 'RICE_1'

        for i, (data, header) in enumerate(reversed(self.reads)):
            # compression currently disabled as its producing stripey data
            read_HDU = fits.ImageHDU(data, header)

            error_array = fits.CompImageHDU(compression_type=compression)

            """ This array contains 16 independent flags indicating various
            status and problem conditions associated with each corresponding
            pixel in the science image. Each flag has a true (set) or false
            (unset) state and is encoded as a bit in a 16-bit integer word.
            Users are advised that this word should not be interpreted as as
            simple integer, but must be converted to base-2 and each bit
            interpreted as a flag. Table 2.5 lists the WFC3 data quality flags.
            """
            data_quality_array = fits.CompImageHDU(
                compression_type=compression)

            """ This array is present only for IR data. It is a 16-bit integer
            array and contains the number of samples used to derive the
            corresponding pixel values in the science image. For raw and
            intermediate data files, the sample values are set to the number of
            readouts that contributed to the science image. For calibrated
            files, the SAMP array contains the total number of valid samples
            used to compute the final science image pixel value, obtained by
            combining the data from all the readouts and rejecting cosmic
            ray hits and saturated pixels. Similarly, when multiple
            exposures (i.e., REPEAT-OBS) are combined to produce a single
            image, the SAMP array contains the total number of samples
            retained at each pixel for all the exposures.
            """
            samples_HDU = fits.CompImageHDU(compression_type=compression)

            """ This array is present only for IR data. This is a
            floating-point array that contains the effective integration
            time associated with each corresponding science image pixel value.
            For raw and intermediate data files, the time value is the total
            integration time of data that contributed to the science image.
            For calibrated datasets, the TIME array contains the combined
            exposure time of the valid readouts or exposures that were used
            to compute the final science image pixel value, after rejection of
            cosmic rays and saturated pixels from the intermediate data.
            """
            integration_time_HDU = fits.CompImageHDU(
                compression_type=compression)

            hdulist.extend(
                [read_HDU, error_array, data_quality_array, samples_HDU,
                 integration_time_HDU])

        hdulist.writeto(out_path)

    def generate_science_header(self):
        """ Generates the primary science header to match HST plus some
         information about the simulation

        :return: fits header
        :rtype: astropy.io.fits.PrimaryHDU
        """

        exp_info = self.exp_info

        science_header = fits.PrimaryHDU()
        h = science_header.header
        now = datetime.datetime.now()

        h['DATE'] = (
        now.strftime("%Y-%m-%d"), 'date this file was written (yyyy-mm-dd)')
        h['FILENAME'] = (exp_info['filename'], 'name of file')
        h['FILETYPE'] = ('SCI', 'type of data found in data file')
        h[''] = ''

        h['TELESCOP'] = (
        self.detector.telescope, 'telescope used to acquire data')
        h['INSTRUME'] = (self.detector.instrument,
                         'identifier for instrument used to acquire data')
        h['EQUINOX'] = (2000.0, 'equinox of celestial coord. system')

        h[''] = ''
        h[''] = '/ DATA DESCRIPTION KEYWORDS'
        h[''] = ''
        # h['ROOTNAME'] = ('i.e ibh707kcq', 'rootname of the observation set')
        # h['IMAGETYP'] = ('EXT', 'type of exposure identifier')
        h['PRIMESI'] = (
        self.detector.instrument, 'instrument designated as prime')

        h[''] = ''
        h[''] = '/ TARGET INFORMATION'
        h[''] = ''
        try:
            target_name = self.planet.name
        except AttributeError:
            target_name = 'None'
        h['TARGNAME'] = (target_name, 'proposer\'s target name')

        try:
            ra = self.planet.ra.degree
        except AttributeError:
            ra = 0.
        h['RA_TARG'] = (ra, 'right ascension of the target (deg) (J2000)')

        try:
            dec = self.planet.dec.degree
        except AttributeError:
            dec = 0.
        h['DEC_TARG'] = (dec, 'declination of the target (deg) (J2000)')

        h[''] = ''
        h[''] = '/ EXPOSURE INFORMATION'
        h[''] = ''
        # These need calculating from the start MJD and exptime
        expstart = exp_info['EXPSTART'].value - 2400000.5
        # TODO convert to UT
        h['DATE-OBS'] = (False, 'UT date of start of observation (yyyy-mm-dd)')
        h['TIME-OBS'] = (False, 'UT time of start of observation (hh:mm:ss)')
        h['EXPSTART'] = (
        expstart, 'exposure start time (Modified Julian Date)')
        h['EXPEND'] = (exp_info['EXPEND'].value - 2400000.5,
                       'exposure end time (Modified Julian Date)')
        h['EXPTIME'] = (exp_info['EXPTIME'].to(u.s).value,
                        'exposure duration (seconds)--calculated')
        # h['EXPFLAG'] = ('INDETERMINATE', 'Exposure interruption indicator')

        h[''] = ''
        h[''] = '/ TARGET OFFSETS (POSTARGS)'
        h[''] = ''
        h['POSTARG1'] = (0., 'POSTARG in axis 1 direction')
        # + for down scan, - for up scan, 0 staring?
        h['POSTARG2'] = (exp_info['SCAN_DIR'], 'POSTARG in axis 2 direction')

        h[''] = ''
        h[''] = '/ INSTRUMENT CONFIGURATION INFORMATION'
        h[''] = ''
        h['OBSTYPE'] = (
        exp_info['OBSTYPE'], 'observation type - imaging or spectroscopic')
        h['OBSMODE'] = (
        'MULTIACCUM', 'operating mode')  # no other modes for WFC3 IR?
        h['SCLAMP'] = ('NONE', 'lamp status, NONE or name of lamp which is on')
        # h['NRPTEXP'] = (1, 'number of repeat exposures in set: default 1')
        if not exp_info['SUBARRAY'] == 1024:
            SUBARRAY = True
        else:
            SUBARRAY = False
        h['SUBARRAY'] = (SUBARRAY, 'data from a subarray (T) or full frame (F)')
        h['SUBTYPE'] = ('SQ{}SUB'.format(exp_info['SUBARRAY']))
        h['DETECTOR'] = (self.detector.detector_type, 'detector in use: UVIS or IR')
        h['FILTER'] = (self.filter.name, 'element selected from filter wheel')
        h['SAMP_SEQ'] = (
        exp_info['SAMPSEQ'], 'MultiAccum exposure time sequence name')
        h['NSAMP'] = (exp_info['NSAMP'], 'number of MULTIACCUM samples')
        # TODO (ryan) add when  known
        h['SAMPZERO'] = (0., 'sample time of the zeroth read (sec)')
        APNAME = 'GRISM{}'.format(exp_info['SUBARRAY'])  # TODO (ryan) fix for non grism
        h['APERTURE'] = (APNAME, 'aperture name')
        h['PROPAPER'] = ('', 'proposed aperture name')  # is this always null?
        # TODO (ryan) change when true
        h['DIRIMAGE'] = ('NONE', 'direct image for grism or prism exposure')

        h[''] = ''
        h[''] = '/ WFC3Sim'
        h[''] = ''
        h['SIM'] = (True, 'WFC3Sim Simulation (T/F)')
        from __init__ import __version__
        h['SIM-VER'] = (__version__, 'WFC3Sim Version Used')
        h['SIM-TIME'] = (exp_info['sim_time'].to(u.s).value,
                         'WFC3Sim exposure generation time (s)')
        h[''] = ''
        h['X-REF'] = (exp_info['x_ref'], 'x position of star on frame (full frame))')
        h['Y-REF'] = (exp_info['y_ref'], 'y position of star on frame (full frame))')
        h['SAMPRATE'] = (exp_info['samp_rate'].to(u.ms).value,
                         'How often exposure is sampled (ms)')
        h['NSE-MEAN'] = (
        exp_info['noise_mean'], 'mean of normal noise (per s per pix)')
        h['NSE-STD'] = (
        exp_info['noise_std'], 'std of normal noise (per s per pix)')
        h['ADD-DRK'] = (exp_info['add_dark'], 'dark current added (T/F)')
        h['ADD-FLAT'] = (exp_info['add_flat'], 'flat field added (T/F)')
        h['ADD-GAIN'] = (exp_info['add_gain'], 'gain added (T/F)')
        h['ADD-NLIN'] = (exp_info['add_non_linear'], 'non-linearity effects added (T/F)')

        h['CSMCRATE'] = (exp_info['cosmic_rate'], 'Rate of cosmic hits (per s)')

        sky_background = exp_info['sky_background'].to(u.ct/u.s).value
        h['SKY-LVL'] = (sky_background, 'multiple of master sky per s')
        h['VSTTREND'] = (exp_info['scale_factor'], 'visit trend scale factor')
        h['CLIPVALS'] = (exp_info['clip_values_det_limits'], 'pixels clipped to detector range (T/F)')

        h['RANDSEED'] = (params.seed, 'seed used for the visit')


        h[''] = ''
        h[''] = '/ WFC3Sim Package Versions Used'
        h[''] = ''
        s = sys.version_info
        py_ver = '{}.{}.{} {}'.format(s.major, s.minor, s.micro,
                                      s.releaselevel)
        h['V-PY'] = (py_ver, 'Python version used')
        h['V-NP'] = (np.__version__, 'np version used')
        h['V-SP'] = (scipy.__version__, 'Scipy version used')
        h['V-AP'] = (astropy.__version__, 'Astropy version used')
        h['V-PD'] = (pandas.__version__, 'Pandas version used')

        planet = self.planet
        if planet is not None:
            h[''] = ''
            h[''] = '/ WFC3Sim Observation Parameters'
            h[''] = ''
            h['mid-tran'] = (float(planet.transittime), 'Time of mid transit (JD)')
            # h['t14'] = (float(planet.calcTransitDuration().rescale(pq.h)), 'Transit Duration (hr)')
            h['period'] = (float(planet.P.rescale(pq.day)), 'Orbital Period (days)')
            h['SMA'] = (float((planet.a / planet.star.R).simplified), 'Semi-major axis (a/R_s)')
            h['INC'] = (float(planet.i.rescale(pq.deg)), 'Orbital Inclination (deg)')
            h['ECC'] = (planet.e, 'Orbital Eccentricity')
            h['PERI'] = ('{}'.format(planet.periastron), 'Argument or periastron')

            # Limb Darkening
            ld1, ld2, ld3, ld4 = tools.get_limb_darkening_coeffs(self.planet.star)
            h['ld1'] = (ld1, 'Non-linear limb darkening coeff 1')
            h['ld2'] = (ld2, 'Non-linear limb darkening coeff 2')
            h['ld3'] = (ld3, 'Non-linear limb darkening coeff 3')
            h['ld4'] = (ld4, 'Non-linear limb darkening coeff 4')

        # not in correct section
        # keywords for analysis (i.e. xref positions until)
        h['STARX'] = (exp_info['x_ref'], 'x position of star on frame (full frame))')

        return science_header

    def generate_read_header(self, read_info):
        """ generates the header for a single data read

        :param read_info: dictionary of parameters from the simulation
        :return:
        """

        read_header = fits.Header()
        h = read_header

        h['CRPIX1'] = (read_info['CRPIX1'], 'x-coordinate of reference pixel')

        h['SAMPTIME'] = (read_info['cumulative_exp_time'].value, 'total integration time (sec)')
        h['DELTATIM'] = (read_info['read_exp_time'].value, 'sample integration time (sec)')


        return read_header