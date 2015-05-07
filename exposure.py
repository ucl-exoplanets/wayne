""" An exposure object, this is designed to be a fits file like object, taking input from a generation function and able
to construct the output along with certain visualisation methods.
"""

import datetime
import os.path

import astropy.io.fits as fits
import astropy.units as u


class Exposure(object):

    def __init__(self, detector=None, filter=None, planet=None, exp_info=None):
        """ Sets up the exposure class which holds the exposure, its reads, and generates the headers and fits files

        :param exp_info: a dictionary with most of the information about the exposure. In future could be replaced by
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

        self.subarray = exp_info['SUBARRAY']

        self.reads = []  # read 0 ->

    def add_read(self, data):
        """ Adds the read to the exposure, will probably need some header information to in future.

        Reads should be added in time order from the zero read to the final read

        :param data: an array to add
        :type data: numpy.ndarray
        :return:
        """

        data = self.crop_subarrray(data, self.subarray)
        self.reads.append(data)

    def generate_fits(self, out_dir='', filename=None):
        """ Saves the exposure as a HST style fits file.

        :param out_dir: director to save output
        :type out_dir: str
        :param filename: filename to save the fits
        :type filename: str

        :return:
        """

        assert(len(self.reads) == (self.exp_info['NSAMP'] + 1))

        if filename is None:
            filename = self.exp_info['filename']

        out_path = os.path.join(out_dir, filename)

        science_header = self.generate_science_header()

        hdulist = fits.HDUList([science_header])

        for i, data in enumerate(reversed(self.reads)):

            # data = np.flipud(data)  # hst fits files are the other way round

            read_HDU = fits.ImageHDU(data)
            error_array = fits.ImageHDU()

            """ This array contains 16 independent flags indicating various status and problem conditions associated
            with each corresponding pixel in the science image. Each flag has a true (set) or false (unset) state and is
            encoded as a bit in a 16-bit integer word. Users are advised that this word should not be interpreted as a
            simple integer, but must be converted to base-2 and each bit interpreted as a flag. Table 2.5 lists the WFC3
            data quality flags.
            """
            data_quality_array = fits.ImageHDU()

            """ This array is present only for IR data. It is a 16-bit integer array and contains the number of samples
            used to derive the corresponding pixel values in the science image. For raw and intermediate data files, the
            sample values are set to the number of readouts that contributed to the science image. For calibrated files,
            the SAMP array contains the total number of valid samples used to compute the final science image pixel
            value, obtained by combining the data from all the readouts and rejecting cosmic ray hits and saturated
            pixels. Similarly, when multiple exposures (i.e., REPEAT-OBS) are combined to produce a single image,
            the SAMP array contains the total number of samples retained at each pixel for all the exposures.
            """
            samples_HDU = fits.ImageHDU()

            """ This array is present only for IR data. This is a floating-point array that contains the effective
            integration time associated with each corresponding science image pixel value. For raw and intermediate
            data files, the time value is the total integration time of data that contributed to the science image.
            For calibrated datasets, the TIME array contains the combined exposure time of the valid readouts or
            exposures that were used to compute the final science image pixel value, after rejection of cosmic
            rays and saturated pixels from the intermediate data.
            """
            integration_time_HDU = fits.ImageHDU()

            hdulist.extend([read_HDU, error_array, data_quality_array, samples_HDU, integration_time_HDU])

        hdulist.writeto(out_path)

    def crop_subarrray(self, data, subarray):
        """ Takes a full frame array and crops it down to the subarray size.

        :param data: data array
        :type data: numpy.ndarray
        :param subarray: subbarray mode (1024, 512, 256, 128, 64)
        :type subarray: int

        :return: cropped array
        :rtype: numpy.ndarray
        """

        i_lower = (1024-subarray)/2
        i_upper = i_lower + subarray

        return data[i_lower:i_upper, i_lower:i_upper]

    def generate_science_header(self):
        """ Generates the primary science header to match HST plus some information about the simulation

        :return: fits header
        :rtype: astropy.io.fits.PrimaryHDU
        """

        exp_info = self.exp_info

        science_header = fits.PrimaryHDU()
        h = science_header.header
        now = datetime.datetime.now()
        h['DATE'] = (now.strftime("%Y-%m-%d"), 'date this file was written (yyyy-mm-dd)')
        h['FILENAME'] = (exp_info['filename'], 'name of file')
        h['FILETYPE'] = ('SCI', 'type of data found in data file')
        h[''] = ''

        h['TELESCOP'] = (self.detector.telescope, 'telescope used to acquire data')
        h['INSTRUME'] = (self.detector.instrument, 'identifier for instrument used to acquire data')
        h['EQUINOX'] = (2000.0, 'equinox of celestial coord. system')

        h[''] = ''
        h[''] = '/ DATA DESCRIPTION KEYWORDS'
        h[''] = ''
        # h['ROOTNAME'] = ('i.e ibh707kcq', 'rootname of the observation set')
        # h['IMAGETYP'] = ('EXT', 'type of exposure identifier')
        h['PRIMESI'] = (self.detector.instrument, 'instrument designated as prime')

        h[''] = ''
        h[''] = '/ TARGET INFORMATION'
        h[''] = ''
        h['TARGNAME'] = (self.planet.name, 'proposer\'s target name')
        # TODO format is wrong, 00 00 00 vs 2.405492604190E+02
        h['RA_TARG'] = (self.planet.ra, 'right ascension of the target (deg) (J2000)')
        h['DEC_TARG'] = (self.planet.dec, 'declination of the target (deg) (J2000)')

        h[''] = ''
        h[''] = '/ EXPOSURE INFORMATION'
        h[''] = ''
        # These need calculating from the start MJD and exptime
        expstart = exp_info['EXPSTART'].value - 2400000.5
        # TODO convert to UT
        h['DATE-OBS'] = (False, 'UT date of start of observation (yyyy-mm-dd)')
        h['TIME-OBS'] = (False, 'UT time of start of observation (hh:mm:ss)')
        h['EXPSTART'] = (expstart, 'exposure start time (Modified Julian Date)')
        h['EXPEND'] = (exp_info['EXPEND'].value - 2400000.5, 'exposure end time (Modified Julian Date)')
        h['EXPTIME'] = (exp_info['EXPTIME'].to(u.s).value, 'exposure duration (seconds)--calculated')
        # h['EXPFLAG'] = ('INDETERMINATE', 'Exposure interruption indicator')

        h[''] = ''
        h[''] = '/ TARGET OFFSETS (POSTARGS)'
        h[''] = ''
        h['POSTARG1'] = (0., 'POSTARG in axis 1 direction')
        # + for down scan, - for up scan, 0 staring?
        h['POSTARG2'] = (exp_info['SCAN_DIR'] , 'POSTARG in axis 2 direction')

        h[''] = ''
        h[''] = '/ WFC3Sim'
        h[''] = ''
        h['SIM'] = (True, 'WFC3Sim Simulation (T/F)')
        from __init__ import __version__
        h['SIM-VER'] = (__version__, 'WFC3Sim Version Used')
        h['SIM-TIME'] = (exp_info['sim_time'].to(u.s).value, 'WFC3Sim exposure generation time (s)')
        h[''] = ''
        h['PSF-MAX'] = (exp_info['psf_max'], 'maximum width of psf tails (pix)')
        h['SAMPRATE'] = (exp_info['samp_rate'].to(u.ms).value, 'How often exposure is sampled (ms)')
        h[''] = (exp_info['scan_speed_var'], 'Scan speed variations (stddev as a % of flux)')

        # keywords for analysis (i.e. xref positions until)
        h['STARXI'] = (exp_info['x_ref'], 'x position of star on frame (full frame))')


        h[''] = ''
        h[''] = '/ INSTRUMENT CONFIGURATION INFORMATION'
        h[''] = ''
        h['OBSTYPE'] = (exp_info['OBSTYPE'], 'observation type - imaging or spectroscopic')
        h['OBSMODE'] = ('MULTIACCUM', 'operating mode')  # no other modes for WFC3 IR?
        h['SCLAMP'] = ('NONE', 'lamp status, NONE or name of lamp which is on')
        # h['NRPTEXP'] = (1, 'number of repeat exposures in set: default 1')
        if not exp_info['SUBARRAY'] == 1024:
            SUBARRAY = True
        else:
            SUBARRAY = False
        h['SUBARRAY'] = (SUBARRAY, 'data from a subarray (T) or full frame (F)')
        # h['SUBTYPE'] = () e.g 'SQ128SUB'
        h['DETECTOR'] = (self.detector.detector_type, 'detector in use: UVIS or IR')
        h['FILTER'] = (self.filter.name, 'element selected from filter wheel')
        h['SAMP_SEQ'] = (exp_info['SAMPSEQ'], 'MultiAccum exposure time sequence name')
        h['NSAMP'] = (exp_info['NSAMP'], 'number of MULTIACCUM samples')
        h['SAMPZERO'] = (0., 'sample time of the zeroth read (sec)')  # TODO add when known
        APNAME = 'GRISM{}'.format(exp_info['SUBARRAY'])  # TODO fix for non grism
        h['APERTURE'] = (APNAME, 'aperture name')
        h['PROPAPER'] = ('', 'proposed aperture name')  # is this always null?
        h['DIRIMAGE'] = ('NONE', 'direct image for grism or prism exposure')  # TODO change when true

        return science_header