""" The detector class should map any fields from the telescope to a pixel map
 on the detector.
"""

import os.path

import numpy as np
from astropy import units as u
import astropy.io.fits as fits
import pandas as pd

import params
import tools


class WFC3_IR(object):
    """ Class containg the various methods and calibrations for the WFC3 IR detector
    """

    def __init__(self):
        """ Initialise the WFC3 IR class.
        :return:
        """

        # Start with a single 1024x1024 array, add complexity in when we need it.
        self._pixel_array = np.zeros((1024, 1024))
        self.pixel_unit = u.Unit('WFC3IR_Pix', 18*u.micron,
                                 doc='Pixel size for the HST WFC3 IR detector')
        self.telescope_area = np.pi * (2.4*u.m/2.)**2
        self.min_counts = -20
        # 5% non-linear limit where nonlinear correction fails
        self.max_counts = 78000  # DN

        self.constant_gain = 2.35
        self.gain_file = os.path.join(params._calb_dir, 'u4m1335mi_pfl.fits')

        self.read_noise = 14.1 / self.constant_gain  # e to DN

        # TODO needs values for other subbarrays
        self.initial_bias = os.path.join(params._data_dir, 'wfc3_ir_initial_bias_256.fits')

        # General Info
        self.telescope = 'HST'
        self.instrument = 'WFC3'
        self.detector_type = 'IR'

        # Init
        self.modes_exp_table, self.modes_calb_table = self._get_modes()

        # QE
        self.qe_file = os.path.join(params._data_dir, 'wfc3_ir_qe_003_syn.fits')
        with fits.open(self.qe_file) as f:
            tbl = f[1].data  # the table is in the data of the second HDU
            self.qe_wl = (tbl.field('WAVELENGTH') * u.angstrom).to(u.micron)
            self.qe_val = tbl.field('THROUGHPUT')

        # Non-linearity
        self.non_linear_file = os.path.join(params._calb_dir, 'u1k1727mi_lin.fits')

    def exptime(self, NSAMP, SUBARRAY, SAMPSEQ):
        """ Retrieves the total exposure time for the modes given

        :param NSAMP: number of sample up the ramp, effects exposure time (2 to 16)
        :type NSAMP: int
        :param SAMPSEQ: Sample sequence to use, effects exposure time
        ('RAPID', 'SPARS10', 'SPARS25', 'SPARS50',
        'SPARS100', 'SPARS200', 'STEP25', 'STEP50', 'STEP100', 'STEP200', 'STEP400'
        :type SAMPSEQ: str
        :param SUBARRAY: subarray to use, effects exposure time and array size.
         (1024, 512, 256, 128, 64)
        :type SUBARRAY: int

        :return: exposure time
        :rtype: astropy.units.quantity.Quantity
        """

        sample_number = NSAMP - 1  # EXPTIME tables quote samp num not NSAMP

        mt = self.modes_exp_table

        exptime_table = mt.ix[(mt['SAMPSEQ'] == SAMPSEQ) &
                              (mt['SAMPNUM'] == sample_number) &
                              (mt['SUBARRAY'] == SUBARRAY)]

        if exptime_table.empty:
            raise WFC3SimSampleModeError(
                "SAMPSEQ = {}, NSAMP={}, SUBARRAY={} is not a permitted combination"
                "".format(SAMPSEQ, NSAMP, SUBARRAY))
        else:
            exptime = exptime_table.TIME.values[0]
            return exptime * u.s

    def gen_pixel_array(self, subarray, light_sensitive=True):
        """ Returns the pixel array as an array of zeroes

        this could return subarray types etc, but lets just keep it out of
         class for now

        :param light_sensitive: only return the light sensitive parts
         (neglecting 5 pixel border)
        :type light_sensitive: bool

        :return: full frame array of zeroes (based on set subarray)
        :rtype: numpy.ndarray
        """

        if light_sensitive:
            if subarray == 1024:
                subarray = 1014
            return np.zeros((subarray, subarray))
        else:
            size = subarray + 10
            if size > 1024:
                size = 1024
            return np.zeros((size, size))

    def add_bias_pixels(self, pixel_array):
        """ converts a light sensitive array to one with the 5 pixel border.
         In future will simulate the function
        of bias pixels but for now returns zeroes.

        :param pixel_array: light sensitive pixel array
        :type pixel_array: np.ndarray

        :return: pixel array with bias pixel border h+10, w+10
        :rtype: numpy.ndarray
        """

        allowed_input = (1014, 512, 256, 128, 64)

        array_size = len(pixel_array)

        if not array_size in allowed_input:
            raise ValueError('array size must be in {} got {}'.format(
                array_size, allowed_input))

        full_array = np.zeros((array_size+10, array_size+10))
        full_array[5:-5, 5:-5] = pixel_array

        return full_array

    def add_dark_current(self, pixel_array, NSAMP, SUBARRAY, SAMPSEQ):
        """ Adds the exact contribution from dark current as specified in
         the super_dark for that mode, must be done after bias pixels added

        :param pixel_array: light sensitive pixel array
        :type pixel_array: np.ndarray
        :param NSAMP: number of sample up the ramp, effects exposure time
         (1 to 15)
        :type NSAMP: int
        :param SAMPSEQ: Sample sequence to use, effects exposure time
         ('RAPID', 'SPARS10', 'SPARS25', 'SPARS50',
        'SPARS100', 'SPARS200', 'STEP25', 'STEP50', 'STEP100', 'STEP200', 'STEP400'
        :type SAMPSEQ: str
        :param SUBARRAY: subarray to use, effects exposure time and array size.
         (1024, 512, 256, 128, 64)
        :type SUBARRAY: int

        :return: pixel array with dark current added
        :rtype: numpy.ndarray
        """

        mt = self.modes_calb_table
        dark_query = mt.ix[(mt['SAMPSEQ'] == SAMPSEQ) &
                           (mt['SUBARRAY'] == SUBARRAY)]

        if dark_query.empty:
            raise WFC3SimNoDarkFileError(
                "No Dark file found for SAMPSEQ = {}, SUBARRAY={}"
                "".format(SAMPSEQ, SUBARRAY))
        else:
            dark_file = dark_query.Dark.values[0]

        dark_file_path = os.path.join(params._calb_dir, dark_file)

        file_index = -(NSAMP)*5

        with fits.open(dark_file_path) as f:
            dark_array = f[file_index].data
            dark_error_array = np.where(f[file_index + 1].data > 0, f[file_index + 1].data, 0.00001)
            return pixel_array + np.random.normal(dark_array, dark_error_array)

    def add_read_noise(self, pixel_array):
        """ Adds read noise, stellar noise and distribution noise to the frame
        values from pg 58 of the data handbook
        """

        return np.random.normal(pixel_array, self.read_noise)

    def get_gain(self, size):

        with fits.open(self.gain_file) as gain:
            gain_data = gain[1].data[5:-5, 5:-5]
            gain_data = self.constant_gain / gain_data

        if size is not None:
            gain_data = tools.crop_central_box(gain_data, size)

        return gain_data

    def get_read_times(self, NSAMP, SUBARRAY, SAMPSEQ):
        """ Retrieves the time of each sample up the ramp for the mode given

        :param NSAMP: number of sample up the ramp, effects exposure time (2 to 16)
        :type NSAMP: int
        :param SAMPSEQ: Sample sequence to use, effects exposure time
        ('RAPID', 'SPARS10', 'SPARS25', 'SPARS50',
        'SPARS100', 'SPARS200', 'STEP25', 'STEP50', 'STEP100', 'STEP200', 'STEP400'
        :type SAMPSEQ: str
        :param SUBARRAY: subarray to use, effects exposure time and array size.
         (1024, 512, 256, 128, 64)
        :type SUBARRAY: int

        :return: array of read times for each sample up the ramp to NSAMP
        :rtype: numpy.ndarray
        """

        if not 2 <= NSAMP <= 16:
            raise WFC3SimSampleModeError(
                "NSAMP must be an integer between 2 and 16, got {}".format(NSAMP))

        sample_number = NSAMP - 1  # EXPTIME tables quote samp num not NSAMP

        mt = self.modes_exp_table

        exptime_table = mt.ix[(mt['SAMPSEQ'] == SAMPSEQ) &
                              (mt['SAMPNUM'] <= sample_number) &
                              (mt['SUBARRAY'] == SUBARRAY)]['TIME']

        if exptime_table.empty:
            raise WFC3SimSampleModeError(
                "SAMPSEQ = {}, NSAMP={}, SUBARRAY={}  is not a permitted "
                "combination".format(SAMPSEQ, NSAMP, SUBARRAY))

        return np.array(exptime_table) * u.s

    def _get_modes(self):
        """ Retrieves table of exposure time for each NSAMP, SAMPSEQ and
         SUBARRAY type from data directory.

        :return: modes table
        :rtype: pandas.DataFrame
        """

        modes_exp_table = pd.read_csv(
            os.path.join(params._data_dir, 'wfc3_ir_mode_exptime.csv'),
            skiprows=1, dtype={ 'SUBARRAY': np.int64, 'SAMPSEQ': np.object,
                                'SAMPNUM': np.int64, 'TIME':np.float},
                                  thousands=',')

        modes_calb_table = pd.read_csv(
            os.path.join(params._data_dir, 'wfc3_ir_mode_calb.csv'),
            skiprows=1, dtype={'SUBARRAY': np.int64, 'SAMPSEQ': np.object,
                                'dark': str})

        return modes_exp_table, modes_calb_table

    def num_exp_per_buffer(self, NSAMP, SUBARRAY):
        """ calculates the maximum number of exposures that can be taken before
         buffer dumping. It does this by checking HST's limits on the number
         of frames (including sample up the ramps) of 304 and the size limit of
         2 full frame 16 sample exposures.

        :param NSAMP: number of sample up the ramp, effects exposure time (1 to 15)
        :type NSAMP: int
        :param SUBARRAY: subarray to use, effects exposure time and array size.
         (1024, 512, 256, 128, 64)
        :type SUBARRAY: int

        :return:  maximum number of exposures before a buffer dump
        :rtype: int
        """

        hard_limit = 304  # headers pg 208

        headers_per_exp = NSAMP + 1  # + 1 for zero read

        # 2 full frame (1024) 16 sample exposures
        total_allowed_reads = 2*16*(1024/SUBARRAY)

        if total_allowed_reads > hard_limit:
            total_allowed_reads = hard_limit

        num_exp = int(np.floor(total_allowed_reads / headers_per_exp))

        return num_exp

    def apply_quantum_efficiency(self, wl, counts):
        """ Applies quantum efficiency corrections to the counts using the
        data in wfc3_ir_qe_003_syn.fits and linearly
        interpolating the gaps

        :param wl: array of wavelengths (corresponding to stellar flux and
         planet spectrum) in u.microns
        :type wl: astropy.units.quantity.Quantity
        :param counts: flux / acounts
        :type counts: astropy.units.quantity.Quantity

        :return: counts scaled by QE
        :rtype: astropy.units.quantity.Quantity
        """

        throughput_values = np.interp(wl, self.qe_wl, self.qe_val, 0., 0.)

        return counts * throughput_values

    def apply_non_linearity(self, pixel_array):  # Angelos code
        """ This uses the non linearity correction (in reverse) to give the
        detector a non linear response. Units are in DN.

        :param pixel_array:
        :return:
        """

        with fits.open(self.non_linear_file) as f:
            # cropping scales the non-linear frame to the input frame
            crop1 = len(f[1].data) / 2 - len(pixel_array) / 2
            crop2 = len(f[1].data) / 2 + len(pixel_array) / 2
            c1 = f[1].data[crop1:crop2, crop1:crop2]
            c2 = f[2].data[crop1:crop2, crop1:crop2]
            c3 = f[3].data[crop1:crop2, crop1:crop2]
            c4 = f[4].data[crop1:crop2, crop1:crop2]

        non_linear_frame = np.zeros_like(pixel_array)

        for i in xrange(len(pixel_array)):  # finding roots isn't vectorised
            for j in xrange(len(pixel_array[0])):
                roots = np.real(np.roots([c4[i][j], c3[i][j], c2[i][j],
                                          c1[i][j] + 1, -pixel_array[i][j]]))
                non_linear_frame[i][j] = roots[np.argmin((roots - pixel_array[i][j]) ** 2)]

        return non_linear_frame


class WFC3SimException(BaseException):
    pass


class WFC3SimSampleModeError(WFC3SimException):
    pass


class WFC3SimNoDarkFileError(WFC3SimException):
    pass
