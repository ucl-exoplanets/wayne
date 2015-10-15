""" This module brings the other modules together to construct frames and generate a visits worth of frames
(Observation). ExposureGenerator combines a spectrum, grism + detector combo and other observables to construct an image
"""

import numpy as np
import quantities as pq
from astropy import units as u
import pylightcurve.fcmodel as pylc
import matplotlib.pylab as plt

from progress import Progress

from wfc3simlog import logger
import tools
import exposure
from trend_generators import visit_trends
from exposure_generator import ExposureGenerator
from visit_planner import VisitPlanner


class Observation(object):
    def __init__(self, outdir=''):
        """ Builds a full observation running the visit planner to get exposure
        times, generates lightcurves for each wavelength element and sample
        time and then runs the exposure generator for each frame.

        You must now call each setup function separately

        :param outdir: location on disk to save the output fits files to. Must exist.
        :type outdir: str

        :return: Nothing, output is saved to outdir
        """

        # TODO (ryan) option to set default values across the board
        # TODO (ryan) validation to instruct which ones need calling

        self.scanning = True
        self.outdir = outdir

        self._visit_trend = False

    def setup_observation(self, x_ref, y_ref, scan_speed):
        """

        :param x_ref: pixel in x axis the reference should be located
        :type x_ref: int

        :param y_ref: pixel in y axis the reference should be located
        :type y_ref: int

        :param scan_speed: rate of scan in pixels/second
        :type scan_speed: astropy.units.quantity.Quantity
        :return:
        """
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.scan_speed = scan_speed

    def setup_simulator(self, sample_rate, clip_values_det_limits=True,
                        psf_approx_factor=False):
        """
        :param sample_rate: How often to sample the exposure (time units)
        :type sample_rate: astropy.units.quantity.Quantity
        """
        self.sample_rate = sample_rate
        self.clip_values_det_limits = clip_values_det_limits
        self.psf_approx_factor = psf_approx_factor

    def setup_target(self, planet, planet_spectrum, planet_wl, stellar_flux,
                     stellar_wl=None):
        """
        :param planet: An ExoData type planet object your observing (holds
         observables like Rp, Rs, i, e, etc)
        :type: exodata.astroclasses.Planet

        :param planet_wl: array of wavelengths (corresponding to stellar flux and
         planet spectrum) in u.microns
        :type planet_wl: astropy.units.quantity.Quantity

        :param stellar_flux: array of stellar flux in units of erg/(angstrom * s * cm^2)
        :type stellar_flux: astropy.units.quantity.Quantity

        :param planet_spectrum: array of the transit depth for the planet spectrum
        :type planet_spectrum: numpy.ndarray

        :param stellar_wl: If given the stellar flux and planet signal are
        binned separately so we rebin the stellar spectrum to the planet bins
        """

        if stellar_wl is not None:
            stellar_flux = tools.rebin_spec(stellar_wl, stellar_flux, planet_wl)

        self.planet = planet
        self.wl = planet_wl
        self.stellar_flux = stellar_flux
        self.planet_spectrum = planet_spectrum
        self._generate_star_information()

    def _generate_star_information(self):
        star = self.planet.star
        # TODO (ryan) option to give limb darkening
        self.ldcoeffs = tools.get_limb_darkening_coeffs(star)
        logger.info(
            "Looked up Limb Darkening coeffs of {}".format(self.ldcoeffs))

    def setup_detector(self, detector, NSAMP, SAMPSEQ, SUBARRAY):
        """
        :param detector: The detector to use
        :type: detector.WFC3_IR

        :param NSAMP: number of sample up the ramp, effects exposure time (1 to 15)
        :type NSAMP: int

        :param SAMPSEQ: Sample sequence to use, effects exposure time ('RAPID',
         'SPARS10', 'SPARS25', 'SPARS50', 'SPARS100', 'SPARS200', 'STEP25',
         'STEP50', 'STEP100', 'STEP200', 'STEP400'
        :type SAMPSEQ: str

        :param SUBARRAY: subarray to use, effects exposure time and array size.
         (1024, 512, 256, 128, 64)
        :type SUBARRAY: int
        """
        self.detector = detector
        self.NSAMP = NSAMP
        self.SAMPSEQ = SAMPSEQ
        self.SUBARRAY = SUBARRAY

    def setup_grism(self, grism):
        """
        :param grism: The grism to use
        :type: grism.grism
        """
        self.grism = grism

    def setup_visit(self, start_JD, num_orbits, exp_start_times=False):
        """ Sets up visit information,

        :param start_JD: The JD the entire visit should start (without overheads)
        :type: float
        :param num_orbits: number of orbits to generate for
        :type: int
        :param exp_start_times: a list of start times for the exposures in JD
        :type: array
        """
        self.start_JD = start_JD  # in days
        self.num_orbits = num_orbits

        if exp_start_times:  # i.e. visit plan is specified
            self.exp_start_times = exp_start_times
            self._start_times_to_visit_info()
        else:
            # TODO detector must be setup first but no indicator of this in code
            self._generate_visit_plan()

    def _generate_visit_plan(self):
        """ Generates the visit plan, requires both setup_detector and
        setup_visit

        :return:
        """
        self.visit_plan = VisitPlanner(self.detector, self.NSAMP,
                                        self.SAMPSEQ, self.SUBARRAY,
                                        self.num_orbits,
                                        # TEMP! to make observations sparser
                                        exp_overhead=3 * u.min)

        self.exp_start_times = self.visit_plan['exp_times'].to(
            u.day) + self.start_JD

        # So this is a weird thing to do, maybe the JD should be added in the
        # visit planner - used in visit trend generation
        self.visit_plan['exp_start_times'] = self.exp_start_times

    def _start_times_to_visit_info(self):
        """ Used to provide the additional information the simulation needs
        from the visit planner when providing exposure times
        """

        self.visit_plan = {
            'exp_start_times': self.exp_start_times,
            # for visit trends
            'orbit_start_index': tools.detect_orbits(self.exp_start_times),
        }

    def setup_reductions(self, add_dark=True, add_flat=True, add_gain=True,
                         add_non_linear=True):
        """
        :param add_dark:
        :param add_flat:
        :return:
        """
        self.add_dark = add_dark
        self.add_flat = add_flat
        self.add_gain = add_gain
        self.add_non_linear = add_non_linear

    def setup_trends(self, ssv_std=False, ssv_period=False, x_shifts=0):
        """
        :param ssv_std: The % of the std(flux_per_pixel) the scan speed
         variations cause. Basic implementation.

        :param x_shifts: pixel fraction to shift the starting x_ref position by
         for each exposure
        :type x_shifts: float
        """
        self.ssv_std = ssv_std
        self.ssv_period = ssv_period
        self.x_shifts = x_shifts

    def setup_noise_sources(self, sky_background=1*u.count/u.s, cosmic_rate=11.,
                            add_final_noise_sources=True):

        self.sky_background = sky_background
        self.cosmic_rate = cosmic_rate
        self.add_final_noise_sources = add_final_noise_sources

    def setup_gaussian_noise(self, noise_mean=False, noise_std=False):
        """
        :param noise_mean: mean noise (per second, per pixel)
        :type noise_mean: float

        :param noise_std: standard deviation of the noise (per second, per pixel)
        :type noise_std: float
        """
        self.noise_mean = noise_mean
        self.noise_std = noise_std

    def setup_visit_trend(self, visit_trend_coeffs):
        """ Adds a visit long trend to the simulation
        :param coeffs: list of coeffs for the trend.

        In future will include multiple trends
        """

        self._visit_trend = visit_trends.HookAndLongTermRamp(
            self.visit_plan, visit_trend_coeffs)

    def generate_lightcurves(self, time_array, depth=False):
        """ Generates lightcurves samples a the time array using pylightcurve.
         orbital parameters are pulled from the planet object given in
         intialisation

        :param time_array: list of times (JD) to sample at
        :type time_array: numpy.ndarray
        :param depth: a signle depth or array of depths to generate for
        :type depth: float or numpy.ndarray

        :return: models, a 2d array containing a lightcurve per depth, sampled
         at timearray
        :rtype: numpy.ndarray
        """

        # TODO (ryan) quick check if out of transit, in that case ones!

        # TODO (ryan) should this be in generate exposure?

        planet = self.planet
        star = self.planet.star

        P = float(planet.P.rescale(pq.day))
        a = float((planet.a / star.R).simplified)
        i = float(planet.i.rescale(pq.deg))
        e = planet.e
        W = float(planet.periastron)
        transittime = float(planet.transittime)

        time_array = time_array.to(u.day).value
        # model for each resolution element.

        if depth:
            planet_spectrum = np.array(
                [depth])  # an array as we want to perform ndim later.
        else:
            planet_spectrum = self.planet_spectrum

        models = np.zeros((len(time_array), len(planet_spectrum)))

        planet_spectrum = np.sqrt(
            planet_spectrum)  # pylc wants Rp/Rs not transit depth

        logger.debug(
            "Generating lightcurves with P={}, a={}, i={}, e={}, W={}, T14={},"
            " mean_depth={}".format(
                P, a, i, e, W, transittime, np.mean(planet_spectrum)
            ))

        for j, spec_elem in enumerate(planet_spectrum):
            models[:, j] = pylc.model(self.ldcoeffs, spec_elem, P, a, e, i, W,
                                      transittime, time_array)

        return models

    def show_lightcurve(self):
        """ Plots the white lightcurve of the planned observation

        :return:
        """

        time_array = self.exp_start_times

        lc_model = self.generate_lightcurves(time_array, self.planet.calcTransitDepth())

        if self._visit_trend:
            trend_model = self._visit_trend.scale_factors
            # have to convert weird model format to flat array
            lc_model = trend_model * lc_model.T[0]

        fig = plt.figure()
        plt.scatter(time_array, lc_model)
        plt.xlabel("Time (JD)")
        plt.ylabel("Transit Depth")
        plt.title("Normalised White Light Curve of observation")

        return time_array, lc_model

    def run_observation(self):
        """ Runs the observation by calling self._generate_exposure for each
         exposure start time
        """

        # multicore doesnt work, i suspect because it needs to pickle this entire class
        # pool = Pool(8)
        # pool.map(gen_frame, to_run)

        # filenums = range(1, len(self.exp_start_times)+1)
        #
        # # gen_exp = lambda args: self._generate_exposure(args[0], args[1])
        #
        # run_params = zip(filenums, self.exp_start_times)
        #
        # # pool = Pool(cpu_count())
        # # pool.map(gen_exp, run_params)

        self._generate_direct_image()  # to calibrate x_ref and y_ref

        num_frames = len(self.exp_start_times)
        progress = Progress(num_frames)
        self.progess = progress

        progress_line = 'Generating frames 0/{} done'.format(num_frames)
        progress.print_status_line(progress_line)
        progress.progress_line = progress_line

        for i, start_time in enumerate(self.exp_start_times):
            filenum = i + 1
            self._generate_exposure(start_time, filenum)

            progress.increment()
            progress_line = 'Generating frames {}/{} done'.format(filenum, num_frames)
            progress.print_status_line(progress_line)

            # so it can be retreived by exposure_generator
            progress.progress_line = progress_line

    def _generate_exposure(self, expstart, number):
        """ Generates the exposure at expstart, number is the filenumber of the exposure

        :param number: file number to save the exposure as
        :param expstart: JD of the start of the exposure

        :return: exposure frame
        :rtype: exposure.Exposure
        """

        index_number = number-1  # for zero indexing

        filename = '{:04d}_raw.fits'.format(number)

        exp_gen = ExposureGenerator(self.detector, self.grism, self.NSAMP,
                                    self.SAMPSEQ, self.SUBARRAY,
                                    self.planet, filename, expstart)

        _, sample_mid_points, sample_durations, read_index = \
            exp_gen._gen_scanning_sample_times(self.sample_rate)

        time_array = (sample_mid_points + expstart).to(u.day)

        star_norm_flux = self.generate_lightcurves(time_array)
        planet_depths = 1 - star_norm_flux

        # x shifts - linear shift with exposure, second exposure shifted by
        #  x_shifts, direct image and first match.

        x_ref = self._try_index(self.x_ref, index_number)
        y_ref = self._try_index(self.y_ref, index_number)
        sky_background = self._try_index(self.sky_background, index_number)

        # X-Shifts
        x_ref += self.x_shifts * index_number

        if self._visit_trend:
            scale_factor = self._visit_trend.get_scale_factor(index_number)
        else:
            scale_factor = None

        exp_frame = exp_gen.scanning_frame(
            x_ref, y_ref, self.wl, self.stellar_flux, planet_depths,
            self.scan_speed, self.sample_rate, sample_mid_points,
            sample_durations, read_index, ssv_std=self.ssv_std,
            ssv_period=self.ssv_period,
            noise_mean=self.noise_mean, noise_std=self.noise_std,
            add_flat=self.add_flat, add_dark=self.add_dark,
            scale_factor=scale_factor, sky_background=sky_background,
            cosmic_rate=self.cosmic_rate, add_gain=self.add_gain,
            add_non_linear=self.add_non_linear,
            clip_values_det_limits=self.clip_values_det_limits,
            add_final_noise_sources=self.add_final_noise_sources,
            progress_bar=self.progess, psf_approx_factor=self.psf_approx_factor,
        )

        exp_frame.generate_fits(self.outdir, filename)

        return exp_frame

    def _try_index(self, value, index):
        """ Tries to get value[index, if it cant returns value]
        :type value: int or list
        """

        try:
            return value[index]
        except TypeError:
            return value

    def _generate_direct_image(self):
        """ generate direct image (for wl calibration) - just assume happens
         1 minute before the rest for now
        """
        filename = '0000_flt.fits'

        di_start_JD = (self.exp_start_times[0] - 1 * u.min).to(u.day)
        di_exp_gen = ExposureGenerator(self.detector, self.grism, self.NSAMP,
                                       self.SAMPSEQ, self.SUBARRAY,
                                       self.planet, filename, di_start_JD)

        try:  # assume that its a list not a single value
            x_ref = self.x_ref[0]
        except TypeError:
            x_ref = self.x_ref

        try:  # assume that its a list not a single value
            y_ref = self.y_ref[0]
        except TypeError:
            y_ref = self.y_ref

        exp = di_exp_gen.direct_image(x_ref, y_ref)
        exp.generate_fits(self.outdir, '0000_flt.fits')