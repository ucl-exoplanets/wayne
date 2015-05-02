""" Observation combines a spectrum, grism + detector combo and other observables to construct an image
"""

import time
# from multiprocessing import Pool, cpu_count

import numpy as np
import quantities as pq
from astropy import units as u
from astropy import constants as const
import pylightcurve.fcmodel as pylc
import matplotlib.pylab as plt

from progress import Progress

from wfc3simlog import logger
import grism
import tools
import exposure


class Observation(object):
    """ Builds a full observation, of separate orbits
    """

    def __init__(self, planet, start_JD, num_orbits, detector, grism, NSAMP, SAMPSEQ, SUBARRAY, wl, stellar_flux,
                 planet_spectrum, sample_rate, x_ref, y_ref, scan_speed, psf_max=4, outdir=''):
        #  initialise all parameters here for now. There could be many options entered through a
        #  Parameter file but this could be done with an interface.
        #  mode handles exp time (NSAMP, SAMPSEQ, SUBARRAY)

        # My current thought is a detector instance should be given per frame, the exposure is then stored in the
        # detector class which can also handle reads, flats, subbarays, and perhaps even translating the coordinates
        # from full frame to etc. This was the point of the exposure frame, but the main reason for that is output and
        # it is detector specific! so perhaps theese should be merged

        logger.info("Initialising Observation: startJD={}, num_orbits={}, detector={}, grism={}, "
                    "NSAMP={}, SAMPSEQ={}, SUBARRAY={}, sample_rate={}, x_ref={}, y_ref={}, scan_speed={},"
                    "psf_max={}, outdir={}".format(start_JD, num_orbits, detector, grism, NSAMP, SAMPSEQ,
                                                   SUBARRAY, sample_rate, x_ref, y_ref, scan_speed, psf_max, outdir))

        self.detector = detector
        self.grism = grism
        self.scanning = True

        self.num_orbits = num_orbits
        self.NSAMP = NSAMP
        self.SAMPSEQ = SAMPSEQ
        self.SUBARRAY = SUBARRAY

        self.sample_rate = sample_rate
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.scan_speed = scan_speed
        self.psf_max = psf_max
        self.outdir = outdir

        self.planet = planet
        self.start_JD = start_JD  # in days
        self.wl = wl
        self.stellar_flux = stellar_flux
        self.planet_spectrum = planet_spectrum

        star = self.planet.star
        self.ldcoeffs = pylc.ldcoeff(star.Z, float(star.T), star.calcLogg(), 'I')  # option to give limb darkening
        logger.info("Looked up Limb Darkening coeffs of {}".format(self.ldcoeffs))

        self.visit_plan = visit_planner(self.detector, self.NSAMP, self.SAMPSEQ, self.SUBARRAY, self.num_orbits,
                                        exp_overhead=3*u.min)  # TEMP! to make observations sparser
        self.exp_start_times = self.visit_plan['exp_times'].to(u.day) + self.start_JD

        self.exptime = self.visit_plan['exptime']
        logger.info("Each exposure will have a expsoure time of {}".format(self.exptime))

    def generate_lightcurves(self, time_array, depth=False):

        # TODO quick check if out of transit, in that case ones!

        # TODO should this be in generate exposure?

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
            planet_spectrum = np.array([depth])  # an array as we want to perform ndim later.
        else:
            planet_spectrum = self.planet_spectrum

        models = np.zeros((len(time_array), len(planet_spectrum)))

        logger.debug("Generating lightcurves with P={}, a={}, i={}, e={}, W={}, T14={}, mean_depth={}".format(
            P, a, i, e, W, transittime, np.mean(planet_spectrum)
        ))

        for j, spec_elem in enumerate(planet_spectrum):
            models[:, j] = pylc.model(self.ldcoeffs, spec_elem, P, a, e, i, W, transittime, time_array)

        return models

    def show_lightcurve(self):
        """ Shows the white lightcurve of the planned observation
        :return:
        """

        time_array = self.exp_start_times

        lc_model = self.generate_lightcurves(time_array, self.planet.calcTransitDepth())

        plt.figure()
        plt.scatter(time_array, lc_model)
        plt.xlabel("Time (JD)")
        plt.ylabel("Transit Depth")
        plt.title("Normalised White Light Curve of observation")

    def run_observation(self):

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

        num_frames = len(self.exp_start_times)
        progress = Progress(num_frames)

        for i, start_time in enumerate(self.exp_start_times):
            filenum = i+1
            self._generate_exposure(filenum, start_time)

            progress.increment()
            progress.print_status_line('Generating frames {}/{} done'.format(filenum, num_frames))

    def _generate_exposure(self, number, expstart):
        """ Generates an exposure, used in a loop
        :return:
        """

        filename = '{:04d}_raw.fits'.format(number)

        exp_gen = ExposureGenerator(self.detector, self.grism, self.NSAMP, self.SAMPSEQ, self.SUBARRAY,
                                    self.planet, filename, expstart)

        _, sample_mid_points, sample_durations, read_index = exp_gen._gen_scanning_sample_times(self.sample_rate)

        time_array = (sample_mid_points + expstart).to(u.day)

        star_norm_flux = self.generate_lightcurves(time_array)
        planet_depths = 1 - star_norm_flux

        exp_frame = exp_gen.scanning_frame(self.x_ref, self.y_ref, self.wl, self.stellar_flux, planet_depths,
                                           self.scan_speed, self.sample_rate, self.psf_max, sample_mid_points,
                                           sample_durations, read_index)

        exp_frame.generate_fits(self.outdir, filename)

        return exp_frame


class ExposureGenerator(object):
    """ Constructs exposures given a spectrum
    """

    def __init__(self, detector, grism, NSAMP, SAMPSEQ, SUBARRAY, planet, filename='0001_raw.fits',
                 start_JD=0*u.day):
        """

        :param detector: detector class, i.e. WFC3_IR()
        :type detector: detector.WFC3_IR
        :param grism: grism class i.e. G141
        :type grism: grism.Grism

        :param start_JD: start JD of the fits file, used in the fits saving
        :return:
        """

        self.detector = detector
        self.grism = grism
        self.planet = planet

        # Maybe these should be set in the detector?
        self.NSAMP = NSAMP
        self.SAMPSEQ = SAMPSEQ
        self.SUBARRAY = SUBARRAY

        # total exptime
        self.exptime = self.detector.exptime(NSAMP, SUBARRAY, SAMPSEQ)

        # samples up the ramp
        self.read_times = self.detector.get_read_times(NSAMP, SUBARRAY, SAMPSEQ)

        self.exp_info = {
            # these should be generated mostly in observation class and defaulted here / for staring also
            'filename': filename,
            'EXPSTART': start_JD,  # JD
            'EXPEND': start_JD + self.exptime.to(u.day),
            'EXPTIME': self.exptime.to(u.s),  # Seconds
            'SCAN': False,
            'SCAN_DIR': None,  # 1 for down, -1 for up - replace with POSTARG calc later
            'OBSTYPE': 'SPECTROSCOPIC',  # SPECTROSCOPIC or IMAGING
            'NSAMP': self.NSAMP,
            'SAMPSEQ': self.SAMPSEQ,
            'SUBARRAY': self.SUBARRAY,
            'psf_max': None,
            'samp_time': 0*u.s,
            'sim_time': 0*u.s
        }

    def scanning_frame(self, x_ref, y_ref, wl, stellar_flux, planet_signal, scan_speed, sample_rate, psf_max=4,
                       sample_mid_points=None, sample_durations=None, read_index=None):
        """

        Note, i need to seperate this into a user friendly version and a version to use with observation as i am already
        seeing clashes (aka sample times generation)

        :param x_ref: star image x position on frame
        :param y_ref: star image y position on frame
        :param wl: wavelength of stellar flux AND planet signal (must be sampled identically)
        :param stellar flux:
        :param planet_signal: (units of transit depth)
        :param scan_speed: (u.pixel/u.ms)
        :param sample_rate: how often to generate a staring frame, shorter times improves accuracy at the expense of
         runtime

        :param read_index: list of the sample indexes that are the final sample for that read

        :return: array with the exposure
        """

        start_time = time.clock()

        scan_speed = scan_speed.to(u.pixel/u.ms)
        sample_rate = sample_rate.to(u.ms)

        # user friendly, else defined by observation class which uses theese values for lightcurve generation
        if sample_mid_points is None and sample_durations is None and read_index is None:
            _, sample_mid_points, sample_durations, read_index = self._gen_scanning_sample_times(sample_rate)

        self.exp_info.update({
            'SCAN': True,
            'SCAN_DIR': 1,
            'psf_max': psf_max,
            'samp_rate': sample_rate,
            'x_ref': x_ref,
            'y_ref': y_ref,
        })

        if planet_signal.ndim == 1:  # depth does not vary with time during exposure
            s_flux = self.combine_planet_stellar_spectrum(stellar_flux, planet_signal)
            # TODO handling cropping elsewhere to avoid doing it all the time, crop flux + depth together
            s_wl, s_flux = tools.crop_spectrum(self.grism.wl_limits[0], self.grism.wl_limits[-1], wl, s_flux)

        # Exposure class which holds the result
        self.exposure = exposure.Exposure(self.detector, self.grism, self.planet, self.exp_info)

        # Zero Read
        self.exposure.add_read(self.detector.gen_pixel_array(light_sensitive=False))

        # we want to treat the sample at the mid point state not the beginning
        # s_ denotes variables that change per sample
        pixel_array = self.detector.gen_pixel_array(light_sensitive=True)  # misses 5 pixel border
        for i, s_mid in enumerate(sample_mid_points):
            s_y_ref = y_ref + (s_mid * scan_speed).to(u.pixel).value
            s_dur = sample_durations[i]

            if planet_signal.ndim == 1:
                pass  # handled above but leaving to point out this needs cleaning up
            else:
                s_flux = self.combine_planet_stellar_spectrum(stellar_flux, planet_signal[i])
                # TODO handling cropping elsewhere to avoid doing it all the time, crop flux + depth together
                s_wl, s_flux = tools.crop_spectrum(self.grism.wl_limits[0], self.grism.wl_limits[-1], wl, s_flux)

            # generate sample frame
            pixel_array = self._gen_staring_frame(x_ref, s_y_ref, s_wl, s_flux, pixel_array, s_dur, psf_max)

            if i in read_index:  # trigger a read including final read
                self.exposure.add_read(self.detector.add_bias_pixels(pixel_array))

        assert(len(self.exposure.reads) == self.NSAMP + 1)  # check to make sure all reads were made

        end_time = time.clock()

        self.exp_info['sim_time'] = (end_time - start_time) * u.s

        return self.exposure

    def _gen_scanning_sample_times(self, sample_rate):
        """ Generates several times to do with samples. Including samples up the ramp. exposures are sampled at the
        sample rate until a read where the duration is reduced to the remainder. Sampling continues at the rate after
        this read for the next one.

        Note this can result in a situation where the final read is extremely short compared to the sample time

        In future durations could be changed here to account for uneven scans. This function is separated so the
        observation class can use it.


        :param sample_rate: how often to sample (in time units)
        :type sample_rate: astropy.uints.core.UnitBase)

        :return:
        """

        sample_rate = sample_rate.to(u.ms)
        read_times = self.read_times.to(u.ms)

        read_index = []  # at what sample number to perform a read (after generation)
        i = -1  # counting reads to create the read index, -1 as first sample is index 0 not 1

        sample_starts = []
        previous_read = 0.
        for read_time in read_times:
            read_time = read_time.value
            starts = np.arange(previous_read, read_time, sample_rate.value)
            sample_starts.append(starts)

            i += len(starts)
            read_index.append(i)
            previous_read = read_time

        sample_starts = np.concatenate(sample_starts)

        _ends = np.roll(sample_starts, -1)
        _ends[-1] = read_times[-1].value

        sample_durations = _ends - sample_starts
        sample_mid_points = sample_starts + (sample_durations/2)

        sample_starts *= u.ms
        sample_durations *= u.ms
        sample_mid_points *= u.ms

        return sample_starts, sample_mid_points, sample_durations, read_index

    def staring_frame(self, x_ref, y_ref, wl, stellar_flux, planet_signal, psf_max=4):
        """ constructs a staring mode frame, given a source position and spectrum scaling

        :param x_ref: star image x position on frame
        :param y_ref: star image y position on frame
        :param wl: wavelength of stellar flux AND planet signal (must be sampled identically)
        :param stellar flux:
        :param planet_signal: (units of transit depth)
        :param psf_max: how many pixels either side of y_ref we want to go. Note if y_ref = 5.9 and psf_max=5 it would
        be from 0 to 10. 0.9999999999999889% of flux between is between -4 and 4 of widest psf. Going one higher than
        required is sensible for cases like above (5.9, 0 to 10) where one half is better represented

        :return: array with the exposure
        """

        self.exp_info.update({
            'SCAN': False,
            'psf_max': psf_max,
            'x_ref': x_ref,
            'y_ref': y_ref,
        })

        # Exposure class which holds the result
        self.exposure = exposure.Exposure(self.detector, self.grism, self.planet, self.exp_info)

        flux = self.combine_planet_stellar_spectrum(stellar_flux, planet_signal)
        wl, flux = tools.crop_spectrum(self.grism.wl_limits[0], self.grism.wl_limits[-1], wl, flux)

        # Zero Read
        self.exposure.add_read(self.detector.gen_pixel_array(light_sensitive=False))

        # Generate first sample up the ramp
        first_read_time = self.read_times[0]
        first_read_array = self.detector.gen_pixel_array(light_sensitive=True)
        first_read_array = self._gen_staring_frame(x_ref, y_ref, wl, flux, first_read_array, first_read_time, psf_max)
        self.exposure.add_read(self.detector.add_bias_pixels(first_read_array))

        # generate subsequent reads by scaling the first read, starting with the second (1)
        for read_time in self.read_times[1:]:
            read_array = first_read_array * (read_time/first_read_time)
            self.exposure.add_read(self.detector.add_bias_pixels(read_array))

        return self.exposure

    def _gen_staring_frame(self, x_ref, y_ref, wl, flux, pixel_array, exptime, psf_max):
        """ Does the bulk of the work in generating the observation. Used by both staring and scanning modes.
        :return:
        """

        # Wavelength calibration, mapping to detector x/y pos
        trace = self.grism.get_trace(x_ref, y_ref)
        x_pos = trace.wl_to_x(wl)
        y_pos = trace.wl_to_y(wl)

        # Overlap detection see if element is split between columns
        #   Note: delta_lambda inefficient, also calculated in self._flux_to_counts
        delta_wl = tools.bin_centers_to_widths(wl)
        #   need to turn wl width to x width
        x_min = trace.wl_to_x(wl-delta_wl/2.)
        x_max = trace.wl_to_x(wl+delta_wl/2.)
        x_min_ = np.floor(x_min)
        x_max_ = np.floor(x_max)

        # effected_elements = np.floor(x_min) != np.floor(x_max)
        # print 'old num effected  = {} ({}%)'.format(np.sum(effected_elements), np.mean(effected_elements)*100)
        # self._overlap_detection(trace, x_pos, wl, psf_max)

        # Scale the flux to photon counts (per pixel / per second)
        count_rate = self._flux_to_counts(flux, wl)

        # TODO scale flux by stellar distance / require it already scaled

        counts = (count_rate * exptime).to(u.photon)

        # Modify the counts by the grism throughput
        counts_tp = self.grism.throughput(wl, counts)

        # TODO QE scaling, done in throughput?

        # the len limits are the same per trace, it is the values in pixel units each pixel occupies, as this is tilted
        # each pixel has a length slightly greater than 1
        psf_len_limits = self._get_psf_len_limits(trace, psf_max)

        counts_tp = counts_tp.to(u.ph).value  # remove the unit now we have "counts" so flux_to_psf works
        # each resolution element (wl, counts_tp)
        for i in xrange(len(wl)):
            wl_i = wl[i]
            count_i = counts_tp[i]

            x = x_pos[i]
            y = y_pos[i]
            # When we only want whole pixels, note we go 5 pixels each way from the round down.
            x_ = int(np.floor(x))  # Int technically floors, but towards zero although we shouldnt ever be negative
            y_ = int(np.floor(y))

            psf = self.grism.flux_to_psf(wl_i, count_i, y)

            # len limits are at 0, we need to shift them up to floor(y_ref). We do this because we are integrating
            # over WHOLE pixels, The psf is centered on y_ref which handles intrapixel y_ref shifts.
            psf_y_lim = psf_len_limits + y_

            # TODO currently the psf wings are uneven, while we may split 5.9 between 0 and 10 we could change the
            # integration to go from 1.8 to 10 or 1.9 to 10.9
            flux_psf = psf.integrate(psf_y_lim)

            # Now we are checking if the widths overlap pixels, this is important at low R. Currently we assume the line
            # is still straight, calculate the proportion in the left and right pixels based on the y midpoint and
            # split accordingly. This doesnt account for cases where the top half may be in one column and the bottom
            #  half in another (High R)
            if not x_min_[i] == x_max_[i]:  # then the element is split across two columns
                # calculate proportion going column x_min_ and x_max_
                x_width = x_max[i] - x_min[i]
                propxmin = (x_max_[i] - x_min[i])/x_width  # (floor(xmax) - xmin)/width = %
                propxmax = 1.-propxmin

                pixel_array[y_-psf_max:y_+psf_max+1, int(x_min_[i])] += flux_psf * propxmin
                pixel_array[y_-psf_max:y_+psf_max+1, int(x_max_[i])] += flux_psf * propxmax
            else:  # all flux goes into one column
                # Note: Ideally we dont want to overwrite te detector, but have a function for the detector to give
                # us a grid. there are other detector effects though so maybe wed prefer multiple detector classes
                # or a .reset() on the class
                pixel_array[y_-psf_max:y_+psf_max+1, x_] += flux_psf

        return pixel_array

    def _overlap_detection(self, trace, x_pos, wl, psf_max):
        """ Overlap detection see if element is split between columns
        :return:
        """

        #   Note: delta_lambda inefficient, also calculated in self._flux_to_counts
        delta_wl = tools.bin_centers_to_widths(wl)
        xangle = trace.xangle()  # angle between x axis and trace line / y axis and psf line

        # we want to calculate the +- in the x position at the top and bottom of the psf, note
        x_diff = psf_max * np.tan(xangle)

        # the lower and upper x limits are then the lower wl limit - x_diff, upper + x_diff

        # need to turn wl width to x width
        delta_wl_half = delta_wl/2.
        x_min = trace.wl_to_x(wl-delta_wl_half) - x_diff
        x_max = trace.wl_to_x(wl+delta_wl_half) + x_diff
        # x_min_ = np.floor(x_min)
        # x_max_ = np.floor(x_max)

        effected_elements = np.floor(x_min) != np.floor(x_max)
        print 'new num effected  = {} ({}%)'.format(np.sum(effected_elements), np.mean(effected_elements)*100)

        x_min = trace.wl_to_x(wl) - x_diff
        x_max = trace.wl_to_x(wl) + x_diff
        effected_elements = np.floor(x_min) != np.floor(x_max)
        print 'nw2 num effected  = {} ({}%)'.format(np.sum(effected_elements), np.mean(effected_elements)*100)

        # TODO if this overlaps, give the y position of the overlap?

        # return y_values

    def _flux_to_counts(self, flux, wl):
        """ Converts flux to photons by scaling to the to the detector pixel size, energy to photons

        We want the counts per second per resolution element given by

        $C = F_\lambda A \frac{\lambda}{hc} Q_\lambda T_\lambda \, \delta\lambda$

        Where

        * A is the area of an unobstructed 2.4m telescope ($45,239 \, cm^2$ for HST)
        * $\delta\lambda$ range of lambda being considered
        * $F_\lambda$ is the flux from the source in $\frac{erg}{cm^2 \, s \, \overset{\circ}{A}}$
        * The factor $\lambda/hc$ converts ergs to photons
        * $Q_\lambda T_\lambda$ is the fractional throughput, Q being instrument sensitivity and T the filter transmission
        * Spectral dispersion in $\overset{\circ}{A}$ / pixel

        :param flux:
        :param wl:
        :return:
        """

        A = self.detector.telescope_area
        lam_hc = wl/(const.h * const.c) * u.photon
        delta_lambda = tools.bin_centers_to_widths(wl)

        # throughput is considered elsewhere

        counts = flux * A * delta_lambda * lam_hc
        counts = counts.decompose()
        counts = counts.to(u.photon / u.s)  # final test to ensure we have eliminated all other units

        return counts

    def _get_psf_len_limits(self, trace, psf_max):
        # The spectral trace forms our wavelength calibration
        psf_pixel_len = trace.psf_length_per_pixel()  # line isnt vertical so the length is slightly more than 1
        psf_pixel_frac = (psf_pixel_len - 1)/2.  # extra bit either side of the line of length 1

        # 0 being our spectrum and then the interval either side (we will convert this using pixel_len)
        #comment example assume a psf_max of 2
        psf_limits = np.arange(-psf_max, psf_max+2)  # i.e array([-2, -1,  0,  1,  2,  3])
        # i.e for pixel_len = 1.2 we would have array([-2.5, -1.3, -0.1,  1.1,  2.3,  3.5]) for max=3
        psf_len_limits = (psf_limits*psf_pixel_len) - psf_pixel_frac

        return psf_len_limits

    def combine_planet_stellar_spectrum(self, stellar, planet):
        """ combines the stellar and planetary spectrum

        varying depth is handled by generating lightcurves for each element, the planet spectrum afterall is just the
        transit depth at maximum for each element

        :param stellar:
        :param planet: units of transit depth

        :return:
        :rtype: np.ndarray
        """

        combined_flux = stellar * (1. - planet)

        return combined_flux


def visit_planner(detector, NSAMP, SAMPSEQ, SUBARRAY, num_orbits=3, time_per_orbit=54*u.min,
                  hst_period=90*u.min, exp_overhead=1*u.min):
    """ Returns the start time of each exposure in minutes starting at 0. Useful for estimating buffer dumps etc.

    Note, this is mainly used in testing, will probably be coupled with the lightcurve and added to observation class.
    In doing so anything detector sepecific should go into the detector classes

    :param detector:
    :param NSAMP:
    :param SAMPSEQ:
    :param SUBARRAY:
    :param orbits:
    :return:
    """

    # note this may need to be done in a slow loop as we have differing overheads, and buffer calcs
    # and other things like changing modes and scan up/down to consider
    # this is slower but not prohibitively so since this isnt run often! The complexity here suggests this should
    # be moved to a class and broken down into smaller calcs

    exptime = detector.exptime(NSAMP, SUBARRAY, SAMPSEQ)
    exp_per_dump = detector.num_exp_per_buffer(NSAMP, SUBARRAY)

    # The time to dump an n-sample, full-frame exposure is approximately 39 + 19 x (n + 1) seconds. Subarrays may also
    # be used to reduce the overhead of serial buffer dumps. ** Instruemtn handbook 10.3
    time_buffer_dump = 5.8 * u.min  # IH 23 pg 209

    # temp defined in input to make tests sparser and account for lack of manouvers
    # exp_overhead = 1*u.min  # IH 23 pg 209 - should be mostly read times - seems long, 1024 only?


    # TODO spacecraft manuvers IR 23 pg 206 for scanning
    exp_times = []
    for orbit_n in xrange(num_orbits):
        if orbit_n == 0:
            guide_star_aq = 6*u.min
        else:
            guide_star_aq = 5*u.min

        start_time = hst_period * orbit_n

        visit_time = start_time + guide_star_aq
        visit_end_time = start_time+time_per_orbit

        # TODO wl calibration exp
        exp_n = 0  # For buffer dumps - with mixed exp types we should really track headers and size
        while visit_time < visit_end_time:

            # you cant convert a list of quantities to an array so we have to either know the length to preset one or
            # use floats in a list and convert after.
            exp_times.append(visit_time.to(u.min).value)  # start of exposure
            visit_time += (exptime + exp_overhead)

            exp_n += 1
            if exp_n > exp_per_dump:
                visit_time += time_buffer_dump
                exp_n = 0


    returnDict = {
        'exp_times': np.array(exp_times)*u.min,  # start_times?
        'NSAMP': NSAMP,
        'SAMPSEQ': SAMPSEQ,
        'SUBARRAY': SUBARRAY,
        'num_exp': len(exp_times),
        'exptime': exptime,
        'num_orbits': num_orbits,
        'exp_overhead': exp_overhead,
        'time_per_orbit': time_per_orbit,
        'hst_period': hst_period,
    }

    return returnDict