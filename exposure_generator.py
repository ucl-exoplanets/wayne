import time

import numpy as np
import scipy
from astropy import units as u
from astropy import constants as const

from wfc3simlog import logger
import tools
import exposure
from trend_generators import cosmic_rays


class ExposureGenerator(object):
    def __init__(self, detector, grism, NSAMP, SAMPSEQ, SUBARRAY, planet,
                 filename='0001_raw.fits', start_JD=0 * u.day,):
        """ Constructs exposures given a spectrum

        :param detector: detector class, i.e. WFC3_IR()
        :type detector: detector.WFC3_IR
        :param grism: grism class i.e. G141
        :type grism: grism.Grism

        :param NSAMP: number of sample up the ramp, effects exposure time (1 to 15)
        :type NSAMP: int
        :param SAMPSEQ: Sample sequence to use, effects exposure time
        ('RAPID', 'SPARS10', 'SPARS25', 'SPARS50',
        'SPARS100', 'SPARS200', 'STEP25', 'STEP50', 'STEP100', 'STEP200', 'STEP400'
        :type SAMPSEQ: str
        :param SUBARRAY: subarray to use, effects exposure time and array size.
        (1024, 512, 256, 128, 64)
        :type SUBARRAY: int

        :param planet: An ExoData type planet object your observing (holds
        observables like Rp, Rs, i, e, etc)
        :type planet: exodata.astroclasses.Planet
        :param filename: name of the generated file, given to the Exposure
        class (for fits header + saving)
        :type filename: str
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
        self.read_times = self.detector.get_read_times(NSAMP, SUBARRAY,
                                                       SAMPSEQ)

        self.exp_info = {
            # these should be generated mostly in observation class and
            # defaulted here / for staring also
            'filename': filename,
            'EXPSTART': start_JD,  # JD
            'EXPEND': start_JD + self.exptime.to(u.day),
            'EXPTIME': self.exptime.to(u.s),  # Seconds
            'SCAN': False,
            'SCAN_DIR': None,
            # 1 for down, -1 for up - replace with POSTARG calc later
            'OBSTYPE': 'SPECTROSCOPIC',  # SPECTROSCOPIC or IMAGING
            'NSAMP': self.NSAMP,
            'SAMPSEQ': self.SAMPSEQ,
            'SUBARRAY': self.SUBARRAY,
            'samp_rate': 0 * u.s,
            'sim_time': 0 * u.s,
            'scan_speed_var': False,
            'noise_mean': False,
            'noise_std': False,
            'add_dark': False,
        }

    def direct_image(self, x_ref, y_ref):
        """ This creates a direct image used to calibrate x_ref and y_ref from
         the observations

        Currently this is very basic, outputting a standard unscaled 2d
         gaussian with no filter. it will only generate a final read and
         zero read.

        :param x_ref:
        :param y_ref:

        :return:
        """

        self.exp_info.update({
            'OBSTYPE': 'IMAGING',
            'x_ref': x_ref,
            'NSAMP': 2,
            'SAMP-SEQ': 'RAPID',
            'y_ref': y_ref,

            'add_flat': False,
            'add_gain': False,
            'add_non_linear': False,
            'add_read_noise': False,
            'cosmic_rate': 0,
            'sky_background': 0*u.ct/u.s,
            'scale_factor': 1,
            'clip_values_det_limits': False,
        })

        # Exposure class which holds the result
        # TODO remove grism and add filter or something
        self.exposure = exposure.Exposure(self.detector, self.grism,
                                          self.planet, self.exp_info)

        # Zero Read
        self.exposure.add_read(
            self.detector.gen_pixel_array(self.SUBARRAY, light_sensitive=False))

        SUBARRAY = self.SUBARRAY

        # Angelos code
        y = np.arange(SUBARRAY, dtype=float)
        x = np.arange(SUBARRAY, dtype=float)
        x, y = np.meshgrid(x, y)
        x0 = x_ref - (507.0 - SUBARRAY / 2.0)
        y0 = x_ref - (507.0 - SUBARRAY / 2.0)
        # generate a 2d gaussian
        di_array = 10000.0 * np.exp(-((x0 - x) ** 2 + (y0 - y) ** 2) / 2.0)

        read_info = {
            'read_exp_time': 0 * u.s,  # TODO add real value
            'cumulative_exp_time': 0 * u.s, # TODO add real value
            'CRPIX1': -5,
        }

        self.exposure.add_read(di_array, read_info)

        return self.exposure

    def scanning_frame(self, x_ref, y_ref, wl, stellar_flux, planet_signal,
                       scan_speed, sample_rate,
                       sample_mid_points=None, sample_durations=None,
                       read_index=None, ssv_std=False, ssv_period=False,
                       noise_mean=False,
                       noise_std=False, add_dark=True, add_flat=True,
                       cosmic_rate=None, sky_background=1*u.count/u.s,
                       scale_factor=None, add_gain=True, add_non_linear=True,
                       clip_values_det_limits=True, add_read_noise=True,
                       progress_bar=None, psf_approx_factor=False):
        """ Generates a spatially scanned frame.

        Note also that the stellar flux and planet signal MUST be binned the
        same, this can be done with wfc3sim.tools.rebin_spec

        Note, i need to seperate this into a user friendly version and a
        version to use with observation as i am already
        seeing clashes (aka sample times generation).

        :param x_ref: pixel in x axis the reference should be located
        :type x_ref: int
        :param y_ref: pixel in y axis the reference should be located
        :type y_ref: int
        :param wl: array of wavelengths (corresponding to stellar flux and
        planet spectrum) in u.microns
        :type wl: astropy.units.quantity.Quantity
                :param stellar_flux: array of stellar flux in units of
                erg/(angstrom * s * cm^2)
        :type stellar_flux: astropy.units.quantity.Quantity
        :param planet_spectrum: array of the transit depth for the planet spectrum
        :type planet_spectrum: numpy.ndarray

        :param scan_speed: rate of scan in pixels/second
        :type scan_speed: astropy.units.quantity.Quantity
        :param sample_rate: How often to sample the exposure (time units)
        :type sample_rate: astropy.units.quantity.Quantity

        :param sample_mid_point: mid point of each sample, None to auto generate
        :param sample_durations: duration of each sample, None to auto generate
        :param read_index: list of the sample indexes that are the final sample
         for that read, None for auto
        :type read_index: list

        :param scan_speed_var: The % of the std(flux_per_pixel) the scan speed
         variations cause. Basic implementation.
        :type scan_speed_var: float

        :param noise_mean: mean noise (per second, per pixel)
        :type noise_mean: float
        :param noise_std: standard deviation of the noise (per second, per pixel)
        :type noise_std: float

        :param scale_factor: Scales the whole frame (per read) by a factor,
          used for visit long trends

        :return: array with the exposure
        """

        start_time = time.clock()

        scan_speed = scan_speed.to(u.pixel / u.ms)
        sample_rate = sample_rate.to(u.ms)

        # user friendly, else defined by observation class which uses these
        #  values for lightcurve generation
        if sample_mid_points is None and sample_durations is None and read_index is None:
            _, sample_mid_points, sample_durations, read_index = \
                self._gen_scanning_sample_times(sample_rate)

        self.exp_info.update({
            'SCAN': True,
            'SCAN_DIR': 1,
            'samp_rate': sample_rate,
            'x_ref': x_ref,
            'y_ref': y_ref,
            'scan_speed_var': ssv_std,
            'scan_speed_period': ssv_period,
            'noise_mean': noise_mean,
            'noise_std': noise_std,
            'add_dark': add_dark,

            'add_flat': add_flat,
            'add_gain': add_gain,
            'add_non_linear': add_non_linear,
            'cosmic_rate': cosmic_rate,
            'sky_background': sky_background,
            'scale_factor': scale_factor,
            'clip_values_det_limits': clip_values_det_limits,
        })

        if planet_signal.ndim == 1:  # depth does not vary with time during exposure
            s_flux = self.combine_planet_stellar_spectrum(stellar_flux,
                                                          planet_signal)
            # TODO handling cropping elsewhere to avoid doing it all the time
            s_wl, s_flux = tools.crop_spectrum(self.grism.wl_limits[0],
                                               self.grism.wl_limits[1], wl,
                                               s_flux)

        # Exposure class which holds the result
        self.exposure = exposure.Exposure(self.detector, self.grism,
                                          self.planet, self.exp_info)

        # Zero Read
        self.exposure.add_read(self.detector.gen_pixel_array(self.SUBARRAY,
                                                             light_sensitive=False))

        # y_ref per sample
        s_y_refs = self._gen_sample_yref(y_ref, sample_mid_points, scan_speed)

        # Scan Speed Variations (flux modulations)
        if ssv_std and ssv_period:
            ssv_scaling = self._flux_ssv_scaling(s_y_refs, ssv_std, ssv_period)

        # Prep for random noise and other trends / noise sources
        read_num = 0
        read_exp_times = self.read_times.to(u.s)
        previous_read_time = 0. * u.ms

        # we want to treat the sample at the mid point state not the beginning
        # s_ denotes variables that change per sample
        pixel_array = self.detector.gen_pixel_array(self.SUBARRAY,
                                                    # misses 5 pixel border
                                                    light_sensitive=True)

        num_samples = len(sample_mid_points)
        if progress_bar is not None:
            progress_line = progress_bar.progress_line + \
                            ' (samp {}/{})'.format(0, num_samples)
            progress_bar.print_status_line(progress_line)

        for i, s_mid in enumerate(sample_mid_points):

            s_y_ref = s_y_refs[i]
            s_dur = sample_durations[i]

            if ssv_std:
                s_dur *= ssv_scaling[i]

            if planet_signal.ndim == 1:
                pass  # handled above but leaving to point out this needs cleaning up
            else:
                s_flux = self.combine_planet_stellar_spectrum(stellar_flux, planet_signal[i])

                # TODO (ryan) handling cropping elsewhere to avoid doing it
                #  all the time, crop flux + depth together
                s_wl, s_flux = tools.crop_spectrum(self.grism.wl_limits[0],
                                                   self.grism.wl_limits[-1],
                                                   wl, s_flux)

            # generate sample frame
            blank_frame = np.zeros_like(pixel_array)
            sample_frame = self._gen_staring_frame(
                x_ref, s_y_ref, s_wl, s_flux, blank_frame, s_dur,
                scale_factor, add_flat, psf_approx_factor)

            pixel_array += sample_frame

            if i in read_index:  # trigger a read including final read

                # TODO (ryan) export all this reduction stuff to own function

                # TODO (ryan) check scaling i.e. DN vs e
                cumulative_exp_time = read_exp_times[read_num]
                read_exp_time = (
                    read_exp_times[read_num] - previous_read_time).to(
                        u.s).value

                array_size = pixel_array.shape[0]  # shouldnt this be subarray?

                if noise_mean and noise_std:
                    noise_array = self._gen_noise(noise_mean * read_exp_time,
                                                  noise_std * read_exp_time)

                    # Note noise is added to the full frame, in reality it will
                    # probably be centered around the scan!
                    # possibility of noise -> mean due to CLT with more reads
                    pixel_array += noise_array

                # TODO (ryan) bad pixels

                if sky_background:
                    bg_count_sec = sky_background.to(u.count/u.s).value
                    master_sky = self.grism.get_master_sky(array_size)
                    bg_count = bg_count_sec * read_exp_time

                    master_sky *= bg_count


                    pixel_array += master_sky

                if add_gain:
                    gain_file = self.grism.get_gain(self.SUBARRAY)
                    pixel_array *= gain_file

                # TODO allow manual setting of cosmic gen
                if cosmic_rate is not None:
                    cosmic_gen = \
                        cosmic_rays.MinMaxPossionCosmicGenerator(
                            cosmic_rate)

                    cosmic_array = cosmic_gen.cosmic_frame(read_exp_time,
                                                           array_size)
                    pixel_array += cosmic_array

                pixel_array_full = self.detector.add_bias_pixels(pixel_array)

                if add_dark:
                    pixel_array_full = self.detector.add_dark_current(
                        pixel_array_full, self.NSAMP, self.SUBARRAY,
                        self.SAMPSEQ)


                read_info = {
                    'cumulative_exp_time': cumulative_exp_time,
                    # not a unit earlier, but should be, req for fits saving
                    'read_exp_time': read_exp_time*u.s,
                    'CRPIX1': 0,
                }

                try:
                    cumulative_pixel_array += pixel_array_full
                except NameError:  # first read
                    cumulative_pixel_array = pixel_array_full

                # need to copy or all reads will be the same
                self.exposure.add_read(cumulative_pixel_array.copy(), read_info)

                previous_read_time = read_exp_times[read_num]
                read_num += 1

                # Now we want to start again with a fresh array
                pixel_array = self.detector.gen_pixel_array(self.SUBARRAY,
                                            light_sensitive=True)

            if progress_bar is not None:
                progress_line = progress_bar.progress_line + \
                                ' (samp {}/{})'.format(i+1, num_samples)
                progress_bar.print_status_line(progress_line)

        # check to make sure all reads were made
        assert (len(self.exposure.reads) == self.NSAMP)

        # Final post generation corrections
        if add_non_linear:
            # we do this at the end, because the correction is based on total
            # flux and we dont want to apply the correction multiple times
            # logger.info('Applying non-linearity correction to frames')
            self.exposure.apply_non_linear()

        # Scale the counts between limits, i.e. 0 to 78k for WFC3 IR
        # Note that some pixel will give negative values irl
        if clip_values_det_limits:
            self.exposure.scale_counts_between_limits()

        if add_read_noise:
            self.exposure.add_read_noise()

        end_time = time.clock()

        self.exp_info['sim_time'] = (end_time - start_time) * u.s

        return self.exposure

    def _gen_sample_yref(self, y_ref, mid_points, scan_speed):
        """ Generates y_ref for each sample

        was used to add scan speed variations, now adding as a % of flux

        :param y_ref:
        :param midpoints:
        :return:
        """

        s_y_refs = y_ref + (mid_points * scan_speed).to(u.pixel).value

        return s_y_refs

    def _flux_ssv_scaling(self, y_mid_points, ssv_std=1.5, ssv_period=0.7,
                          start_phase='rand'):
        """ Provides the scaling factors to adjust the flux by the scan speed
         variations, should be more physical i.e adjusting exposure time

        assuming sinusoidal variations

        Notes:
            Currently based on a single observation, needs more analysis and
                - Modulations
                - Based on scan speed not y_refs
                - variations in phase

        :param start_phase: pahse to start the ssv or 'rand'
        :type start_phase: float or str

        :return:
        """

        zeroed_y_mid = y_mid_points - y_mid_points[0]

        sin_func = lambda x, std, phase, mean, period: std * np.sin(
            (period * x) + phase) + mean

        if start_phase == 'rand':
            start_phase = np.random.random() * 2*np.pi

            # total exptime will change if a multiple of period doesnt fit,
            # so we need to scale the total flux by the a reference
            ssv_0 = self._flux_ssv_scaling(y_mid_points, ssv_std, ssv_period,
                                                  start_phase=0)
            ssv_0_mean = np.mean(ssv_0)

        ssv_scaling = sin_func(zeroed_y_mid, ssv_std / 100., start_phase, 1.,
                               ssv_period)

        if start_phase == 'rand':
            # do the scaling, assumes samples have same exp time - which they mostly do
            ssv_scaling *= ssv_0_mean / np.mean(ssv_scaling)

        return ssv_scaling

    def _gen_scanning_sample_times(self, sample_rate):
        """ Generates several times to do with samples. Including samples up
        the ramp. exposures are sampled at the sample rate until a read
        where the duration is reduced to the remainder. Sampling continues
        at  the rate after this read for the next one.

        Note this can result in a situation where the final read is extremely
         short compared to the sample time

        In future durations could be changed here to account for uneven scans.
         This function is separated so the observation class can use it.

        :param sample_rate: how often to sample (in time units)
        :type sample_rate: astropy.units.quantity.Quantity)

        :return:
        """

        sample_rate = sample_rate.to(u.ms)
        read_times = self.read_times.to(u.ms)

        read_index = []  # at what sample number to perform a read (after generation)
        # counting reads to create the read index, -1 as first sample is index 0 not 1
        i = -1

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
        sample_mid_points = sample_starts + (sample_durations / 2)

        sample_starts *= u.ms
        sample_durations *= u.ms
        sample_mid_points *= u.ms

        return sample_starts, sample_mid_points, sample_durations, read_index

    def staring_frame(self, x_ref, y_ref, wl, stellar_flux, planet_signal,
                      psf_max=4):
        """ Constructs a staring mode frame, given a source position and
        spectrum scaling

        NOTE THAT THIS FUNCTION IS NOT AS UP TO DATE AS SCANNING AND PROBABLY
        WILL NOT BE UNTIL THE DEVELOPMENT PERIOD SLOWS DOWN

        :param x_ref: pixel in x axis the reference should be located
        :type x_ref: int
        :param y_ref: pixel in y axis the reference should be located
        :type y_ref: int
        :param wl: array of wavelengths (corresponding to stellar flux and
         planet spectrum) in u.microns
        :type wl: astropy.units.quantity.Quantity
        :param stellar_flux: array of stellar flux in units of
         erg/(angstrom * s * cm^2)
        :type stellar_flux: astropy.units.quantity.Quantity
        :param planet_spectrum: array of the transit depth for the planet spectrum
        :type planet_spectrum: numpy.ndarray

        :param psf_max: how many pixels the psf tails span, 0.9999999999999889%
         of flux between is between -4 and 4 of
        the widest psf
        :type psf_max: int

        :return: array with the exposure
        """

        raise NotImplementedError('Sorry, needs drastically updating')

        self.exp_info.update({
            'SCAN': False,
            'psf_max': psf_max,
            'x_ref': x_ref,
            'y_ref': y_ref,
        })

        # Exposure class which holds the result
        self.exposure = exposure.Exposure(self.detector, self.grism,
                                          self.planet, self.exp_info)

        flux = self.combine_planet_stellar_spectrum(stellar_flux,
                                                    planet_signal)
        wl, flux = tools.crop_spectrum(self.grism.wl_limits[0],
                                       self.grism.wl_limits[-1], wl, flux)

        # Zero Read
        self.exposure.add_read(
            self.detector.gen_pixel_array(self.SUBARRAY, light_sensitive=False))

        # Generate first sample up the ramp
        first_read_time = self.read_times[0]
        first_read_array = self.detector.gen_pixel_array(self.SUBARRAY,
                                                         light_sensitive=True)
        first_read_array = self._gen_staring_frame(x_ref, y_ref, wl, flux,
                                                   first_read_array,
                                                   first_read_time, psf_max)
        self.exposure.add_read(self.detector.add_bias_pixels(first_read_array))

        # generate subsequent reads by scaling the first read, starting with the second (1)
        for read_time in self.read_times[1:]:
            read_array = first_read_array * (read_time / first_read_time)
            self.exposure.add_read(self.detector.add_bias_pixels(read_array))

        return self.exposure

    def _gen_staring_frame(self, x_ref, y_ref, wl, flux, pixel_array, exptime,
                           scale_factor=None, add_flat=True, psf_approx_factor=False):
        """ Does the bulk of the work in generating the observation. Used by
         both staring and scanning modes.
        :return:
        """

        wl = wl.to(u.micron)

        # Wavelength calibration, mapping to detector x/y pos
        trace = self.grism.get_trace(x_ref, y_ref)
        x_pos = trace.wl_to_x(wl)
        y_pos = trace.wl_to_y(wl)

        psf_std = self.grism.wl_to_psf_std(wl)

        # Modify the flux by the grism throughput Units e / (s A)
        count_rate = self.grism.apply_throughput(wl, flux)
        count_rate = count_rate.to(u.photon / u.s / u.angstrom)

        # Scale the flux to photon counts (per pixel / per second)
        count_rate = self._flux_to_counts(count_rate, wl)
        count_rate = count_rate.to(u.photon / u.s)

        # Smooth spectrum (simulating spectral PSF) 4.5 is approximate stdev
        # remove the units first as the kernal dosent like it
        count_rate = count_rate.to(u.photon / u.s)  # sanity check

        counts = (count_rate * exptime).to(u.photon)
        counts = self.detector.apply_quantum_efficiency(wl, counts)

        # Finally, scale the lightcurve by the ramp
        if scale_factor is not None:
            counts *= scale_factor

        counts = counts.to(u.photon).value  # final unit check

        counts = np.random.poisson(counts)  # poisson noise

        # each resolution element (wl, counts_tp)
        for i in xrange(len(wl)):
            wl_x = x_pos[i]
            wl_y = y_pos[i]
            wl_counts = counts[i]
            wl_psf_std = psf_std[i]

            # we need to convert full frame numbers into subbarry numbers for
            # indexing the array, the wl solution uses full frame numbers
            sub_scale = 507 - (self.SUBARRAY / 2)  # int div, sub should be even
            wl_x_sub = wl_x - sub_scale
            wl_y_sub = wl_y - sub_scale

            pixel_array = _psf_distribution(wl_counts, wl_x_sub, wl_y_sub,
                                            wl_psf_std, pixel_array, psf_approx_factor)

        if add_flat:
            flat_field = self.grism.get_flat_field(x_ref, y_ref,
                                               self.SUBARRAY)
            pixel_array *= flat_field

        return pixel_array

    def _flux_to_counts(self, flux, wl):
        """ Converts flux to photons by scaling to the to the detector pixel
         size, energy to photons

        We want the counts per second per resolution element given by

        $C = F_\lambda A \frac{\lambda}{hc} Q_\lambda T_\lambda \, \delta\lambda$

        Where

        * A is the area of an unobstructed 2.4m telescope ($45,239 \, cm^2$ for HST)
        * $\delta\lambda$ range of lambda being considered
        * $F_\lambda$ is the flux from the source in $\frac{erg}{cm^2 \, s \,
         \overset{\circ}{A}}$
        * The factor $\lambda/hc$ converts ergs to photons
        * $Q_\lambda T_\lambda$ is the fractional throughput, Q being
         instrument sensitivity and T the filter transmission
        * Spectral dispersion in $\overset{\circ}{A}$ / pixel

        :param wl: array of wavelengths (corresponding to stellar flux and
         planet spectrum) in u.microns
        :type wl: astropy.units.quantity.Quantity
        :param flux: array of stellar flux in units of erg/(angstrom * s * cm^2)
        :type flux: astropy.units.quantity.Quantity

        :return: photon counts (same size as flux) ph/s
        :rtype: astropy.units.quantity.Quantity
        """

        # A = self.detector.telescope_area
        # lam_hc = wl / (const.h * const.c) * u.photon
        delta_lambda = tools.bin_centers_to_widths(wl)

        # throughput is considered elsewhere
        counts = flux * delta_lambda
        # final test to ensure we have eliminated all other units
        counts = counts.to(u.photon / u.s)

        return counts

    def combine_planet_stellar_spectrum(self, stellar_flux, planet_spectrum):
        """ combines the stellar and planetary spectrum

        combined_flux = F_\star * (1-transit_depth)

        Varying depth is handled by generating lightcurves for each element,
         the planet spectrum after all is just the transit depth at maximum
         for each element

        :param stellar_flux: array of stellar flux in units of
        erg/(angstrom * s * cm^2)
        :type stellar_flux: astropy.units.quantity.Quantity
        :param planet_spectrum: array of the transit depth for the planet spectrum
        :type planet_spectrum: numpy.ndarray

        :return: combined flux
        :rtype: astropy.units.quantity.Quantity
        """

        combined_flux = stellar_flux * (1. - planet_spectrum)

        return combined_flux

    def _gen_noise(self, mean, std):
        """ Generates noise from a normal distribution at the mean and std
         given at the size of a subbarray-10
        :param mean:
        :param std:
        :return:
        """

        dim = self.SUBARRAY

        if dim == 1024:
            dim = 1014

        noise = np.random.normal(mean, std, (dim, dim))

        return noise


def _psf_distribution(counts, x_pos, y_pos, psf_std, pixel_array, approx_factor=False):
    """ Distributes electrons across the 2D guassian PSF by acting like a
     glorified electron thrower. Coordinates are generated by 2 guassian
     distributions
    """

    if not counts:  # zero counts
        return pixel_array

    single_count = 1

    if approx_factor:
        counts = np.int_(np.rint(counts/approx_factor))
        single_count = approx_factor

    xx = np.int_(np.random.normal(x_pos, psf_std, counts))
    yy = np.int_(np.random.normal(y_pos, psf_std, counts))

    for i in xrange(counts):
        try:
            pixel_array[yy[i]][xx[i]] += single_count
        except IndexError:  # off the detector
            pass

    return pixel_array