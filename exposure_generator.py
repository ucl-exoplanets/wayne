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
            'psf_max': None,
            'samp_rate': 0 * u.s,
            'sim_time': 0 * u.s,
            'scan_speed_var': False,
            'noise_mean': False,
            'noise_std': False,
            'add_dark': False,
        }

        self.ssv_period = 0.7

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
                       scan_speed, sample_rate, psf_max=4,
                       sample_mid_points=None, sample_durations=None,
                       read_index=None, ssv_std=False, noise_mean=False,
                       noise_std=False, add_dark=True, add_flat=True,
                       cosmic_rate=None, sky_background=1*u.count/u.s,
                       scale_factor=None, add_gain=True, add_non_linear=True,
                       clip_values_det_limits=True, add_final_noise_sources=True,
                       stellar_noise=True, spectrum_psf_smoothing=True):
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
        :param psf_max: how many pixels the psf tails span, 0.9999999999999889%
         of flux between is between -4 and 4 of
        the widest psf
        :type psf_max: int

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
            'psf_max': psf_max,
            'samp_rate': sample_rate,
            'x_ref': x_ref,
            'y_ref': y_ref,
            'scan_speed_var': ssv_std,
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
        if ssv_std:
            ssv_scaling = self._flux_ssv_scaling(s_y_refs, ssv_std)
            self.exposure.ssv_scaling = ssv_scaling  # temp to see whats going on
            self.exposure.s_y_refs = s_y_refs

        # Prep for random noise and other trends / noise sources
        read_num = 0
        read_exp_times = self.read_times.to(u.s)
        previous_read_time = 0. * u.ms

        # we want to treat the sample at the mid point state not the beginning
        # s_ denotes variables that change per sample
        pixel_array = self.detector.gen_pixel_array(self.SUBARRAY,
                                                    # misses 5 pixel border
                                                    light_sensitive=True)
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
                x_ref, s_y_ref, s_wl, s_flux, blank_frame, s_dur, psf_max,
                scale_factor, add_flat, spectrum_psf_smoothing, stellar_noise)

            pixel_array += sample_frame

            if i in read_index:  # trigger a read including final read

                if stellar_noise == 'poi_per_samp':
                    pixel_array = self.add_stellar_noise(pixel_array)

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

        # check to make sure all reads were made
        assert (len(self.exposure.reads) == self.NSAMP)

        # Final post generation corrections
        if add_non_linear:
            # we do this at the end, because the correction is based on total
            # flux and we dont want to apply the correction multiple times
            logger.info('Applying non-linearity correction to frames')
            self.exposure.apply_non_linear()

        # Scale the counts between limits, i.e. 0 to 78k for WFC3 IR
        # Note that some pixel will give negative values irl
        if clip_values_det_limits:
            self.exposure.scale_counts_between_limits()

        if add_final_noise_sources:
            self.exposure.add_final_noise_sources()

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

    def _flux_ssv_scaling(self, y_mid_points, ssv_std=1.5):
        """ Provides the scaling factors to adjust the flux by the scan speed
         variations, should be more physical i.e adjusting exposure time

        assuming sinusoidal variations

        Notes:
            Currently based on a single observation, needs more analysis and
                - Modulations
                - Based on scan speed not y_refs
                - variations in phase

        :return:
        """

        zeroed_y_mid = y_mid_points - y_mid_points[0]

        sin_func = lambda x, std, phase, mean, period: std * np.sin(
            (period * x) + phase) + mean

        ssv_scaling = sin_func(zeroed_y_mid, ssv_std / 100., np.pi, 1.,
                               self.ssv_period)

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
                           psf_max, scale_factor=None, add_flat=True,
                           spectrum_psf_smoothing=True, stellar_noise=False):
        """ Does the bulk of the work in generating the observation. Used by
         both staring and scanning modes.
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
        x_min = trace.wl_to_x(wl - delta_wl / 2.)
        x_max = trace.wl_to_x(wl + delta_wl / 2.)
        x_min_ = np.floor(x_min).astype('int16')
        x_max_ = np.floor(x_max).astype('int16')

        # effected_elements = np.floor(x_min) != np.floor(x_max)
        # print 'old num effected  = {} ({}%)'.format(np.sum(effected_elements),
        #  np.mean(effected_elements)*100)
        # self._overlap_detection(trace, x_pos, wl, psf_max)

        # Modify the flux by the grism throughput Units e / (s A)
        count_rate = self.grism.apply_throughput(wl, flux)
        count_rate = count_rate.to(u.photon / u.s / u.angstrom)

        # Scale the flux to photon counts (per pixel / per second)
        count_rate = self._flux_to_counts(count_rate, wl)
        count_rate = count_rate.to(u.photon / u.s)

        # Smooth spectrum (simulating spectral PSF) 4.5 is approximate stdev
        # remove the units first as the kernal dosent like it
        count_rate = count_rate.to(u.photon / u.s)  # sanity check
        wl = wl.to(u.micron)

        if spectrum_psf_smoothing:
            count_rate = self.grism.gaussian_smoothing(wl, count_rate.value)
            count_rate = (count_rate * u.photon / u.s).to(u.photon / u.s)

        if stellar_noise == 'poi_per_sub':
            count_rate *= 7*u.s  # scale to ~ sub sample time
            count_rate = self.add_stellar_noise(count_rate.value) * u.photon
            count_rate /= 7*u.s
            # np.savetxt('/Users/ryan/Downloads/spec_poi_per_sub.txt', np.array([wl.value, count_rate.value]).T)

        counts = (count_rate * exptime).to(u.photon)

        if stellar_noise == 'norm':
            noise = np.random.normal(0, 1)  # then scale to stddev
            noise *= np.sqrt(counts.value)
            counts += noise * u.photon
            # np.savetxt('/Users/ryan/Downloads/spec_norm_per_sub.txt', np.array([wl.value, counts.value]).T)

        counts = self.detector.apply_quantum_efficiency(wl, counts)

        # Finally, scale the lightcurve by the ramp
        if scale_factor is not None:
            counts *= scale_factor

        # the len limits are the same per trace, it is the values in pixel
        #  units each pixel occupies, as this is tilted
        # each pixel has a length slightly greater than 1
        psf_len_limits = self._get_psf_len_limits(trace, psf_max)

        # This is a 2d array of the psf limits per wl element so we can
        #  integrate the whole sample at once

        psf_limit_array = _build_2d_limits_array(psf_len_limits, self.grism,
                                                 wl, y_pos)

        # now we just need to integrate between the limits on each row of this
        #  vector. scipy.stats.norm.cdf will return
        # values for 2d vectors.
        # TODO (ryan) currently the psf wings are uneven, while we may split
        #  5.9
        #  between 0 and 10 we could change the
        # integration to go from 1.8 to 10 or 1.9 to 10.9
        binned_fluxes = _integrate_2d_limits_array(psf_limit_array,
                                                   counts.to(u.ph).value)

        # each resolution element (wl, counts_tp)
        for i in xrange(len(wl)):
            x = x_pos[i]
            y = y_pos[i]
            # When we only want whole pixels, note we go 5 pixels each way from
            #  the round down.
            # Int technically floors, but towards zero, although we shouldnt ever be negative
            x_ = int(np.floor(x))
            y_ = int(np.floor(y))

            # retrieve counts from vectorised integration
            flux_psf = binned_fluxes[i]

            # we need to convert full frame numbers into subbarry numbers for
            #  indexing the array, the wl solution uses
            # full frame numbers
            sub_scale = 507 - (self.SUBARRAY / 2)  # int div, sub should be even
            y_sub_ = y_ - sub_scale
            x_min_sub_ = x_min_ - sub_scale
            x_max_sub_ = x_max_ - sub_scale
            x_sub_ = x_ - sub_scale

            # Now we are checking if the widths overlap pixels, this is
            #  important at low R. Currently we assume the line is still
            # straight, calculate the proportion in the left and right pixels
            #  based on the y midpoint and split accordingly. This doesnt
            # account for cases where the top half may be in one column and the
            #  bottom half in another (High R)
            row_index_min = y_sub_ - psf_max
            row_index_max = y_sub_ + psf_max + 1

            if not x_min_[i] == x_max_[i]:
                # then the element is split across two columns
                # calculate proportion going column x_min_ and x_max_
                x_width = x_max[i] - x_min[i]
                # (floor(xmax) - xmin)/width = %
                propxmin = (x_max_[i] - x_min[i]) / x_width
                propxmax = 1. - propxmin

                try:
                    pixel_array[row_index_min:row_index_max, x_min_sub_[i]] += flux_psf * propxmin
                    pixel_array[row_index_min:row_index_max, x_max_sub_[i]] += flux_psf * propxmax
                except IndexError:  # spectrum is beyond the frame edge
                    pass
            else:  # all flux goes into one column
                # Note: Ideally we dont want to overwrite te detector, but have
                # a function for the detector to give us a grid. there are
                # other detector effects though so maybe wed prefer multiple
                #  detector classes or a .reset() on the class
                try:
                    pixel_array[row_index_min:row_index_max, x_sub_] += flux_psf
                except IndexError:  # spectrum is beyond the frame edge
                    pass


        if add_flat:
            flat_field = self.grism.get_flat_field(x_ref, y_ref,
                                               self.SUBARRAY)
            pixel_array *= flat_field

        return pixel_array

    def _overlap_detection(self, trace, wl, psf_max):
        """ Overlap detection see if element is split between columns. it does
         this by comparing the end points of the psf including the width of
         the bin.

        Note: this function is not used anywhere yet as the code for handling
         these cases is not yet written. The detected cases is much higher,
         especially with lower R inputs. It converges to the central width
          case at high R

        :param trace: trace line
        :type trace: grism.SpectrumTrace
        :param wl: array of wavelengths (corresponding to stellar flux and
         planet spectrum) in u.microns
        :type wl: astropy.units.quantity.Quantity
        :param psf_max: how many pixels the psf tails span, 0.9999999999999889%
         of flux between is between -4 and 4 of
        the widest psf
        :type psf_max: int

        :return: nothing yet, dont know what i need until i have a solution!
        """

        #   Note: delta_lambda inefficient, also calculated in self._flux_to_counts
        delta_wl = tools.bin_centers_to_widths(wl)
        xangle = trace.xangle()  # angle between x axis and trace line / y axis and psf line

        # we want to calculate the +- in the x position at the top and bottom of the psf, note
        x_diff = psf_max * np.tan(xangle)

        # the lower and upper x limits are then the lower wl limit - x_diff, upper + x_diff

        # need to turn wl width to x width
        delta_wl_half = delta_wl / 2.
        x_min = trace.wl_to_x(wl - delta_wl_half) - x_diff
        x_max = trace.wl_to_x(wl + delta_wl_half) + x_diff
        # x_min_ = np.floor(x_min)
        # x_max_ = np.floor(x_max)

        effected_elements = np.floor(x_min) != np.floor(x_max)
        print 'new num effected  = {} ({}%)'.format(
            np.sum(effected_elements), np.mean(effected_elements) * 100)

        x_min = trace.wl_to_x(wl) - x_diff
        x_max = trace.wl_to_x(wl) + x_diff
        effected_elements = np.floor(x_min) != np.floor(x_max)
        print 'nw2 num effected  = {} ({}%)'.format(
            np.sum(effected_elements), np.mean(effected_elements) * 100)

        # TODO (ryan) if this overlaps, give the y position of the overlap?

        # return y_values

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

        A = self.detector.telescope_area
        lam_hc = wl / (const.h * const.c) * u.photon
        delta_lambda = tools.bin_centers_to_widths(wl)

        # throughput is considered elsewhere

        counts = flux * delta_lambda
        # counts = counts.decompose()
        # final test to ensure we have eliminated all other units
        counts = counts.to(u.photon / u.s)

        return counts

    def _get_psf_len_limits(self, trace, psf_max):
        """ Obtains the limits of the psf per pixel. As the trace is slightly
         inclined the length per pixel is > 1
        so we aren't integrating for y + 0 to y+1 but (for example) y+0 to
         y+1.007 to y+2.014. This function returns
        theese limits from -psfmax to psfmax

        :param trace: trace line
        :type trace: grism.SpectrumTrace
        :param psf_max: how many pixels the psf tails span, 0.9999999999999889%
         of flux between is between -4 and 4 of
        the widest psf
        :type psf_max: int

        :return: integration limits per pixel of the psf.
        :rtype: numpy.ndarray
        """
        # The spectral trace forms our wavelength calibration
        # line isnt vertical so the length is slightly more than 1
        psf_pixel_len = trace.psf_length_per_pixel()

        # extra bit either side of the line of length 1
        psf_pixel_frac = (psf_pixel_len - 1) / 2.



        # 0 being our spectrum and then the interval either side (we will
        #  convert this using pixel_len)
        # comment example assume a psf_max of 2
        # i.e array([-2, -1,  0,  1,  2,  3])
        psf_limits = np.arange(-psf_max, psf_max + 2)
        # i.e for pixel_len = 1.2 we would have array([-2.5, -1.3, -0.1,  1.1,
        #   2.3,  3.5]) for max=3
        psf_len_limits = (psf_limits * psf_pixel_len) - psf_pixel_frac

        return psf_len_limits

    def combine_planet_stellar_spectrum(self, stellar_flux, planet_spectrum,
                                        poisson_noise=True):
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

    def add_stellar_noise(self, counts):
        """ Resamples each flux element as a poisson distribution (stellar noise)
        """

        noisey_counts = np.random.poisson(counts)

        return noisey_counts

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

def _build_2d_limits_array(psf_len_limits, grism, wl, y_pos):
    # Integrate gaussian per wl - vectorised
    # ---

    psf_std = grism.wl_to_psf_std(wl)
    psf_means = y_pos

    # len limits are at 0, we need to shift them up to floor(y_ref). We do
    #  this because we are integrating
    # over WHOLE pixels, The psf is centered on y_ref which handles intrapixel
    #  y_ref shifts. to integrate we need the (limits - mean)/std so we have
    #  (psf_len_limits + y_ - y)/std
    y_pos_norm = (np.floor(y_pos) - psf_means)

    # create a 2d array of the len limits per wl in rows i.e
    # [[-4, -3 ... 3, 4],...[100, 101, ... 105, 106]]
    psf_limits_array = np.ones((len(y_pos_norm), len(psf_len_limits)))
    psf_limits_array = psf_limits_array * psf_len_limits

    # normalise by adding y_pos_norm and dividing off the stddev
    # here we transpose y from 1xn to nx1 (need to reshape first to add dimension)
    y_pos_vector = y_pos_norm.reshape(1, len(y_pos_norm)).T
    psf_std_vector = psf_std.reshape(1, len(psf_std)).T

    psf_limits_array = (psf_limits_array + y_pos_vector) / psf_std_vector

    return psf_limits_array


def _integrate_2d_limits_array(limit_array, counts):
    """ Integrates the 2d array between limits. Must be normalised to a
     standard gaussian. return array will have 1 less column.

    :return:
    """

    cdf = scipy.stats.norm.cdf(limit_array)
    area = (np.roll(cdf, -1, axis=1) - cdf)[:,
           :-1]  # x+1 - x, [-1] is x_0 - x_n and unwanted

    binned_flux = area * counts.reshape(1, len(counts)).T

    return binned_flux