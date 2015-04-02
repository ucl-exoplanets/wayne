""" Observation combines a spectrum, grism + detector combo and other observables to construct an image
"""

import numpy as np
from astropy import units as u
from astropy import constants as const

import detector
import grism
import tools

# TODO Observation class - controls everything, multiple exposures


class Exposure(object):
    """ Constructs exposures given a spectrum
    """

    def __init__(self, detector, grism):
        """

        :param detector: detector class, i.e. WFC3_IR()
        :type detector: detector.Detector
        :param grism: grism class i.e. G141
        :type grism: grism.Grism
        :return:
        """

        self.detector = detector
        self.grism = grism

    def scanning_frame(self, x_ref, y_ref, wl, stellar_flux, planet_signal, scan_speed, exptime, sampletime):
        """

        :param x_ref: star image x position on frame
        :param y_ref: star image y position on frame
        :param wl: wavelength of stellar flux AND planet signal (must be sampled identically)
        :param stellar flux:
        :param planet_signal: (units of transit depth)
        :param scan_speed: (u.pixel/u.ms)
        :param exptime: total exposure time /sec
        :param sampletime: how often to generate a staring frame, shorter times improves accuracy at the expense of
         runtime

        :return: array with the exposure
        """

        scan_speed = scan_speed.to(u.pixel/u.ms)
        exptime = exptime.to(u.ms)
        sampletime = sampletime.to(u.ms)
        # convert times to miliseconds and then call value to remove units so arrange works
        # s_ denotes variables that change per sample
        for s_time in np.arange(0, exptime.value, sampletime.value) * u.ms:

            s_y_ref = y_ref + (s_time * scan_speed).to(u.pixel).value

            # TODO we can vary the transit depth by generating a lightcurve from the signal at our sample times

            # generate the staring frame at our sample time, writes directly to detector frame.
            self.staring_frame(x_ref, s_y_ref, wl, stellar_flux, planet_signal, exptime)

        return self.detector.pixel_array

    def staring_frame(self, x_ref, y_ref, wl, stellar_flux, planet_signal, exptime, psf_max=5):
        """ constructs a staring mode frame, given a source position and spectrum scaling

        :param x_ref: star image x position on frame
        :param y_ref: star image y position on frame
        :param wl: wavelength of stellar flux AND planet signal (must be sampled identically)
        :param stellar flux:
        :param planet_signal: (units of transit depth)
        :param exptime:
        :param psf_max: how many pixels either side of y_ref we want to go. Note if y_ref = 5.9 and psf_max=5 it would
        be from 0 to 10. 0.9999999999999889% of flux between is between -4 and 4 of widest psf. Going one higher than
        required is sensible for cases like above (5.9, 0 to 10) where one half is better represented

        :return: array with the exposure
        """

        flux = self.combine_planet_stellar_spectrum(stellar_flux, planet_signal)
        # TODO improve the inefficient cropping, we are doing it once per sample is scanning
        wl, flux = tools.crop_spectrum(self.grism.wl_limits[0], self.grism.wl_limits[-1], wl, flux)

        # Wavelength calibration, mapping to detector x/y pos
        trace = self.grism.get_trace(x_ref, y_ref)
        x_pos = trace.wl_to_x(wl)
        y_pos = trace.wl_to_y(wl)

        # Overlap detection see if element is split between columns

        # Note: delta_lambda inefficient, also calculated in self._flux_to_counts
        delta_wl = tools.bin_centers_to_widths(wl)
        # need to turn from wl width to x width
        x_min = trace.wl_to_x(wl-delta_wl/2.)
        x_max = trace.wl_to_x(wl+delta_wl/2.)
        x_min_ = np.floor(x_min)
        x_max_ = np.floor(x_max)

        # Scale the flux to photon counts (per pixel / per second)
        count_rate = self._flux_to_counts(flux, wl)

        # TODO scale flux by stellar distance / require it already scaled

        counts = (count_rate * exptime).to(u.photon)

        # Modify the counts by the grism throughput
        counts_tp = self.grism.throughput(wl, counts)

        # TODO QE scaling

        # the len limits are the same per trace, it is the values in pixel units each pixel occupies, as this is tilted
        # each pixel has a length slightly greater than 1
        psf_len_limits =  self._get_psf_len_limits(trace, psf_max)

        spectrum = np.array([wl, counts_tp]).T  # [(wl_1, counts_1),...,(wl_n, counts_n)}

        # each resolution element
        for i, (wl_i, count_i) in enumerate(spectrum):
            x = x_pos[i]
            y = y_pos[i]
            # When we only want whole pixels, note we go 5 pixels each way from the round down.
            x_ = int(np.floor(x))  # Int technically floors, but towards zero although we shouldnt ever be negative
            y_ = int(np.floor(y))


            psf = self.grism.flux_to_psf(wl_i, count_i, y)

            # len limits are at 0, we need to shift them up to floor(y_ref). We do this because we are integrating
            # over WHOLE pixels, The psf is centered on y_ref which handles intrapixel y_ref shifts.
            psf_y_lim = psf_len_limits + y_

            # these are the count values integrated on our detector grid
            flux_psf = []
            for j in xrange(psf_max*2+1):
                # TODO currently the psf wings are uneven, while we may split 5.9 between 0 and 10 we could change the
                # integration to go from 1.8 to 10 or 1.9 to 10.9
                val = psf.integrate(psf_y_lim[j], psf_y_lim[j+1])

                flux_psf.append(val)

            # Now we are checking if the widths overlap pixels, this is important at low R. Currently we assume the line
            # is still straight, calculate the proportion in the left and right pixels based on the y midpoint and
            # split accordingly. This doesnt account for cases where the top half may be in one column and the bottom
            #  half in another (High R)
            if not x_min_[i] == x_max_[i]:  # then the element is split across two columns
                # calculate proportion going column x_min_ and x_max_
                x_width = x_max[i] - x_min[i]
                propxmin = (x_max_[i] - x_min[i])/x_width  # (floor(xmax) - xmin)/width = %
                propxmax = 1.-propxmin

                self.detector.pixel_array[y_-psf_max:y_+psf_max+1, int(x_min_[i])] += flux_psf * propxmin
                self.detector.pixel_array[y_-psf_max:y_+psf_max+1, int(x_max_[i])] += flux_psf * propxmax
            else:  # all flux goes into one column
                # Note: Ideally we dont want to overwrite te detector, but have a function for the detector to give
                # us a grid. there are other detector effects though so maybe wed prefer multiple detector classes
                # or a .reset() on the class
                self.detector.pixel_array[y_-psf_max:y_+psf_max+1, x_] += flux_psf

        return self.detector.pixel_array

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

        flux *= u.sr  # remove the solid angle dependence given by astropy.blackbody, need to include it later to
                      # account for distance to the star


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
        """

        combined_flux = stellar * (1. - planet)

        return combined_flux