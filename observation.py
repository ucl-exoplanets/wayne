""" Observation combines a spectrum, grism + detector combo and other observables to construct an image
"""

import numpy as np

import detector
import grism

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

    def staring_frame(self, x_ref, y_ref, spectrum, scaling):
        """ constructs a staring mode frame, given a source position and spectrum scaling

        :param x_ref: star image x position on frame
        :param y_ref: star image y position on frame
        :param scaling: Scaling factor to convert transit-depth to counts, probably to be replaced later
        :param spectrum: (wl_list, flux_list) a combined planet-star spectrum in ($\mu m$, ???)

        :return: array with the exposure
        """

        wl_list, flux_list = spectrum

        # Wavelength calibration, mapping to detector x/y pos
        trace = self.grism.get_trace(x_ref, y_ref)
        x_pos = trace.wl_to_x(wl_list)
        y_pos = trace.wl_to_y(wl_list)

        # TODO scale spectrum by exposure time into counts

        # Scale the flux to ?? units
        flux_list_s = flux_list * scaling

        # Modify the flux by the grism throughput
        flux_list_tp = self.grism.throughput(wl_list, flux_list_s)

        spectrum = np.array([wl_list, flux_list_tp]).T  # [(wl_1, flux_1),...,(wl_n, flux_n)}

        # how many pixels either side of y_ref we want to go. Note if y_ref = 5.9 it would be from
        # 0 to 10,  4 contains 0.9999999999999889% of flux between -4 and 4 of widest psf so we go one higher
        # to account for fringe cases (i.e. 5.1 vs 5.9)
        psf_max = 5

        # The spectral trace forms our wavelength calibration
        trace = self.grism.get_trace(x_ref, y_ref)
        psf_pixel_len = trace.psf_length_per_pixel()  # line isnt vertical so the length is slightly more than 1
        psf_pixel_frac = (psf_pixel_len - 1)/2.  # extra bit either side of the line of length 1

        # 0 being our spectrum and then the interval either side (we will convert this using pixel_len)
        #comment example assume a psf_max of 2
        psf_limits = np.arange(-psf_max, psf_max+2)  # i.e array([-2, -1,  0,  1,  2,  3])
        # i.e for pixel_len = 1.2 we would have array([-2.5, -1.3, -0.1,  1.1,  2.3,  3.5]) for max=3
        psf_len_limits = (psf_limits*psf_pixel_len) - psf_pixel_frac

        self._grid = []
        # each resolution element
        for i, (wl, flux) in enumerate(spectrum):
            x = x_pos[i]
            y = y_pos[i]
            x_ = np.floor(x)
            y_ = np.floor(y)

            psf = self.grism.flux_to_psf(wl, flux, y)
            psf_y_lim = psf_len_limits + y_  # now we have the limits we want to integrate over

            # these are the count values integrated on our detector grid
            flux_psf = []
            for j in xrange(psf_max*2+1):
                val = psf.integrate(psf_y_lim[j], psf_y_lim[j+1])
                if np.isnan(val) or np.isinf(val):
                    val = 0.
                flux_psf.append(val)

            self._grid.append(flux_psf) # debug
            # Ideally we dont want to overwrite te detector, but have a function for the detector to give
            # us a grid. there are other detector effects though so maybe wed prefer multiple detector classes
            # or a .reset() on the class
            self.detector.pixel_array[y_-psf_max:y_+psf_max+1, x_] += flux_psf

        return self.detector.pixel_array