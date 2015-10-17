""" Functions that generate scan speed variations. These functions take input
of the sample expsoure time...

Note all functions here take the same input (to get_subsample_exposure_times) and
return the same output

"""

import numpy as np
import astropy.units as u


class SSVSine(object):

    def __init__(self, stddev=1.5, period=0.7, start_phase='rand'):
        """ Provides the scaling factors to adjust the flux by the scan speed
         variations, should be more physical i.e adjusting exposure time

        assuming sinusoidal variations

        Notes:
            Currently based on a single observation, needs more analysis and
                - Modulations
                - Based on scan speed not y_refs
                - variations in phase

        :param start_phase: phase to start the ssv or 'rand'
        :type start_phase: float or str
        """

        self.stddev = stddev
        self.period = period
        self.start_phase = start_phase

    def get_subsample_exposure_times(self, y_mid_points, sample_durations,
                                     subsample_exptime, total_exptime):
        stddev = self.stddev
        period = self.period
        start_phase = self.start_phase

        zeroed_y_mid = y_mid_points - y_mid_points[0]

        sin_func = lambda x, std, phase, mean, period: std * np.sin(
            (period * x) + phase) + mean

        if start_phase == 'rand':
            start_phase = np.random.random() * 2*np.pi

            # total exptime will change if a multiple of period doesnt fit,
            # so we need to scale the total flux by the a reference
            ssv_0 = self._flux_ssv_scaling(y_mid_points, stddev, period,
                                                  start_phase=0)
            ssv_0_mean = np.mean(ssv_0)

        ssv_scaling = sin_func(zeroed_y_mid, stddev / 100., start_phase, 1.,
                               period)

        if start_phase == 'rand':
            # do the scaling, assumes samples have same exp time - which they mostly do
            ssv_scaling *= ssv_0_mean / np.mean(ssv_scaling)

        return sample_durations * ssv_scaling


class SSVModulatedSine(object):
    # TODO this function isnt really implemented well, samples can be over /
    # under exposed due to it not also altering the sample midpoints and y_refs
    # If using use a small sample rate
    def __init__(self, amplitude=10, period=1.1, blip_proba=1):
        """creates a list of exposure times for the sub-samples with variable
         amplitude, period, phase and possible scan speed blips while keeping
         the total exposure time constant

        exptime: total exposure time of the scan
        sub_exptime: mean exposure time for the sub-samples
        amplitude: % level of variability for the sub_exptime
        period: period of variability for the sub_exptime
        blip_proba: % probability for a bump to appear in a frame
        """
        self.amplitude = amplitude
        self.period = period
        self.blip_proba = blip_proba

    def get_subsample_exposure_times(self, y_mid_points, sample_durations,
                                     read_times, sample_rate):
        """
        read_times: exposure times of the samples (seconds)
        sample_rate: mean exposure time for the sub-samples (seconds)
        """

        read_times = read_times.to(u.s).value
        sample_rate = sample_rate.to(u.s).value

        period = self.period
        amplitude = self.amplitude

        exptime = np.round(read_times[-1], 6)
        tt = np.arange(0, exptime, sample_rate)


        amp1 = np.ones_like(tt)
        amp2 = np.random.normal(0.1, 0.05) * np.sin((2 * np.pi / np.random.normal(2.0 * exptime, 0.5 * exptime)) * tt
                                                    + np.random.random() * 2 * np.pi)
        if 100.0 * np.random.random() < self.blip_proba:
            amp3 = np.random.normal(1.0, 0.1) * np.exp(-(tt - np.random.random() * exptime) ** 2 / (2 * (period / 2) ** 2))
        else:
            amp3 = 0

        final_amp = sample_rate * (amplitude / 100.0) * (amp1 + amp2 + amp3)

        per1 = np.ones_like(tt)
        per2 = np.random.normal(0.1, 0.05) * np.sin((2 * np.pi / np.random.normal(2.0 * exptime, 0.5 * exptime)) * tt
                                                    + np.random.random() * 2 * np.pi)
        final_per = period * (per1 + per2)

        final_phase = np.random.random() * 2 * np.pi
        final_sub_exptimes = np.round(sample_rate + final_amp * np.sin((2 * np.pi / final_per) * tt + final_phase), 6)

        difference = int((10 ** 6) * np.round(exptime - np.sum(final_sub_exptimes), 6))
        if difference < 0:
            for i in range(abs(difference)):
                final_sub_exptimes[np.random.randint(len(final_sub_exptimes))] -= 0.000001
        else:
            for i in range(abs(difference)):
                final_sub_exptimes[np.random.randint(len(final_sub_exptimes))] += 0.000001

        breaks = []
        for i in read_times:
            breaks.append(np.argmin(abs(np.cumsum(final_sub_exptimes) - i)))

        difference = int((10 ** 6) * np.round(read_times[0] - np.sum(final_sub_exptimes[:breaks[0] + 1]), 6))
        dis = np.int_(np.random.power(3, abs(difference)) * (breaks[0] + 1))
        if difference < 0:
            for i in dis:
                final_sub_exptimes[breaks[0] - i] -= 0.000001
                final_sub_exptimes[np.random.randint(breaks[0] + 1, len(final_sub_exptimes))] += 0.000001
        else:
            for i in dis:
                final_sub_exptimes[breaks[0] - i] += 0.000001
                final_sub_exptimes[np.random.randint(breaks[0] + 1, len(final_sub_exptimes))] -= 0.000001

        for read in range(1, len(read_times) - 1):
            difference = int((10 ** 6) * np.round(read_times[read] - np.sum(final_sub_exptimes[:breaks[read] + 1]), 6))
            if difference < 0:
                for i in range(abs(difference)):
                    final_sub_exptimes[np.random.randint(breaks[read - 1] + 1, breaks[read] + 1)] -= 0.000001
                    final_sub_exptimes[np.random.randint(breaks[read] + 1, len(final_sub_exptimes))] += 0.000001
            else:
                for i in range(abs(difference)):
                    final_sub_exptimes[np.random.randint(breaks[read - 1] + 1, breaks[read] + 1)] += 0.000001
                    final_sub_exptimes[np.random.randint(breaks[read] + 1, len(final_sub_exptimes))] -= 0.000001

        read_indexes = breaks

        return (final_sub_exptimes * u.s).to(u.ms), read_indexes