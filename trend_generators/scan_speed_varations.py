""" Functions that generate scan speed variations. These functions take input
of the sample expsoure time...
"""

import numpy as np
import astropy.units as u


class SSVModulatedSine(object):

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

    def get_subsample_exposure_times(self, subsample_exptime, total_exptime):
        """ returns the exposure time per subsample

        :param sample_exptime: from the configurations, i.e 0.1s
        :param total_exptime:  total exposure time i.e 27s
        :return:
        """

        subsample_exptime = subsample_exptime.to(u.s).value
        total_exptime = total_exptime.to(u.s).value

        exptime = np.round(total_exptime, 6)
        tt = np.arange(0, exptime, subsample_exptime)

        amp1 = np.ones_like(tt)
        amp2 = np.random.normal(0.1,0.05)*np.sin((2*np.pi/np.random.normal(
            2.0*exptime,0.5*exptime))*tt+np.random.random()*2*np.pi)
        if 100.0*np.random.random() < self.blip_proba:
            amp3 =  np.random.normal(1.0,0.1) * \
                    np.exp(-(tt - np.random.random()*exptime) ** 2
                           / (2 * (self.period/2) ** 2))
        else:
            amp3 = 0

        final_amp = subsample_exptime*(self.amplitude/100.0)*(amp1+amp2+amp3)

        per1 = np.ones_like(tt)
        per2 = np.random.normal(0.1,0.05)*np.sin((2*np.pi/np.random.normal(
            2.0*exptime,0.5*exptime))*tt+np.random.random()*2*np.pi)
        final_per = self.period*(per1+per2)

        final_phase = np.random.random()*2*np.pi

        final_sub_exptimes = np.round(subsample_exptime + final_amp*np.sin(
            (2*np.pi/final_per)*tt + final_phase), 6)
        difference = np.round(exptime - np.sum(final_sub_exptimes),6)

        if difference < 0:
            for i in range(abs(int((10**6)*difference))):
                final_sub_exptimes[np.random.randint(len(final_sub_exptimes))] -= 0.000001
        else:
            for i in range(abs(int((10**6)*difference))):
                final_sub_exptimes[np.random.randint(len(final_sub_exptimes))] += 0.000001

        return (final_sub_exptimes * u.s).to(u.ms)