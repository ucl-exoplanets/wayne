""" Handles visit long trends (scaling factors) applied to the observation. The
classic cases are the `hook' and long term ramp
"""

import numpy as np


class BaseVisitTrend(object):
    """ Visit trends take input the visit planner output and generate a
     scaling factor that will be multiplied per exposure.

     They must implement the method

     _gen_scaling_factors

     which outputs a list of scaling factors, one per exposure
    """

    def __init__(self, visit_plan, coeffs=None):

        self.visit_plan = visit_plan
        self.coeffs = coeffs

        self.scale_factors = self._gen_scaling_factors(visit_plan, coeffs)

    def _gen_scaling_factors(self, visit_plan, coeffs):

        return np.ones(len(visit_plan['exp_start_times']))


    def get_scale_factor(self, exp_num):
        """ Returns the scale factor for the exposure number i
        :param exp_num:
        :return:
        """

        return self.scale_factors[exp_num]


class HookAndLongTermRamp(BaseVisitTrend):

    def ramp_model(self, t, t_v, t_0, a1, b1, b2):
        """ Combined hook and long term ramp model
        :param t: time_array
        :param a1: linear ramp gradient
        :param b1: exponential hook coeff1
        :param b2: exponential hook coeff2
        :param t_v: start time of visit
        :param t_0: array of orbit start times (per exposure)

        :return: ramp_model
        """

        t = np.array(t)  # wipes units if any
        t_v = np.array(t_v)

        ramp = (1-a1*(t-t_v))*(1-b1*np.exp(-b2*(t-t_0)))
        return ramp

    def _gen_scaling_factors(self, visit_plan, coeffs):
        t = visit_plan['exp_start_times']
        t_v = t[0]
        t_0 = gen_orbit_start_times_per_exp(t, visit_plan['orbit_start_index'])

        ramp = self.ramp_model(t, t_v, t_0, *coeffs)

        return ramp

    # i have implemented the trend classes, need to test them and then put them
    # into the generator and config


def gen_orbit_start_times_per_exp(time_array, obs_start_index):
    """Generates t0, the time of an orbit for each orbit so it can vectorised

    i.e for each element time_array there will be a matching element in t_0 giving the
    orbit start time.
    """
    obs_index = obs_start_index[:]
    obs_index.append(len(time_array))
    t_0 = np.zeros(len(time_array))

    for i in xrange(len(obs_index)-1):
        t_0[obs_index[i]:obs_index[i+1]] = time_array[obs_start_index[i]]

    return t_0