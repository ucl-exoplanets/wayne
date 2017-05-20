""" Handles visit long trends (scaling factors) applied to the observation. The
classic cases are the `hook' and long term ramp
"""

import abc

import numpy as np


class BaseVisitTrend(object):
    """ Visit trends take input the visit planner output and generate a
    scaling factor that will be multiplied per exposure.

    They must implement the method `_gen_scaling_factors` which outputs
    a list of scaling factors, one per exposure
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, visit_plan, coeffs=None):
        self.visit_plan = visit_plan
        self.coeffs = coeffs

        self.scale_factors = self._gen_scaling_factors(visit_plan, coeffs)

    @abc.abstractmethod
    def _gen_scaling_factors(self, visit_plan, coeffs):
        pass

    def get_scale_factor(self, exp_num):
        """ Returns the scale factor for the exposure number `exp_num`."""
        return self.scale_factors[exp_num]


class HookAndLongTermRamp(BaseVisitTrend):
    def _gen_scaling_factors(self, visit_plan, coeffs):
        t = visit_plan['exp_start_times']
        t_0 = gen_orbit_start_times_per_exp(t, visit_plan['orbit_start_index'])

        ramp = self.ramp_model(t, t_0, *coeffs)
        return ramp

    @staticmethod
    def ramp_model(t, t_0, a1, b1, b2, to):
        """ Combined hook and long term ramp model
        :param t: time_array
        :param t_0: array of orbit start times (per exposure)
        :param a1: linear ramp gradient
        :param b1: exponential hook coeff1
        :param b2: exponential hook coeff2

        :return: ramp_model
        """

        t = np.array(t)  # wipes units if any
        ramp = (1 - a1 * (t - to)) * (1 - b1 * np.exp(-b2 * (t - t_0)))
        return ramp


def gen_orbit_start_times_per_exp(time_array, obs_start_index):
    """Generates t0, the time of an orbit for each orbit so it can vectorised

    i.e for each element time_array there will be a matching element in t_0 giving the
    orbit start time.
    """
    obs_index = obs_start_index[:]
    obs_index.append(len(time_array))
    t_0 = np.zeros(len(time_array))

    for i in xrange(len(obs_index) - 1):
        t_0[obs_index[i]:obs_index[i + 1]] = time_array[obs_start_index[i]]

    return t_0
