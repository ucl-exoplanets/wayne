from __future__ import division
import unittest

import numpy as np
import astropy.units as u

from wayne.trend_generators import visit_trends


class ExampleVisitTrend(visit_trends.BaseVisitTrend):
    def _gen_scaling_factors(self, visit_plan, args, **kwargs):
        return np.ones(len(visit_plan['exp_start_times']))


class Test_BaseVisitTrend(unittest.TestCase):

    def setUp(self):
        self.visit_plan = {'exp_start_times': [1*u.day, 2*u.day, 3*u.day]}
        self.visit_trend = ExampleVisitTrend(self.visit_plan, None)

    def test__gen_scaling_factors(self):

        scale_factors = self.visit_trend._gen_scaling_factors(
            self.visit_plan, None)

        self.assertTrue(np.all(scale_factors == np.ones(3)))

    def test__get_scale_factor(self):

        self.visit_trend.scale_factors = np.arange(5)

        self.assertEqual(self.visit_trend.get_scale_factor(0), 0)
        self.assertEqual(self.visit_trend.get_scale_factor(3), 3)


class Test_HookAndLongTermRamp(unittest.TestCase):

    def test__gen_scaling_factors(self):
        self.visit_plan = {
            'exp_start_times': (
                np.array([6, 9, 12, 95, 98, 101])*u.min).to(u.day),
            'orbit_start_index': [0, 3],
        }

        visit_trend = visit_trends.HookAndLongTermRamp(
            self.visit_plan, (0.005, 0.0011, 400, 9/60/24))

        np.testing.assert_array_almost_equal(
            visit_trend.scale_factors,
            [0.99891,  0.99952,  0.99978,  0.9986 ,  0.99921,  0.99947],
            decimal=5
        )