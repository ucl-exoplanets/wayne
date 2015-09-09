import unittest

import numpy as np
import astropy.units as u

from ...trend_generators import visit_trends
from .. import unittest_tools


class Test_BaseVisitTrend(unittest.TestCase):

    def setUp(self):
        self.visit_plan = {'exp_times': [1*u.day, 2*u.day, 3*u.day]}
        self.visit_trend = visit_trends.BaseVisitTrend(self.visit_plan, None)

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
            'exp_times': (np.array([6, 9, 12, 95, 98, 101])*u.min).to(u.day),
            'orbit_start_index': [0, 3],
        }

        visit_trend = visit_trends.HookAndLongTermRamp(
            self.visit_plan, (1, -0.0005, 1.2e-4, 400))

        unittest_tools.assertArrayAlmostEqual(
            visit_trend.scale_factors,
            [0.99988,0.99994681,0.99997525,0.9998491,0.99991591,0.99994435], 5)