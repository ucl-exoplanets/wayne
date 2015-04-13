import unittest

import pandas as pd
import astropy.units as u

from .. import detector


# test WFC3_IR detector as the default cant do much
class Test_Detector(unittest.TestCase):

    def test__init__(self):
        detector.WFC3_IR()

    def test_get_modes(self):
        det = detector.WFC3_IR()

        df = det._get_modes()

        self.assertIsInstance(df, pd.core.frame.DataFrame)
        self.assertEqual(len(df), 300)
        self.assertEqual(len(df.columns), 4)

    def test_get_exptime_works(self):
        det = detector.WFC3_IR()

        self.assertEqual(det.exptime(NSAMP=1, SAMPSEQ='RAPID', SUBARRAY=1024), 2.932*u.s)
        self.assertEqual(det.exptime(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=64), 0.912*u.s)
        # 161.3020000002 != 161.302, almost equal doesnt like units
        self.assertAlmostEqual(det.exptime(NSAMP=8, SAMPSEQ='SPARS25', SUBARRAY=512).value, 161.302, 3)

    def test_getexptime_raises_WFC3SimSampleModeError_if_invalid_NSAMP(self):
        det = detector.WFC3_IR()

        with self.assertRaises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=16, SAMPSEQ='RAPID', SUBARRAY=1024)

        with self.assertRaises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=0, SAMPSEQ='RAPID', SUBARRAY=1024)

    def test_getexptime_raises_WFC3SimSampleModeError_if_invalid_SAMPSEQ(self):
        det = detector.WFC3_IR()

        with self.assertRaises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=15, SAMPSEQ='WRONG', SUBARRAY=1024)

        with self.assertRaises(detector.WFC3SimSampleModeError):
            # 128 with SPARS25 not permitted
            det.exptime(NSAMP=15, SAMPSEQ='SPARS25', SUBARRAY=128)

    def test_getexptime_raises_WFC3SimSampleModeError_if_invalid_SUBARRAY(self):
        det = detector.WFC3_IR()

        with self.assertRaises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=1023)

        with self.assertRaises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=0)