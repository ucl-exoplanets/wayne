import pytest

import pandas as pd
import astropy.units as u
import numpy as np
import numpy.testing

from wayne import detector


class Test_WFC3_IR:

    def test__init__(self):
        detector.WFC3_IR()

    def test_get_modes(self):
        det = detector.WFC3_IR()

        df, df2 = det._get_modes()

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (360, 4)

        assert isinstance(df2, pd.DataFrame)
        assert df2.shape == (19, 3)

    def test_get_exptime_works(self):
        det = detector.WFC3_IR()

        assert det.exptime(NSAMP=2, SAMPSEQ='RAPID', SUBARRAY=1024) == 2.932*u.s
        assert det.exptime(NSAMP=16, SAMPSEQ='RAPID', SUBARRAY=64) == 0.912*u.s
        assert det.exptime(NSAMP=8, SAMPSEQ='SPARS25', SUBARRAY=512).value - 161.302 < 0.001

    def test_exptime_return_type(self):
        # Testing that the return is a single quantity and not a single quantity in an array.
        # this results in x += exptime + overhead failing with "ValueError: non-broadcastable output"
        det = detector.WFC3_IR()

        exp_time = det.exptime(NSAMP=2, SAMPSEQ='RAPID', SUBARRAY=1024)

        x = 4. * u.s
        x += exp_time + 6. * u.s  # should run without ValueError

    def test_getexptime_raises_WFC3SimSampleModeError_if_invalid_NSAMP(self):
        det = detector.WFC3_IR()

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=17, SAMPSEQ='RAPID', SUBARRAY=1024)

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=0, SAMPSEQ='RAPID', SUBARRAY=1024)

    def test_getexptime_raises_WFC3SimSampleModeError_if_invalid_SAMPSEQ(self):
        det = detector.WFC3_IR()

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=15, SAMPSEQ='WRONG', SUBARRAY=1024)

        with pytest.raises(detector.WFC3SimSampleModeError):
            # 128 where SPARS25 not permitted
            det.exptime(NSAMP=15, SAMPSEQ='SPARS25', SUBARRAY=128)

    def test_getexptime_raises_WFC3SimSampleModeError_if_invalid_SUBARRAY(self):
        det = detector.WFC3_IR()

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=1023)

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.exptime(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=0)

    def test_get_read_times_works(self):
        det = detector.WFC3_IR()

        exptimes = det.get_read_times(NSAMP=5, SAMPSEQ='RAPID', SUBARRAY=1024).to(u.s).value
        answer = np.array([2.932, 5.865, 8.797, 11.729])
        numpy.testing.assert_array_almost_equal(exptimes, answer, 3)

        exptimes = det.get_read_times(NSAMP=3, SAMPSEQ='SPARS10', SUBARRAY=256).to(u.s).value
        answer = np.array([0.278, 7.624])
        numpy.testing.assert_array_almost_equal(exptimes, answer, 3)

    def test_get_read_times_raises_WFC3SimSampleModeError_if_invalid_NSAMP(self):
        det = detector.WFC3_IR()

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.get_read_times(NSAMP=17, SAMPSEQ='RAPID', SUBARRAY=1024)

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.get_read_times(NSAMP=0, SAMPSEQ='RAPID', SUBARRAY=1024)

    def test_get_read_times_raises_WFC3SimSampleModeError_if_invalid_SAMPSEQ(self):
        det = detector.WFC3_IR()

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.get_read_times(NSAMP=15, SAMPSEQ='WRONG', SUBARRAY=1024)

        with pytest.raises(detector.WFC3SimSampleModeError):
            # 128 with SPARS25 not permitted
            det.get_read_times(NSAMP=15, SAMPSEQ='SPARS25', SUBARRAY=128)

    def test_get_read_times_raises_WFC3SimSampleModeError_if_invalid_SUBARRAY(self):
        det = detector.WFC3_IR()

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.get_read_times(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=1023)

        with pytest.raises(detector.WFC3SimSampleModeError):
            det.get_read_times(NSAMP=15, SAMPSEQ='RAPID', SUBARRAY=0)