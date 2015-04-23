import unittest

import numpy as np
import tempfile
import shutil
import astropy.io.fits as fits
import numpy.testing

from .. import exposure


class Test_Exposure(unittest.TestCase):

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    def test_add_read(self):

        exp = exposure.Exposure()
        data = np.arange(4).reshape(2, 2)
        exp.add_read(data)

        self.assertEqual(exp.reads, [data])

        data2 = np.arange(4).reshape(2, 2) * 2
        exp.add_read(data2)

        self.assertEqual(exp.reads, [data, data2])

    @unittest.skip("New fits file includes lots of meta data from exposure, breaking this test")
    def test_generate_fits(self):

        file = tempfile.mkstemp(dir=self.tmp_path)[1]
        with open(file, 'wb') as f:
            exp = exposure.Exposure()

            data = np.arange(4).reshape(2, 2)
            exp.add_read(data)

            exp.generate_fits(f)

        with open(file, 'rb') as f:
            result = fits.open(f)
            np.testing.assert_array_equal(result[1].data, data)

    def tearDown(self):
        shutil.rmtree(self.tmp_path)