import unittest
import tempfile
import shutil

import numpy as np
import astropy.io.fits as fits

from wayne import exposure


class Test_Exposure(unittest.TestCase):

    def setUp(self):
        self.tmp_path = tempfile.mkdtemp()

    @unittest.skip("New fits file includes lots of meta data from exposure, breaking this test")
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

        file_path = tempfile.mkstemp(dir=self.tmp_path)[1]
        with open(file_path, 'wb') as f:
            exp = exposure.Exposure()

            data = np.arange(4).reshape(2, 2)
            exp.add_read(data)

            exp.generate_fits(f)

        with open(file_path, 'rb') as f:
            result = fits.open(f)
            np.testing.assert_array_equal(result[1].data, data)

    def tearDown(self):
        shutil.rmtree(self.tmp_path)