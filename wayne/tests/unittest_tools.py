"""
Tools for unit testing
"""

import numpy as np


def assertArrayAlmostEqual(arr1, arr2, places):
    """ numpy.testing.assert_array_almost_equal() is not working in some cases, basically saying 3.456 != 3.456 to 3dp,
    this is a quick reimplementation that works

    :param arr1:
    :param arr2:
    :param places:
    :return:
    """

    accuracy = float('.' + '0'*places + '1')

    res = arr1 - arr2
    comparison = res < accuracy

    if not np.all(comparison):
        percent_mismatch = (1-comparison.mean())*100
        raise AssertionError("{}% mismatch \n [0] {} \n [1] {}".format(percent_mismatch, arr1, arr2))