""" Contains functions useful to the program
"""

import numpy as np


def crop_spectrum(min_wl, max_wl, wl, flux):
    """ crops the spectrum strictly between the limits provided, only works if wl is ordered

    :param min_wl:
    :param max_wl:
    :param wl: list of spectrum wl
    :type wl: numpy.ndarray
    :param flux: list of spectrum flux
    :type flux: numpy.ndarray
    :return: wl, flux (cropped)
    :rtype: (numpy.ndarray, numpy.ndarray)
    """

    wl_min_nearest = wl-min_wl
    # any negative values are < min_wl and are excluded by assigning them the highest value in the array
    wl_min_nearest[wl_min_nearest < 0] = wl_min_nearest.max()
    imin = wl_min_nearest.argmin()

    wl_max_nearest = wl-max_wl
    wl_max_nearest[wl_max_nearest > 0] = wl_max_nearest.min()  # any positive values are > min_wl and are excluded
    imax = wl_max_nearest.argmax() + 1  # plus 1 because we want this value included in the slice

    return wl[imin:imax], flux[imin:imax]


def bin_centers_to_edges(centers):
    """ Converts bin centers to edges, handling uneven bins. Bins are assumed to
    be between the midpoint of the surrounding centers, the edges are assumed to
    extend out by the midpoint to the next/previous point.
    """

    bin_range = (centers-np.roll(centers, 1))/2.
    # Handle the start point (the roll means [0] will be centers[-1]
    bin_range[0] = bin_range[1]

    # len(edges) = len(centers)+1
    bin_edges = np.zeros(len(centers)+1)
    bin_edges[:-1] = centers - bin_range

    # now handle end point
    bin_edges[-1] = centers[-1] + bin_range[-1]

    return bin_edges


def bin_centers_to_widths(centers):
    """ Converts centers to the width per bin

    :param bin_edges:
    :return:
    """

    bin_range = (centers-np.roll(centers, 1))/2.
    # Handle the start point (the roll means [0] will be centers[-1]
    bin_range[0] = bin_range[1]

    # Now we need to to add each gap to the following gap. so we need to roll
    bin_range_roll = np.roll(bin_range, -1)
    # handle the end this time
    bin_range_roll[-1] = bin_range[-1]

    # len(edges) = len(centers)
    bin_width = bin_range + bin_range_roll

    return bin_width
