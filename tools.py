""" Contains functions useful to the program
"""


def crop_spectrum(min_wl, max_wl, wl, flux):
    """ crops the spectrum strictly between the limits provided, only works if wl is ordered

    :param min_wl:
    :param max_wl:
    :param wl: list of spectrum wl
    :type wl: numpy.ndarray
    :param flux: list of spectrum flux
    :type flux: numpy.ndarray
    :return: wl, flux (cropped)
    """

    wl_min_nearest = wl-min_wl
    # any negative values are < min_wl and are excluded by assigning them the highest value in the array
    wl_min_nearest[wl_min_nearest < 0] = wl_min_nearest.max()
    imin = wl_min_nearest.argmin()

    wl_max_nearest = wl-max_wl
    wl_max_nearest[wl_max_nearest > 0] = wl_max_nearest.min()  # any positive values are > min_wl and are excluded
    imax = wl_max_nearest.argmax() + 1  # plus 1 because we want this value included in the slice

    return wl[imin:imax], flux[imin:imax]
