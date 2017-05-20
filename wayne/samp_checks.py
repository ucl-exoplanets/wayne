# coding: utf-8
import exodata
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.analytic_functions import blackbody_lambda

from wayne import detector
from wayne import grism
from wayne import observation
from wayne import tools

plt.style.use('ggplot')

exodata = exodata.OECDatabase(
    '/Users/ryan/git/open_exoplanet_catalogue/systems')
gj1214b = exodata.planetDict['Gliese 1214 b']

source_spectra = np.loadtxt('hj_10000.dat')
source_spectra = source_spectra.T  # turn to (wl list, flux list)
source_spectra = np.array(tools.crop_spectrum(0.9, 1.8, *source_spectra))
# for some reason the unit doesnt stay with some other methods
# source_spectra = (source_spectra[0] * pq.micron, source_spectra[1]) 

depth = 0.02120791

planet = gj1214b
g141 = grism.Grism()
pixel_unit = g141.detector.pixel_unit

wl_p = source_spectra[0] * u.micron
depth_p = source_spectra[1]

wl_bb = wl_p
f_bb = blackbody_lambda(wl_bb, planet.star.T)

x_ref = 550
y_ref = 550

f_bb_tmp = f_bb * u.sr / 1e20  # temp factor to account for distance


def gen_frame((samp, psf_max)):
    exp_scan = observation.ExposureGenerator(detector.WFC3_IR(), g141,
                                             NSAMP=15, SAMPSEQ='RAPID',
                                             SUBARRAY=256, planet=gj1214b)

    exp_scan_data = exp_scan.scanning_frame(x_ref, y_ref, wl_p, f_bb_tmp,
                                            depth_p, 1 * u.pixel / u.s, samp,
                                            psf_max)

    # exp_scan_data.generate_fits('output/', 'R1000_256_RAPID_15_1ps_{}ms_{}psf.fits'.format(
    #     int(samp.to(u.ms).value), psf_max
    # ))


# to_run = [
#     (500*u.ms, 5),
#     (500*u.ms, 4),
#     (100*u.ms, 5),
#     (100*u.ms, 4),
#     (10*u.ms, 5),
#     (10*u.ms, 4),
#     (10*u.ms, 3),
#     (1*u.ms, 5),
# ]
#
# pool = Pool(8)
# pool.map(gen_frame, to_run)


gen_frame((500 * u.ms, 4))
