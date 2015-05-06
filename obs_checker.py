import numpy as np
import matplotlib.pyplot as plt
from astropy.analytic_functions import blackbody_lambda
from astropy import units as u
import exodata
import quantities as pq  # exodata still uses this

import observation
import detector
import grism
import tools


plt.style.use('ggplot')

exodb = exodata.OECDatabase('/Users/ryan/git/open_exoplanet_catalogue/systems')
planet = exodb.planetDict['WASP-18 b']

source_spectra = np.loadtxt('hj_1000.dat')
source_spectra = source_spectra.T  # turn to (wl list, flux list)
source_spectra = np.array(tools.crop_spectrum(0.9, 1.8, *source_spectra))

g141 = grism.Grism()
det = detector.WFC3_IR()

wl_p = source_spectra[0] * u.micron
depth_p = source_spectra[1] 

wl_bb = wl_p
f_bb = blackbody_lambda(wl_bb, planet.star.T)
f_bb_tmp = f_bb * u.sr / 1e20  # temp factor to account for distance

x_ref = 450
y_ref = 350
NSAMP = 15
SAMPSEQ = 'SPARS10'
SUBARRAY = 512
start_JD = float((planet.transittime - 115*pq.min).rescale(pq.day)) * u.day  # convert pq to u
num_orbits = 2
sample_rate = 0.1*u.s
scan_speed = 180*u.pixel/(111.8*u.s)
outdir = '/Users/ryan/Dropbox/phd/wfc3sim/visit2'
psf_max = 4


# obs_proof = exp_scan = observation.ExposureGenerator(det, g141, NSAMP, SAMPSEQ, SUBARRAY, planet, '0001_proof.fits')
# exp_proof = exp_scan.scanning_frame(x_ref, y_ref, wl_p, f_bb_tmp, depth_p, scan_speed, sample_rate, psf_max)
# exp_proof.generate_fits(outdir, '0001_proof.fits')

obs = observation.Observation(planet, start_JD, num_orbits, det, g141, NSAMP, SAMPSEQ, SUBARRAY, wl_p, f_bb_tmp,
                              depth_p, sample_rate, x_ref, y_ref, scan_speed, psf_max, outdir, scan_speed_var=0.6)
print('exptime per frame = ', obs.exptime)
obs.show_lightcurve()
# plt.show()

obs.run_observation()