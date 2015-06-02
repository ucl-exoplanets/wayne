""" This script allows you to run a visit from a parameter file. Currently parameter files are in python format, see
example_par.py.
"""

import argparse
import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from astropy.analytic_functions import blackbody_lambda
from astropy import units as u
import exodata
import quantities as pq  # exodata still uses this
import yaml

import observation
import detector
import grism
import tools
import params


parser = argparse.ArgumentParser(description='Simulate a visit using WFC3Sim and a parameter file')
parser.add_argument('-p', required=True, dest='parfile', type=str,
                   help='location of a WFC3Sim parameter file')

args = parser.parse_args()

with open(args.parfile, 'r') as ymlfile:
    cfg = yaml.load(ymlfile)


plt.style.use('ggplot')

outdir = cfg['general']['outdir']

if not os.path.exists(outdir):
    os.mkdir(outdir)

shutil.copy2(args.parfile, os.path.join(outdir, os.path.basename(args.parfile)))  # copy parfile to output

exodb = exodata.OECDatabase(cfg['general']['oec_location'])

seed = cfg['general']['seed']
np.random.seed(seed)
params.seed = seed  # tell params what the seed is now we've change it

planet = exodb.planetDict[cfg['target']['name']]

source_spectra = np.loadtxt(cfg['target']['spectrum_file'])
source_spectra = source_spectra.T  # turn to (wl list, flux list)
source_spectra = np.array(tools.crop_spectrum(0.9, 1.8, *source_spectra))

# if, else in future
g141 = grism.Grism()
det = detector.WFC3_IR()

wl_p = source_spectra[0] * u.micron
depth_p = source_spectra[1]

wl_bb = wl_p
f_bb = blackbody_lambda(wl_bb, planet.star.T)
f_bb_tmp = f_bb * u.sr / cfg['target']['flux_scale']

x_ref = cfg['observation']['x_ref']
y_ref = cfg['observation']['y_ref']
NSAMP = cfg['observation']['NSAMP']
SAMPSEQ = cfg['observation']['SAMPSEQ']
SUBARRAY = cfg['observation']['SUBARRAY']

start_JD = float((planet.transittime - 115*pq.min).rescale(pq.day)) * u.day  # convert pq to u
num_orbits = cfg['observation']['num_orbits']
sample_rate = cfg['observation']['sample_rate'] * u.ms
scan_speed = cfg['observation']['scan_speed'] * (u.pixel/u.s)
psf_max = cfg['observation']['psf_max']

ssv_std = cfg['observation']['ssv_std']
x_shifts = cfg['observation']['x_shifts']

noise_mean = cfg['observation']['noise_mean']
noise_std = cfg['observation']['noise_std']

add_dark = cfg['observation']['add_dark']

obs = observation.Observation(planet, start_JD, num_orbits, det, g141, NSAMP, SAMPSEQ, SUBARRAY, wl_p, f_bb_tmp,
                              depth_p, sample_rate, x_ref, y_ref, scan_speed, psf_max, outdir,
                              ssv_std=ssv_std, x_shifts=x_shifts, noise_mean=noise_mean, noise_std=noise_std,
                              add_dark=add_dark)

obs.show_lightcurve()
# plt.show()

obs.run_observation()
