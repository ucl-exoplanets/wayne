""" This is just a test file, used for running the code end-to-end for testing and profiling

python -m cProfile -o prof run.py
python dumpprof.py > stats.txt

dumpprof.py
---
import pstats
pstats.Stats('prof').strip_dirs().sort_stats("cumulative").print_stats()
---
"""

import numpy as np
from astropy.analytic_functions import blackbody_lambda
from astropy import units as u
import matplotlib.pyplot as plt

import tools
import detector
import grism
import observation

source_spectra = np.loadtxt('/Users/ryan/Dropbox/notebooks/data/hj_10000.dat')
source_spectra = source_spectra.T  # turn to (wl list, flux list)
source_spectra = np.array(tools.crop_spectrum(0.9, 1.8, *source_spectra))  # basic cropping here

x_ref = 550
y_ref = 550

wl_p = source_spectra[0] * u.micron
depth_p = source_spectra[1]

wl_bb = wl_p
f_bb = blackbody_lambda(wl_bb, 3250)

f_bb_tmp = f_bb / 1e20  # temp factor to account for distance

exp = observation.Exposure(detector.WFC3_IR(), grism.Grism())
exp_data = exp.staring_frame(x_ref, y_ref, wl_p, f_bb_tmp, depth_p, 1.*u.s)

# output (confirm it ran correctly!)

crop_data = exp_data[540:560, 550:750]
plt.matshow(crop_data, cmap='gist_heat')
plt.savefig('run_test.png')
