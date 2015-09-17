""" This script allows you to run a visit from a parameter file. Currently
 parameter files are in python format, see example_par.py.

Usage:
    run_visit.py [-p <parameter_file>]

Options:
    parameter_file  parameter file location
"""

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
from astropy.analytic_functions import blackbody_lambda
from astropy import units as u
import exodata
import quantities as pq  # exodata still uses this
import yaml
import docopt

import observation
import detector
import grism
import tools
import params


if __name__ == '__main__':

    arguments = docopt.docopt(__doc__)
    parameter_file = arguments['<parameter_file>']

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)


    plt.style.use('ggplot')

    outdir = cfg['general']['outdir']

    if not os.path.exists(outdir):
        os.mkdir(outdir)

     # copy parfile to output
    shutil.copy2(parameter_file, os.path.join(outdir, os.path.basename(parameter_file)))

    exodb = exodata.OECDatabase(cfg['general']['oec_location'])

    seed = cfg['general']['seed']
    np.random.seed(seed)
    params.seed = seed  # tell params what the seed is now we've change it

    planet = exodb.planetDict[cfg['target']['name']]

    source_spectra = np.loadtxt(cfg['target']['spectrum_file'])
    source_spectra = source_spectra.T  # turn to (wl list, flux list)
    source_spectra = np.array(tools.crop_spectrum(0.9, 1.8, *source_spectra))

    # if, else in future
    g141 = grism.G141()
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

    # convert pq to u
    start_JD = float((planet.transittime - 0.16*pq.day).rescale(pq.day)) * u.day
    num_orbits = cfg['observation']['num_orbits']
    sample_rate = cfg['observation']['sample_rate'] * u.ms
    scan_speed = cfg['observation']['scan_speed'] * (u.pixel/u.s)
    psf_max = cfg['observation']['psf_max']

    ssv_std = cfg['observation']['ssv_std']
    x_shifts = cfg['observation']['x_shifts']

    noise_mean = cfg['observation']['noise_mean']
    noise_std = cfg['observation']['noise_std']

    add_dark = cfg['observation']['add_dark']
    add_flat = cfg['observation']['add_flat']

    sky_background = cfg['observation']['sky_background'] * u.count/u.s
    cosmic_rate = cfg['observation']['cosmic_rate']


    obs = observation.Observation(outdir)

    obs.setup_detector(det, NSAMP, SAMPSEQ, SUBARRAY)
    obs.setup_grism(g141)
    obs.setup_target(planet, wl_p, f_bb_tmp, depth_p)
    obs.setup_visit(start_JD, num_orbits)
    obs.setup_reductions(add_dark, add_flat)
    obs.setup_observation(x_ref, y_ref, scan_speed)
    obs.setup_simulator(sample_rate, psf_max)
    obs.setup_trends(ssv_std, x_shifts)
    obs.setup_noise_sources(sky_background, cosmic_rate)
    obs.setup_gaussian_noise(noise_mean, noise_std)

    visit_trend_coeffs = cfg['trends']['visit_trend_coeffs']

    if visit_trend_coeffs is not None:
        obs.setup_visit_trend(visit_trend_coeffs)

    obs.show_lightcurve()
    plt.savefig(os.path.join(outdir, 'visit_plan.png'))
    plt.close()
    
    obs.run_observation()
