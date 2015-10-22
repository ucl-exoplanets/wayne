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
import exodata.astroquantities as aq
import quantities as pq  # exodata still uses this
import yaml
import docopt
import seaborn

from wfc3simlog import logger
import observation
import detector
import grism
import tools
import params
from trend_generators import scan_speed_varations

seaborn.set_style("whitegrid")

if __name__ == '__main__':
    arguments = docopt.docopt(__doc__)
    parameter_file = arguments['<parameter_file>']

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    logger.info('WFC3Sim Started, parsing config file')

    outdir = cfg['general']['outdir']
    params.outdir = outdir

    if not os.path.exists(outdir):
        os.mkdir(outdir)

     # copy parfile to output
    shutil.copy2(parameter_file, os.path.join(outdir, os.path.basename(parameter_file)))

    exodb = exodata.OECDatabase(cfg['general']['oec_location'])

    seed = cfg['general']['seed']
    np.random.seed(seed)
    params.seed = seed  # tell params what the seed is now we've change it

    # if, else in future
    g141 = grism.G141()
    det = detector.WFC3_IR()

    planet = exodb.planetDict[cfg['target']['name']]

    # modify planet params if given Note that exodata uses a different unit
    # package at present
    if cfg['target']['transit_time']:
        planet.transittime = cfg['target']['transit_time'] * aq.JD

    source_spectra = np.loadtxt(cfg['target']['planet_spectrum_file'])
    source_spectra = source_spectra.T  # turn to (wl list, flux list)
    source_spectra = np.array(tools.crop_spectrum(0.9, 1.8, *source_spectra))

    wl_p = source_spectra[0] * u.micron
    depth_p = source_spectra[1]

    rebin_resolution = cfg['target']['rebin_resolution']
    if rebin_resolution:
        new_wl = tools.wl_at_resolution(rebin_resolution,
                                        g141.wl_limits[0].value,
                                        g141.wl_limits[1].value)

        depth_p = tools.rebin_spec(wl_p.value, depth_p, new_wl)
        wl_p = new_wl * u.micron


    stellar_spec_file = cfg['target']['stellar_spectrum_file']
    if stellar_spec_file:
        stellar_wl, stellar_flux = tools.load_pheonix_stellar_grid_fits(stellar_spec_file)

        stellar_flux = tools.rebin_spec(stellar_wl, stellar_flux, np.array(wl_p))

        flux_units = u.erg / (u.angstrom * u.s * u.sr * u.cm**2)
        stellar_flux *= flux_units
    else:  # use blackbody
        stellar_flux = blackbody_lambda(wl_p, planet.star.T)

    stellar_flux_scaled = stellar_flux * cfg['target']['flux_scale'] * u.sr 

    x_ref = cfg['observation']['x_ref']
    y_ref = cfg['observation']['y_ref']
    NSAMP = cfg['observation']['NSAMP']
    SAMPSEQ = cfg['observation']['SAMPSEQ']
    SUBARRAY = cfg['observation']['SUBARRAY']

    # convert pq to u
    start_JD = cfg['observation']['start_JD'] * u.day
    num_orbits = cfg['observation']['num_orbits']
    sample_rate = cfg['observation']['sample_rate'] * u.ms
    scan_speed = cfg['observation']['scan_speed'] * (u.pixel/u.s)

    ssv_type = cfg['observation']['ssv_type']
    if ssv_type == 'sine':
        ssv_class = scan_speed_varations.SSVSine
    elif ssv_type == 'mod-sine':
        ssv_class = scan_speed_varations.SSVModulatedSine

    ssv_coeffs = cfg['observation']['ssv_coeffs']
    if ssv_coeffs:
        ssv_gen = ssv_class(*ssv_coeffs)
    else:
        ssv_gen = None

    x_shifts = cfg['observation']['x_shifts']
    y_shifts = cfg['observation']['y_shifts']

    noise_mean = cfg['observation']['noise_mean']
    noise_std = cfg['observation']['noise_std']

    add_dark = cfg['observation']['add_dark']
    add_flat = cfg['observation']['add_flat']
    add_gain_variations = cfg['observation']['add_gain_variations']
    add_non_linear = cfg['observation']['add_non_linear']
    add_read_noise = cfg['observation']['add_read_noise']

    sky_background = cfg['observation']['sky_background']
    cosmic_rate = cfg['observation']['cosmic_rate']

    clip_values_det_limits = cfg['observation']['clip_values_det_limits']

    exp_start_times = cfg['observation']['exp_start_times']

    if exp_start_times:  # otherwise we use the visit planner
        logger.info('Visit planner disabled: using start times from {}'.format(exp_start_times))
        exp_start_times = np.loadtxt(exp_start_times) * u.day

    # check to see if we have numbers of file paths, and load accordingly
    if isinstance(x_ref, str):
        x_ref = np.loadtxt(x_ref)

    if isinstance(y_ref, str):
        y_ref = np.loadtxt(y_ref)

    if isinstance(sky_background, str):
        sky_background = np.loadtxt(sky_background)
    sky_background *= u.count/u.s

    obs = observation.Observation(outdir)

    obs.setup_detector(det, NSAMP, SAMPSEQ, SUBARRAY)
    obs.setup_grism(g141)
    obs.setup_target(planet, depth_p, wl_p, stellar_flux_scaled)
    obs.setup_visit(start_JD, num_orbits, exp_start_times)
    obs.setup_reductions(add_dark, add_flat, add_gain_variations, add_non_linear)
    obs.setup_observation(x_ref, y_ref, scan_speed)
    obs.setup_simulator(sample_rate, clip_values_det_limits)
    obs.setup_trends(ssv_gen, x_shifts, y_shifts)
    obs.setup_noise_sources(sky_background, cosmic_rate, add_read_noise)
    obs.setup_gaussian_noise(noise_mean, noise_std)

    visit_trend_coeffs = cfg['trends']['visit_trend_coeffs']

    if visit_trend_coeffs:
        obs.setup_visit_trend(visit_trend_coeffs)

    obs.show_lightcurve()
    plt.savefig(os.path.join(outdir, 'visit_plan.png'))
    plt.close()

    obs.run_observation()
