""" This script allows you to run a visit from a parameter file in yaml format.
see example_par.yml.

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


class WFC3SimConfigError(BaseException):
    pass


def run():
    arguments = docopt.docopt(__doc__)
    parameter_file = arguments['<parameter_file>']

    with open(parameter_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    logger.info('WFC3Sim Started, parsing config file')

    outdir = cfg['general']['outdir']
    params.outdir = outdir


    oec_path = cfg['general']['oec_location']
    if oec_path:
        exodb = exodata.OECDatabase(oec_path)
    else:
        exodb = exodata.load_db_from_url()

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # copy parfile to output
    shutil.copy2(parameter_file, os.path.join(outdir, os.path.basename(parameter_file)))

    try:
        seed = cfg['general']['seed']
    except KeyError:
        seed = None
    if not seed or seed is None:
        np.random.seed(seed)
        params.seed = np.random.get_state()[1][0]  # tell params what the seed is for exp header

    grisms = {
        'G141': grism.G141(),
        'G102': grism.G102()
    }
    chosen_grism = grisms[cfg['observation']['grism']]
    det = detector.WFC3_IR()

    rebin_resolution = cfg['target']['rebin_resolution']

    # Check for transmission spectroscopy mode
    try:
        planet_spectrum = cfg['target']['planet_spectrum_file']
        shutil.copy2(planet_spectrum, os.path.join(outdir, os.path.basename(planet_spectrum)))

        transmission_spectroscopy = True
    except KeyError:
        transmission_spectroscopy = False

    if transmission_spectroscopy:
        try:
            planet = exodb.planetDict[cfg['target']['name']]
        except KeyError:
            planet = cfg['target']['name']

        # modify planet params if given Note that exodata uses a different unit
        # package at present
        try:
            transittime = cfg['target']['transit_time'] * aq.JD
        except KeyError:
            transittime = None

        try:
            period = cfg['target']['period'] * aq.day
        except KeyError:
            period = None

        try:
            rp = cfg['target']['rp'] * aq.R_j
        except KeyError:
            rp = None

        try:
            sma = cfg['target']['sma'] * aq.au
        except KeyError:
            sma = None

        try:
            stellar_radius = cfg['target']['stellar_radius'] * aq.R_s
        except KeyError:
            stellar_radius = None

        try:
            inclination = cfg['target']['inclination'] * aq.deg
        except KeyError:
            inclination = None

        try:
            eccentricity = cfg['target']['eccentricity']
        except KeyError:
            eccentricity = None

        try:
            ldcoeffs = cfg['target']['ldcoeffs']
        except KeyError:
            ldcoeffs = None

        try:
            periastron = cfg['target']['periastron']
        except KeyError:
            periastron = None

        wl_planet, depth_planet = tools.load_and_sort_spectrum(planet_spectrum)
        wl_planet, depth_planet = np.array(
            tools.crop_spectrum(0.9, 1.8, wl_planet, depth_planet))

        wl_planet = wl_planet * u.micron

        if rebin_resolution:
            new_wl = tools.wl_at_resolution(
                rebin_resolution, chosen_grism.wl_limits[0].value,
                chosen_grism.wl_limits[1].value)

            depth_planet = tools.rebin_spec(wl_planet.value, depth_planet, new_wl)
            wl_planet = new_wl * u.micron

    else:
        depth_planet = None
        planet = None
        transittime = None
        period = None
        rp = None
        sma = None
        stellar_radius = None
        inclination = None
        eccentricity = None
        ldcoeffs = None
        periastron = None
        try:
            planet = exodb.planetDict[cfg['target']['name']]
        except KeyError:
            planet = cfg['target']['name']

    stellar_spec_file = cfg['target']['stellar_spectrum_file']
    if stellar_spec_file:
        wl_star, flux_star = tools.load_pheonix_stellar_grid_fits(stellar_spec_file)

        if transmission_spectroscopy:
            flux_star = tools.rebin_spec(wl_star, flux_star, np.array(wl_planet))
        elif rebin_resolution:  # not transmission spectro mode
            new_wl = tools.wl_at_resolution(
                rebin_resolution, chosen_grism.wl_limits[0].value,
                chosen_grism.wl_limits[1].value)

            flux_star = tools.rebin_spec(wl_star, flux_star, new_wl)
            wl_star = new_wl

        flux_units = u.erg / (u.angstrom * u.s * u.sr * u.cm**2)
        flux_star = flux_star * flux_units
    else:  # use blackbody
        if transmission_spectroscopy:
            flux_star = blackbody_lambda(wl_planet, planet.star.T)
        else:
            raise WFC3SimConfigError(
                "Must give the stellar spectrum if not using transmission spectroscopy")

    stellar_flux_scaled = flux_star * cfg['target']['flux_scale'] * u.sr

    if transmission_spectroscopy:
        wl = wl_planet
    else:
        wl = wl_star * u.micron

    x_ref = cfg['observation']['x_ref']
    y_ref = cfg['observation']['y_ref']
    NSAMP = cfg['observation']['NSAMP']
    SAMPSEQ = cfg['observation']['SAMPSEQ']
    SUBARRAY = cfg['observation']['SUBARRAY']

    # convert pq to u
    start_JD = cfg['observation']['start_JD'] * u.day
    num_orbits = cfg['observation']['num_orbits']

    spatial_scan = cfg['observation']['spatial_scan']

    if spatial_scan:
        sample_rate = cfg['observation']['sample_rate'] * u.ms
        scan_speed = cfg['observation']['scan_speed'] * (u.pixel/u.s)
    else:
        sample_rate = False
        scan_speed = False

    ssv_classes = {
        'sine': scan_speed_varations.SSVSine,
        'mod-sine': scan_speed_varations.SSVModulatedSine,
    }
    ssv_type = cfg['observation']['ssv_type']
    if ssv_type:
        try:
            ssv_class = ssv_classes[ssv_type]
        except KeyError:
            raise WFC3SimConfigError("Invalid ssv_type given")

        ssv_coeffs = cfg['observation']['ssv_coeffs']
        ssv_gen = ssv_class(*ssv_coeffs)
    else:
        ssv_gen = None


    x_shifts = cfg['observation']['x_shifts']
    x_jitter = cfg['observation']['x_jitter']
    y_shifts = cfg['observation']['y_shifts']
    y_jitter = cfg['observation']['y_jitter']

    noise_mean = cfg['observation']['noise_mean']
    noise_std = cfg['observation']['noise_std']

    add_dark = cfg['observation']['add_dark']
    add_flat = cfg['observation']['add_flat']
    add_gain_variations = cfg['observation']['add_gain_variations']
    add_non_linear = cfg['observation']['add_non_linear']
    add_read_noise = cfg['observation']['add_read_noise']
    add_initial_bias = cfg['observation']['add_initial_bias']

    sky_background = cfg['observation']['sky_background']
    cosmic_rate = cfg['observation']['cosmic_rate']

    clip_values_det_limits = cfg['observation']['clip_values_det_limits']

    try:
        exp_start_times = cfg['observation']['exp_start_times']
    except KeyError:
        exp_start_times = False

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
    sky_background = sky_background * u.count/u.s

    obs = observation.Observation(outdir)

    obs.setup_detector(det, NSAMP, SAMPSEQ, SUBARRAY)
    obs.setup_grism(chosen_grism)
    obs.setup_target(planet, wl, depth_planet, stellar_flux_scaled, transittime,
                     ldcoeffs, period, rp, sma, inclination, eccentricity, periastron,
                     stellar_radius)
    obs.setup_visit(start_JD, num_orbits, exp_start_times)
    obs.setup_reductions(add_dark, add_flat, add_gain_variations, add_non_linear,
                         add_initial_bias)
    obs.setup_observation(x_ref, y_ref, spatial_scan, scan_speed)
    obs.setup_simulator(sample_rate, clip_values_det_limits)
    obs.setup_trends(ssv_gen, x_shifts, x_jitter, y_shifts, y_jitter)
    obs.setup_noise_sources(sky_background, cosmic_rate, add_read_noise)
    obs.setup_gaussian_noise(noise_mean, noise_std)

    visit_trend_coeffs = cfg['trends']['visit_trend_coeffs']

    if visit_trend_coeffs:
        obs.setup_visit_trend(visit_trend_coeffs)

    obs.show_lightcurve()
    plt.savefig(os.path.join(outdir, 'visit_plan.png'))
    plt.close()

    obs.run_observation()

if __name__ == '__main__':
    run()
