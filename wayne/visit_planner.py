import numpy as np
from astropy import units as u


def VisitPlanner(detector, NSAMP, SAMPSEQ, SUBARRAY, num_orbits=3,
                 time_per_orbit=54 * u.min, hst_period=95 * u.min,
                 exp_overhead=1 * u.min):
    """ Returns the start time of each exposure in minutes starting at 0.
     Useful for estimating buffer dumps etc.

    Note: Currently does not include:
    * spacecraft manuvers
    * buffer dump times based on frame size + number
    * variable final read times (subarray dependant)
    * support for single / combined scan up / scan down modes

    :param detector: detector class, i.e. WFC3_IR()
    :type detector: detector.WFC3_IR
    :param grism: grism class i.e. G141
    :type grism: grism.Grism

    :param NSAMP: number of sample up the ramp, effects exposure time (1 to 15)
    :type NSAMP: int
    :param SAMPSEQ: Sample sequence to use, effects exposure time
    ('RAPID', 'SPARS10', 'SPARS25', 'SPARS50',
    'SPARS100', 'SPARS200', 'STEP25', 'STEP50', 'STEP100', 'STEP200', 'STEP400'
    :type SAMPSEQ: str
    :param SUBARRAY: subarray to use, effects exposure time and array size.
    (1024, 512, 256, 128, 64)
    :type SUBARRAY: int

    :param num_orbits: number of orbits
    :type num_orbits: int
    :param time_per_orbit: Time target is in sight per orbit
    :type time_per_orbit: astropy.units.quantity.Quantity
    :param hst_period: How long it takes for HST to complete an orbit
    :type: astropy.units.quantity.Quantity
    :param exp_overhead: Overhead (downtime) per exposure. Can be varied to
    account for spacefract manuvers or one
    direction scanning
    :type exp_overhead: astropy.units.quantity.Quantity

    :return: Dictionary containing information about the run
    :rtype: dict

    Output dictionary contains:

    * 'exp_times': array of exposure start times
    * 'NSAMP': number of samples
    * 'SAMPSEQ': SAMPSEQ mode
    * 'SUBARRAY': SUBARRAY mode
    * 'num_exp': number of exposures in visit
    * 'exptime': how long each exposure takes
    * 'num_orbits': number of orbits specified
    * 'exp_overhead': overhead per exposure
    * 'time_per_orbit': Time target visible in single orbit
    * 'hst_period': Orbital period of HST
    """

    # note this may need to be done in a slow loop as we have differing
    # overheads, and buffer calcs
    # and other things like changing modes and scan up/down to consider
    # this is slower but not prohibitively so since this isnt run often!
    # The complexity here suggests this should
    # be moved to a class and broken down into smaller calcs

    exptime = detector.exptime(NSAMP, SUBARRAY, SAMPSEQ)
    exp_per_dump = detector.num_exp_per_buffer(NSAMP, SUBARRAY)

    # The time to dump an n-sample, full-frame exposure is approximately
    # 39 + 19 x (n + 1) seconds. Subarrays may also
    # be used to reduce the overhead of serial buffer dumps. ** Instruemtn handbook 10.3
    time_buffer_dump = 5.8 * u.min  # IH 23 pg 209

    # temp defined in input to make tests sparser and account for lack of manouvers
    # exp_overhead = 1*u.min  # IH 23 pg 209 - should be mostly read times
    # - seems long, 1024 only?


    # TODO spacecraft manuvers IR 23 pg 206 for scanning
    exp_times = []
    orbit_start_index = []
    buffer_dump_index = []
    for orbit_n in xrange(num_orbits):
        if orbit_n == 0:
            guide_star_aq = 6 * u.min
        else:
            guide_star_aq = 5 * u.min

        # record the expnum of each orbit start
        orbit_start_index.append(len(exp_times))
        start_time = hst_period * orbit_n

        visit_time = start_time + guide_star_aq
        visit_end_time = start_time + time_per_orbit

        exp_n = 0  # For buffer dumps - with mixed exp types we should really
        #  track headers and size
        while visit_time < visit_end_time:

            # you cant convert a list of quantities to an array so we have to
            #  either know the length to preset one or
            # use floats in a list and convert after.
            exp_times.append(visit_time.to(u.min).value)  # start of exposure
            visit_time += (exptime + exp_overhead)

            exp_n += 1
            if exp_n > exp_per_dump:
                visit_time += time_buffer_dump
                exp_n = 0

                buffer_dump_index.append(len(exp_times))

    returnDict = {
        'exp_times': np.array(exp_times) * u.min,  # start_times?
        'NSAMP': NSAMP,
        'SAMPSEQ': SAMPSEQ,
        'SUBARRAY': SUBARRAY,
        'num_exp': len(exp_times),
        'exptime': exptime,
        'num_orbits': num_orbits,
        'exp_overhead': exp_overhead,
        'time_per_orbit': time_per_orbit,
        'hst_period': hst_period,
        'buffer_dump_index': buffer_dump_index,
        'orbit_start_index': orbit_start_index,
    }

    return returnDict
