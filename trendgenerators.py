""" This class contains generator types for creating several different types
of trend and noise. The idea is each type of generator has a base class which
defines the main methods but each can be overridden to create totally custom
trend types as long as they keep the same output function.
"""

import numpy as np


class BaseCosmicGenerator(object):

    def __init__(self):
        """ This is the base class, it defines generates 11 cosmics per second
        of energy 25000. The methods are designed to be overidden by other
        classes.

        :return:
        """

        pass

    def _number_of_cosmics(self, time, size=1024):
        """ Given a time returns the number of cosmic impacts, the number is
        static and not array size dependant

        :param time: time in seconds
        :return:
        """

        full_frame_rate = 11

        return full_frame_rate * time

    def _rate_full_frame_to_size(self, full_frame_rate, size):
        """ Converts counts per full frame (1024, 1024) to size
        """

        if isinstance(size, int):
            num_pixels = size * size
        else:
            num_pixels = size[0] * size[1]

        rate_size = full_frame_rate/(1024.*1024.) * num_pixels

        return rate_size

    def _generate_cosmic_energies(self, number):
        """ Generates cosmic energies for `number` of cosmics

        :param number: number of cosmic to generate energies for
        :return:
        """

        return [25000] * number

    def _generate_array(self, size):
        """ Generates a 2D array of size `size`. Can either be an int
        (and square), or a tuple of (height, width)
        :param size:
        :return:
        """

        try:  # assuming a square array (single valued size)
            size = int(size)
            array = np.zeros((size, size))
        except TypeError:  # cant convert because we have a tuple
            array = np.zeros(size)

        return array

    def _cosmics_to_array(self, list_of_energies, array):
        """ Takes a list of cosmic energies and randomly places them across the
        array

        :param list_of_energies:
        :return:
        """

        number_of_cosmics = len(list_of_energies)

        y_pos = np.random.randint(0, len(array), number_of_cosmics)
        x_pos = np.random.randint(0, len(array[0]), number_of_cosmics)

        for i, cosmic_energy in enumerate(list_of_energies):
            array[y_pos[i], x_pos[i]] += cosmic_energy

        return array

    def cosmic_frame(self, time, size=1024):
        """ generates a frame of cosmic arrays given a time and array size

        :param time: time period to generate cosmics over
        :param size: size of the array either int square or tuple of
        (height, width)
        :return:
        """

        num_cosmics = self._number_of_cosmics(time, size)
        cosmic_energies = self._generate_cosmic_energies(num_cosmics)
        cosmic_array = self._generate_array(size)

        cosmic_array = self._cosmics_to_array(cosmic_energies, cosmic_array)

        return cosmic_array


class MinMaxPossionCosmicGenerator(BaseCosmicGenerator):

    def __init__(self, rate=11., min_count=10000, max_count=35000):
        """ Generates cosmic rays between a minimum and maximum level sampled
        on a poisson distribution at a rate of `rate` per second per full frame
        1024x1024

        :param rate: number of cosmic hits per second (modelled as a poisson)
        """

        BaseCosmicGenerator.__init__(self)

        self.rate = rate
        self.min_count = min_count
        self.max_count = max_count

    def _number_of_cosmics(self, time, size=1024):
        """ Poisson distribution
        """

        size_rate = self._rate_full_frame_to_size(self.rate, size)

        return np.random.poisson(size_rate*time)

    def _generate_cosmic_energies(self, number):
        """ Cosmic energies are a generated randomly between min_count to
        max_count
        """

        energies = np.random.randint(self.min_count, self.max_count, number)

        if number == 1:
            energies = [energies]

        return energies