import unittest

import numpy as np

from ...trend_generators import cosmic_rays


class Test_BaseCosmicGenerator(unittest.TestCase):

    def setUp(self):
        self.cosmic_gen = cosmic_rays.BaseCosmicGenerator()

    def test__number_of_cosmics(self):
        self.assertEqual(self.cosmic_gen._number_of_cosmics(1),
                         11)
        self.assertEqual(self.cosmic_gen._number_of_cosmics(2),
                         22)

    def test__generate_cosmic_energies(self):

        self.assertEqual(self.cosmic_gen._generate_cosmic_energies(1),
                         [25000])
        self.assertEqual(self.cosmic_gen._generate_cosmic_energies(2),
                         [25000, 25000])

    def test__generate_array(self):

        self.assertTrue(np.all(self.cosmic_gen._generate_array(50) ==
                        np.zeros((50, 50))))
        self.assertTrue(np.all(self.cosmic_gen._generate_array((20, 30)) ==
                        np.zeros((20, 30))))

    def test__cosmics_to_array(self):
        energy_list = [1, 10, 5]
        cosmic_array = self.cosmic_gen._cosmics_to_array(energy_list,
                                                         np.zeros((10,10)))

        self.assertEqual(np.sum(cosmic_array), np.sum(energy_list))
        self.assertEqual(cosmic_array.shape, (10, 10))

    def test_cosmic_frame(self):
        cosmic_array = self.cosmic_gen.cosmic_frame(2, 10)
        self.assertEqual(np.sum(cosmic_array), 11*2*25000)
        self.assertEqual(cosmic_array.shape, (10, 10))


class Test_MinMaxPossionCosmicGenerator(unittest.TestCase):

    def setUp(self):
        self.cosmic_gen = \
            cosmic_rays.MinMaxPossionCosmicGenerator(11, 10000, 35000)

    def test__number_of_cosmics(self):

        energies = [self.cosmic_gen._number_of_cosmics(1) for i in xrange(100)]
        mean_energy = np.mean(energies)

        self.assertTrue(10 <= mean_energy <= 12,
                        'Small chance of failing due to random sample - '
                        '10 <= {} <= 12'.format(mean_energy))

    def test__generate_cosmic_energies(self):

        energies = self.cosmic_gen._generate_cosmic_energies(3)
        self.assertEqual(len(energies), 3)

        min = self.cosmic_gen.min_count
        max = self.cosmic_gen.max_count

        for energy in energies:
            self.assertTrue(min <= energy <= max)