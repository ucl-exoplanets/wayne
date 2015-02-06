""" Simulation for a grism, including templates for WFC3's on board G102 and G141 grisms. These classes take a raw
spectrum (or in future through the instrument) and simulates its passing through the grism as a field. The detector class
then maps the field to pixels.
"""


class Grism(object):

    def __init__(self):
        pass


class G141(Grism):
    pass