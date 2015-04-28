# coding=utf-8
""" Package for simulating spectral observations using the WFC3 instrument in Python
"""

__author__ = 'Ryan Varley'
__version__ = '0.2.150428c'

# package imports
import grism
import detector
import params
import tools
import observation
import exposure

# Dependencies with settings
import matplotlib.pyplot as plt
# set it once here, then the user can override after package imports. I dont generally like the idea of overriding
# user package settings but if we do it once on import and they follow pep8 it shouldnt be an issue.
plt.style.use('ggplot')