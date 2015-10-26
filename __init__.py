# coding=utf-8
""" Package for simulating spectral observations using the WFC3 instrument in Python
"""

__author__ = 'Ryan Varley'
__version__ = '0.5b.151026a'
__short_version__ = '0.5b'
__all__ = ('detector', 'observation' 'exposure', 'grism', 'models', 'params', 'tools')

import detector, exposure, grism, models, tools, params, observation

# Dependencies with settings
import matplotlib.pyplot as plt
# set it once here, then the user can override after package imports. I dont generally like the idea of overriding
# user package settings but if we do it once on import and they follow pep8 it shouldnt be an issue.
plt.style.use('ggplot')