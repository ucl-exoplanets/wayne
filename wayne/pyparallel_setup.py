from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import platform
import os

files_location = os.path.abspath(os.path.dirname(__file__))

current_location = os.path.abspath('.')
os.chdir(files_location)

system = platform.system()

if system == 'Linux':
    ext = Extension("pyparallel",sources=["pyparallel.pyx", "pyparallel_menu.c"],extra_compile_args=['-fopenmp','-lm','-O3'],
            extra_link_args=['-lgomp'])

    setup(name="pyparallel_library",ext_modules = cythonize([ext]))

else:
    ext = Extension("pyparallel",sources=["pyparallel.pyx", "pyparallel_menu.c"],extra_compile_args=['-fopenmp','-lm','-O3'],
            extra_link_args=['-lgomp'], include_dirs=[numpy.get_include()])

    setup(name="pyparallel_library",ext_modules = cythonize([ext]))

os.chdir(current_location)