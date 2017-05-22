import os

import numpy
from setuptools import setup, find_packages, Extension

VERSION = '1.0.0.dev'

SHORT_DESCRIPTION = """
Wayne is a instrument simulator primarily for HST WFC3 IR grism spectroscopy.
"""

LONG_DESCRIPTION = """
Wayne is a telescope simulator used to generate realistic data with most of
the noise sources, reductions and systematics present in real data. Currently 
only the HST WFC IR instrument has been implemented. The project has a 
particular focus on transmission spectroscopy of exoplanets.
"""

# TODO (ryan) scrap req.txt and define versions here or parse it here
install_requires = ['docopt', 'numpy', 'scipy', 'matplotlib', 'pysynphot',
                    'astropy', 'pandas', 'exodata', 'quantities',
                    'pyfits', 'cython', 'ephem', 'pymc']

_here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="Wayne",
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url='https://github.com/ucl-exoplanets/Wayne',
    author='Ryan Varley',
    author_email='ryan@ryanvarley.uk',
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],

    entry_points={
        'console_scripts': [
            'wayne = wayne.run_visit:run',
        ],
    },

    packages=find_packages(),
    setup_requires=["pytest-runner"],
    install_requires=install_requires,
    tests_require=['pytest'],
    include_package_data=True,
    zip_safe=False,
    ext_modules=[
        Extension("pyparallel",
                  sources=[os.path.join('wayne', x) for x in
                           ("pyparallel.pyx", "pyparallel_menu.c")],
                  extra_compile_args=['-fopenmp', '-lm', '-O3'],
                  extra_link_args=['-lgomp'],
                  include_dirs=[numpy.get_include()])
    ]
)
