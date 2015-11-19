from setuptools import setup, find_packages
import codecs
import os
import re

here = os.path.abspath(os.path.dirname(__file__))


# Read the version number from a source file.
def find_version(*file_paths):
    # Open in Latin-1 so that we avoid encoding errors.
    # Use codecs.open for Python 2 compatibility
    print os.path.join(here, *file_paths)
    with codecs.open(os.path.join(here, *file_paths), 'r', 'latin1') as f:
        version_file = f.read()

    # The version line must have the form
    # __version__ = 'ver'
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


# Get the long description from the relevant file
with codecs.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = ['docopt', 'numpy', 'scipy', 'matplotlib', 'pysynphot',
                    'astropy', 'pandas', 'exodata', 'quantities', 'seaborn',]

setup(
    name="Wayne",
    version=find_version('wayne', '__init__.py'),
    description="Wayne is a instrument simulator primarily for HST WFC3 IR grism spectroscopy.",
    long_description=long_description,
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

    # What does your project relate to?
    # keywords='sample setuptools development',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages.
    packages=find_packages(),

    # List run-time dependencies here.  These will be installed by pip when your
    # project is installed.
    install_requires=install_requires,

    # If there are data files included in your packages that need to be
    # installed, specify them here.  If using Python 2.6 or less, then these
    # have to be included in MANIFEST.in as well.
    include_package_data=True,
    zip_safe=False,
    # test_suite='nose.collector'
)
