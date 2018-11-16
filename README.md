![](https://travis-ci.com/ucl-exoplanets/wayne.svg?token=n6r52fwBN2Mdz9psdp2V&branch=master) [![codecov](https://codecov.io/gh/ucl-exoplanets/wayne/branch/master/graph/badge.svg?token=2EJkVpjGHV)](https://codecov.io/gh/ucl-exoplanets/wayne)


# Wayne

Wayne is a telescope simulator used to generate realistic data with all noise sources, reductions and systematics. Currently it is applied to the **HST WFC IR** instrument which uses two grisms, G102 and G141 for spectroscopy. The project has a particular focus on **transmission spectroscopy** of exoplanets.

For WFC3 IR, Wayne currently simulates:
* Staring Mode
* Spatial Scan Mode
* Scan Speed Variations
* Ramp or hook trend
* X and Y positional shifts
* Cosmic Rays
* Flat Field, Bias, Gain etc
* Read noise, stellar noise, sky background etc

Wayne is designed to be adapted to more instruments and systematics.

## Installation

Wayne is a standalone program that should be ran in its own environement. It has not been tested with different package versions. To avoid issues installing wayne, or the installation breaking other python programs, we recomend using either [virtualenvs](http://python-guide-pt-br.readthedocs.io/en/latest/dev/virtualenvs/) or [conda environments](https://conda.io/docs/using/envs.html). This will save you alot of headache.

If you are using conda, create a virtual environment, activate it, and install numpy/gcc/cython

    conda create -n wayne python=2
    source activate wayne
    conda install numpy==1.14.3
    conda install gcc==4.8.5
    conda install cython==0.28.2

Clone this repo, move to the new directory and install Wayne

    git clone https://github.com/ucl-exoplanets/wayne.git
    cd wayne
    python setup.py install

Test an example

    cd examples
    wayne -p hd209458b_12181_simulation_parameters.yml
    
*If you are developing the package, you should install in edit mode instead so you are not required to install the package again after every update*

    pip install -e .

## Running

Wayne is ran by using the `wayne` command and a parameter file.

A bunch of these are included in the `examples` folder. Move to this directory and give one a try.

    wayne -p hd209458b_12181_simulation_parameters.yml
    
This will take a while.
    
## FAQ

### I would like to contribute

Great! An easy way to get involved is by addressing some of the raised issues or improving the test coverage by adding new tests. If you have a new feature idea you would like to implement please raise an issue first explaining what you plan to do. All contributions should be submitted using pull requests.

### I am having trouble installing / using Wayne

Please raise a github issue explaining the problem

### What does Wayne stand for?

After struggling to find a good name we just chose one. [Wayne](http://coppermind.net/wiki/Wayne) is a character in the Mistborn (Era 2) books who is an expert at imitation.
