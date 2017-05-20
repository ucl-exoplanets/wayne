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

You need install this repo and download the data needed to your system.

First clone this repo

    git clone https://github.com/ucl-exoplanets/wayne.git

Then move to the new directory and install it
    
    cd wayne
    python setup.py install
    
*If you are developing the package, you should install in edit mode instead so you are not required to install the package again after every update*

    pip install -e .

## Running

Wayne is ran by using the `wayne` command and a parameter file.

A bunch of these are included in the `examples` folder. Move to this directory and give one a try.

    wayne -p hd209458b_12181_simulation_parameters.yml
    
This will take a while.
    
## FAQ 

### What does Wayne stand for?

After struggling to find a good name we just chose one. [Wayne](http://coppermind.net/wiki/Wayne) is a character in the Mistborn (Era 2) books who is an expert at imitation.
