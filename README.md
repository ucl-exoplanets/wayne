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

Wayne is designed to be adapted to more instruments and systematics. Wayne got his name from the [character](http://coppermind.net/wiki/Wayne) in from the Era 2 Mistborn books who is an expert at imitation.

## Installation

You need install this repo and download the data needed to your system.

First clone this repo

    git clone https://github.com/ucl-exoplanets/wayne.git

Then move to the new directory and install it
    
    cd wayne
    python setup.py install
    
*If you are developing the package, you should install in edit mode instead so you are not required to install the package again after every update*

    pip install -e .

Now you need the data files from [here](https://www.dropbox.com/s/49cyy7el37d58a6/wayne_files.zip?dl=0). Put these in the folder `~/.wayne` or elsewhere and set the `WAYNE_DATA` environment variable.

You can optionally install the [Open exoplanet Catalogue](https://github.com/OpenExoplanetCatalogue/open_exoplanet_catalogue) and set the location to it in the parameter files. Otherwise leave this blank and the catalogue will be obtained from the web.

## Running

Wayne is ran by using the `wayne` command and a parameter file.

A bunch of these are included in the `examples` folder you can download [here](https://www.dropbox.com/s/2qswujobc97z5a9/wayne_examples.zip?dl=0). Move to this directory and give one a try.

    wayne -p HD209458b_par.yml
    
This will take a while (~6h)

You should take a look at example_par.yml in this folder for information on the configuration file. The other files show the file setup in a few other scenarios.
    
## FAQ 

### What package versions do you use?

    V-PY    = '2.7.10 final'       / Python version used                            
    V-NP    = '1.9.3   '           / NumPy version used                                
    V-SP    = '0.16.0  '           / SciPy version used                             
    V-AP    = '1.0.4   '           / AstroPy version used                           
    V-PD    = '0.16.2  '           / Pandas version used 

### Im getting a weird pandas column error

update pandas

### Im getting a weird numpy *= error

update numpy