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

Currently this is clunky. Add this folder to your python path, install the requirements (including this one https://github.com/ucl-exoplanets/pylightcurve)

ask Ryan or email ryanjvarley@gmail.com for help. I will package this up soon.

## Running

Once install move to the wayne directory and edit a parameter file e.g. HD209458b_par.yml, in particular the `oec_location` and `outdir`.

Then run by typing

```
python run_visit.py -p HD209458b_par.yml
```

Good Luck
