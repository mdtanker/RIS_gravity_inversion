# Airborne Gravity Processing Workflow

Here are some notes on the common workflows for processing airborne gravity data.

## Resources:
* Oasis Montaj support documents
* Hinze et al. 2005, 10.1190/1.1988183

## Initial Corrections

* split Sorties into lines
    * removal of takeoff/landing and turns
* remove turbulance
* quality control
* positioning offset (GPS relative to sensor)
* lag correction (time offset between GPS and sensor)
* block level shifts (tares in the data)
* resampling data (ex. 10hz -> 1Hz)

## Observed Gravity

* Atmospheric correction
    * δgatm =0.874 - 9.9×10-5h + 3.56×10-9h2
    * found in Hinze et al. 2005
    * used in Oasis Montaj
* Machine drift
    * [Gsolve](https://www.sciencedirect.com/science/article/pii/S2352711018300566)
    * Constant shift on line-by-line basis to bring to absolute gravity from base station.
        * average the pre- and post-flight readings of each flight
        * remove the difference between that average and the local g value
* Tidal correction
    * body (solid-earth) tides
        * [More info](https://geodesyworld.github.io/SOFTS/solid.htm)
        * up to 30cm
        * does not average to zero
        * Equitorial bulge partially due to 1)Earth's rotation and 2) average attraction of the Moon and Sun
            * called permanent tidal deformation (PTD)
        * Software:
            * [LongmanTide](https://github.com/bradyzp/LongmanTide/)
                * uses Longman 1959 formula
            * [pyGrav](https://github.com/basileh/pyGrav/tree/master/main_code)
                * Hector and Hinderer 2016: https://doi.org/10.1016/j.cageo.2016.03.010
                * uses Cattin et al. 2015 functions:
            * [Gsolve](https://www.sciencedirect.com/science/article/pii/S2352711018300566)
                * uses Longman 1959 formula
            * [gTOOLS (matlab)](https://www.sciencedirect.com/science/article/pii/S0098300421003095)
            * [GravProcess (matlab)](https://www.sciencedirect.com/science/article/pii/S0098300415000886?via%3Dihub)
            * [GSynth (matlab)](https://github.com/Thomas-Loudis/gsynth)
            * [GMeterPy](https://github.com/opengrav/gmeterpy/blob/master/gmeterpy/corrections/atmosphere.py)
                * for atmospheric pressure and polar motion correction
            * [earthtide (R/C++)](https://github.com/jkennel/earthtide)
    * ocean tides
        * up to 10m of local sea level change
* Aircraft manoeuvres
* Eötvös correction
    * 3 equations: Exact, Glicken, Harlan
    * earth's rotation produces outward centrifugal accel
    * objects moving relative to earth experience additional accel
    * vertical component of this accel is relatived to curved earth, known as the Eotvos effect (<30mGals)
    * R. B. Harlan, "Eötvös corrections for airborne gravimetry", Journal of Geophysics Research, vol 73, no 14 (July 15, 1968).
    * M. Glicken, "Eötvös corrections for a moving gravity meter", Geophysics, vol 27, no 4 (1962), pp. 531-533.
    * Oasis Montaj [docs](https://my.seequent.com/support/search/help/oasismontaj--content_gxhelp_g_geosoft_gx_gravity_eotvoscorrection.htm?page=3&types=&product=&keyword=airborne%20&kbtypes=&language=en_US&name=E%C3%B6tv%C3%B6s%20Correction) have equations
* Levelling
    * Systematic errors
        * typically removed with known corrections (IGRF, heading, etc.)
    * Systematic noise
        * residual error
        * typically from gradient or elevation differences between data
    * software
        * [x2sys (GMT)](https://www.sciencedirect.com/science/article/pii/S0098300409002945?via%3Dihub)
            * [Example from Wei Ji](https://github.com/weiji14/deepicedrain/blob/v0.3.0/atlxi_lake.ipynb)
            * x2sys_init and x2sys_cross are wrapped by pyGMT, but not x2sys_solve yet.
        * Oasis Montaj
            *  Simple Levelling (2-steps)
                * tie lines adjust to match statistical average or trend of observed crossing survey lines
                * survey lines adjusted to exactly match tie lines
            * Empirical of microlevelling
                * gridding technique to filter residual noise
            * Workflow
                * Simple
                    * assume cross-overs on average are 0
                    * discard mis-ties in areas of high horizontal gradients
                    * analyize mis-ties and remove outliers
                    * statistically level tie lines to match flight lines (shifted, linearly trended ,splined, or b-splined)
                    * calc new intersections
                    * manually adjust specific mis-ties
                    * optionally filter the mis-tie values
                    * apply difference at cross-overs, and linearly interpolate between cross-over points
                * Careful
                    * apply shifts, tilts, spline, of tensioned spline corrections to individual lines
* Microlevelling
    * Oasis Montaj
        * Extract Noise
            * default grid cell is 1/5 line spacing
            * uses a decorrugation (directional high-pass) filter
            * 6th-order high-pass butterworth filter with a default cutoff wavelength of 4x the flight line spacing combined with a directional filter
        * Microlevel
            * low pass filter the noise to retrieve any more geological signal
            * subtract from data to get microleveled grid

## Gravity Disturbance
Disturbance vs Anomaly

* Anomaly is the difference between observed gravity on the geoid and the normal (theoretical) gravity on the same lat/lon point on the ellipsoid
    * used in geodesy to calculate the geoid
* Disturbance is difference between observed and normal gravity, both on the ellipsoid, at the same point
    * this is used by geophysics to model the subsurface

This can result in a discrepancy of over 10mGal (Oliveira et al. 2018; Should geophysicists use the gravity disturbance or the anomaly?)

Use a closed-form normal gravity formula to get the normal gravity at the observation point on the ellipsoid. Subtract it from the observed gravity to get the gravity disturbance. Observed gravity should be the signal only from the gravitational attraction of every massive body in the Earth (should it include the Eotvos correction?)
