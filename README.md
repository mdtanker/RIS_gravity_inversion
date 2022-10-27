# RIS_gravity_inversion
Here we use gravity data from Antarctica's Ross Ice Shelf region, mainly airborne data from the ROSETTA-Ice project, to perform a gravity inversion to model the sub-ice shelf bathymetry. 

## Install

Install the required dependencies with either `conda` or `mamba` with the environment.yml file:

    mamba env create -f environment.yml

If you get an error from pyproj while using the package try:

    conda remove --force pyproj
    pip install pyproj

Add local antarctic_plots repo in editable mode:

    cd antarctic_plots 
    pip install -e . 
    <!-- pip install antarctic_plots --no-binary :all: -->

Open and work through `RIS_inversion.ipynb` to see an example of the inversion.

## To Do

### Inversion-specific
#### Features
* increase grid spacing of buffer zone
    * use discretize package, rioxarray.pad_box or xarray.pad 
* implement depth-dependent density 
    * both ice and sediment (like harmonica.tesseroid_gravity)
#### Bugs
* test with layers with NaN's (instead of ice and water elev = 0 for no ice, make nans)
* fix Gauss-Newton Least Squares Solution 
* fix Steepest Descent Least Squares Solution
* fix density inversion
* fix Jacobian calculation with annulus
#### Improvements
* use xrft for coherency and power spectrum: https://xrft.readthedocs.io/en/latest/MITgcm_example.html
* use xrscipy for coherency and spectra: https://xr-scipy.readthedocs.io/en/latest/spectral.html
* use vd.median_distance for constraints grid construction
* use xarray-spatial proximity for constraints grid: https://xarray-spatial.org/user_guide/proximity.html
* use rioxarray.reproject_match for resampling
* use hm.EquivalentSources.jacobian to calculate jac.
* use Dask Bags to parallelize forward grav calculations of a dict of prism layers
#### Misc
* add hm.EquivalentSources as method for regional calc
* use vd.Trend.jacobian for regularization
* use vd.base.least_squares for optimization
* add spacing parameter to grdfilters

### Other
#### Features
* use rosetta ice thickness
* do forward model with rosetta density
* use Boule to remove normal gravity
    - using rosetta "FAG_levelled" channel
#### Bugs
* fix constraints on GL plotting
* fix pyvista to work on remote server
    * use pvxarray for plotting topographies in 3D

## Questions:
* should we use raw gravity observation points or interpolated grid?



## Other Gravity Inversion Software 

### Geosoft Oasis Montaj 
* 3D 
* Square grid network 

### MiraGeoscience 
* 3D
* offer Vertical Prisms inversion

### Fatiando a Terra / Harmonica 

### SimPEG 
* 3D
* mesh-based, not well suited (yet) for vertical prism inversion

### PyGIMLI 
* 3D
* mesh-based

### GNS Woodward Fortran Code 
* 3D 
* Triangular grid network 
* Irregular grid spacing 
* Accepts constraint cells 

### GNS Nagy / Woodward Python Code 
* 3D 
* Same optimization of GNS fortran code 
* Square grid network 
* different grid spacing for each layer 

### Growth 3.0 Fortran Code
* 3D
* accepts seeds
