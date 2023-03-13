# RIS_gravity_inversion
Here we use gravity data from Antarctica's Ross Ice Shelf region, mainly airborne data from the ROSETTA-Ice project, to perform a gravity inversion to model the sub-ice shelf bathymetry.

## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/RIS_gravity_inversion

## Dependencies

Install the required dependencies with either `conda` or `mamba` with the environment.yml file:

    cd RIS_gravity_inversion

    make install

Activate the newly created environment:

    conda activate RIS_gravity_inversion

If you get errors related to the PyProj EPSG database, try the following:

    mamba install -c conda-forge proj-data --force-reinstall -y

or

    conda remove --force pyproj -y
    pip install pyproj --force-reinstall

Run the tests to make sure the install worked correctyl:

    make test


## Data sources

### Bedmap2
Surface Coverage | Bed Coverage
:---:|:---:
![](figures/bedmap2_surface_coverage.JPG)|![](figures/bedmap2_bed_coverage.JPG)

For the Ross Ice Shelf:
* surface elevation from satellite altimetry (Griggs and Bamber, 2011)
* ice thickness from satellite altimetry measurements of freeboard (Griggs and Bamber, 2011)
* bed from Timmerman et al. 2010


## To Do

### Inversion-specific
#### Features
* for gmt surface:
    * from Scheinert et al. 2016 "It is recommended to use values of 0.25, ... , 0.3 for potential field data, whereas a larger tension factor (0.35) should be used for topography data [Wessel and Smith, 2015]. Here a tension factor of 0.3 was utilized."
    * use this for creating a surface between constraint points

#### Bugs
* test with layers with NaN's (instead of ice and water elev = 0 for no ice, make nans)
* fix density inversion

#### Improvements
* use xrft for coherency and power spectrum: https://xrft.readthedocs.io/en/latest/MITgcm_example.html
* use xrscipy for coherency and spectra: https://xr-scipy.readthedocs.io/en/latest/spectral.html
* use xarray-spatial proximity for constraints grid: https://xarray-spatial.org/user_guide/proximity.html
* use rioxarray.reproject_match for resampling
* use Dask Bags to parallelize forward grav calculations of a dict of prism layers

### Other
#### Features
* use rosetta ice thickness
* do forward model with rosetta density
* use Boule to remove normal gravity
    - using rosetta "FAG_levelled" channel
#### Bugs
* fix pyvista to work on remote server
    * use pvxarray for plotting topographies in 3D



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
