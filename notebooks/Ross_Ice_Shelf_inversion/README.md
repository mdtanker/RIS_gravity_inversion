# Ross Ice Shelf Gravity Inversion

This folder contains all the notebooks relevant to performing a gravity inversion to recover the sub-Ross Ice Shelf bathymetry. This code is used for Chapter 4 of this thesis.

## Prepare topography data
`prepare_grids.ipynb` downloads Bedmap2 and BedMachine v3 data, references them to the WGS84 ellipsoid, crops them to the inversion region, and saves them locally as `.zarr` files for quick loading. It also preps unused basement and moho depths.

## Create starting bathymetry model
`RIS_bathymetry_data.ipynb` creates the starting bathymetry model. This uses BedMachine v3 gridded data outside of the ice shelf and the 223 seismic constraint points within the ice shelf. To avoid re-interpolating the already gridded data outside the ice shelf, a buffer zone of gridded points combined with the ice shelf constraints to be used in the interpolation. This buffer zone is then masked, and the gridded ice shelf data is merged to the BedMachine data outside of the ice shelf. This process is repeated with a series of interpolation parameters (tension and damping) for two methods of gridding (tensioned minimum curvature and bi-cubic splines).

## Clean and level the gravity data
`ROSETTA_levelling.ipynb` downloads ROSETTA-Ice airborne gravity data, plots data by line to help identify data outliers, removes the anomalous data, upwards continues the lines individually using equivalent sources, projects the theoretical intersections of flight lines, interpolates the gravity values onto these points, calculates the misties. With these misties, iterative levelling is performed, where flight lines are levelled two tie lines, then tie lights are levelled to flight lines. This process is repeated with 0th, 1st, and 2nd order trends.

## Interpolate the gravity data
`RIS_gravity_data.ipynb` takes this levelled gravity data and interpolates it over the inversion domain with equivalent sources. It also contains some unused code for removing outliers and including various other gravity data.

## Isolate the residual component of gravity
`RIS_gravity_reduction.ipynb` uses this gridded gravity disturbance data and reduces it to the residual component of the topo-free gravity disturbance which is the input to the inversion. This process includes calculating the terrain mass effect, from the forward gravity of all anomalous terrain masses with respect to the normal Earth (the ellipsoid). From this topo-free disturbance, the regional field is estimated with the constraint-point minimization method, and once removed yields the residual gravity.

## Run the inversion
`RIS_inversion.ipynb` uses the residual gravity data to invert for bathymetry beneath the Ross Ice Shelf. This is done both as a single inversion, with a specified regularization damping parameter, and as a cross-validation where a series of damping parameters are testing, and the result with the best score is retained.

## Run the Monte Carlo simulations
`RIS_montecarlo_uncertainties.ipynb` performs several Monte Carlo simulations. The "full" simulation involves the random sampling of the gridding gravity disturbance data and constraint point depths, and the Latin Hypercube sampling of the densities of ice, water, and seafloor, and the tension factors used in gridding the starting bathymetry and gridding the regional field. With these sampled inversion inputs, 10 full inversion workflows are run (create starting bed, reduce gravity from disturbance to residual, perform cross-validated inversion) and the resulting 10 bathymetry models are used to calculate cell-specific statistics. The mean is used as the final bathymetry model, while the standard deviation is used as the uncertainty in the bathymetry. This Monte Carlo simulation is repeated with the individual sampling of 1) the gravity data, 2) the constraint points, 3) the density values, 4) the starting bed tension factor, and 5) the regional field tension factor. These additional simulations reveal the component of the total uncertainties which result from these aspects of the inversion.

## Plot the results
`RIS_inversion_figures` is a large notebook which creates all of the figures, plus others, for Chapter 4 of this thesis.

## Misc.
`RIS_outline.ipynb` is a short notebook to create a shapefile of the outline of the Ross Ice Shelf from the grounding line / ice front of the MEaSUREs Antarctic Boundaries data.
