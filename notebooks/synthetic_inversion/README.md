# Synthetic gravity inversion

This folder contains all the notebooks relevant to performing a synthetic gravity inversion for bathymetry. This code is used for Chapter 3 of this thesis.

## Synthetic gravity and starting bathymetry model
`synthetic_simple_model.ipynb` uses various gaussian functions to create a synthetic bathymetry model.  From this, the synthetic observed gravity data is created from the forward gravity calculation of this synthetic bathymetry. To simulate having limited knowledge of the bathymetry prior to the inversion, a low-resolution version of this bathymetry is created. Randomly located constraint points are used to sampled the full resolution bathymetry, and the entire region is then interpolated using just these sampled values. This interpolated surface is the starting bathymetry model. Additionally, a deeper synthetic layer is created and its forward gravity is calculated. This represents the regional gravity field.

## Run the inversion and ensembles
`synthetic_simple_inversion.ipynb` uses this starting bathymetry and the synthetic gravity data to recover the true bathymetry with an inversion. A series of cross-validated inversions are performed. These include using two methods of calculating the vertical derivative of gravity for the Jacobian matrix, inversion with and without noise, inversions with full resolution vs low-resolution gravity data. Finally, an ensemble of 100 inversions are run with 10 levels of noise in the gravity data and 10 levels of gravity data spacing.

## Repeat with a regional component of gravity
`synthetic_simple_inversion_with_regional.ipynb`. This notebook repeats the same steps as the above notebook, but the input gravity data includes a regional component (resulting from the synthetic crustal layer) that needs to be estimated and removed. We explore various methods to accomplish this, before repeating the various inversions and the ensemble described above.
