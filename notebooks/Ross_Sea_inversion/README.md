# Ross Sea semi-synthetic gravity inversion

This folder contains all the notebooks relevant to performing a gravity inversion to recover the bathymetry for a semi-synthetic model of the Ross Sea. This model uses real bathymetry data from a portion of the Ross Sea acquired from shipborne multibeam bathymetry, compiled by IBCSO. To generate a long-wavelength regional gravity field, real acoustic basement topography data is used from shipborne seismic surveying, compiled from the ANTOSTRAT project. These two real topographic datasets from the Ross Sea are then forward modelled to create synthetic observed gravity data. We refer to this model as *semi-synthetic* since the topography data is real, while the observed gravity is synthetically modelled from the topography data. With this model, we perform a series of inversion to both test the performance of the inversion algorithm, and explore the effects of various factors, such as survey design, data noise, and methodological parameter values. We start with simple idealistic scenarios and from there increase the complexity to simulate more realistic scenarios. Below we outlines each of these scenarios, which are each associated with a Juptyer Notebook in this folder.

## 01 optimal scenario
See notebook here: [01_optimal_scenario.ipynb](01_optimal_scenario.ipynb)

This inversion represents the most basic scenario with the below characteristics:
* full resolution gravity data
    - observed gravity is equal to forward gravity of the true bathymetyry
* no gravity noise
* true density contrast is known
* no regional field

## 02 with a density contrast cross-validation
See notebook here: [02_with_density_CV.ipynb](02_with_density_CV.ipynb)

This inversion builds complexity upon the previous by assuming we don't know what the optimal density contrast of the bathymetry is. We estimate the optimal density contrast using a cross-validation technique.

## 03a with an airborne survey
See notebook here: [03a_with_airborne_survey.ipynb](03a_with_airborne_survey.ipynb)

This inversion builds upon #02 by using a synthetic airborne survey, instead of a full resolution observed gravity. To do this, we sample the forward gravity of the true bathymetry along flight line paths. With the flight line data, we re-grid over the entire region. This allows to test the effects of the loss of data with limited survey coverage.

##
See notebook here: [03b_with_noise.ipynb](03b_with_noise.ipynb)

This inversion builds upon #02 by adding synthetic random noise to the gravity data. It uses the full resolution gravity data, *not** the airborne survey, but adds noise to it. The noise has both short-wavelength and long-wavelength components, simulating both noise data a
##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

##
See notebook here: [.ipynb](.ipynb)

## Synthetic gravity and starting bathymetry model
`Ross_Sea_synthetic_model.ipynb` downloads IBCSO bathymetry and ANTOSTRAT basement data for the Ross Sea. From this, the synthetic observed gravity data is created from the forward gravity calculations of these grids. A synthetic airborne survey is created by sampling this gravity data onto flight paths and adding noise. To simulate having limited knowledge of the bathymetry prior to the inversion, a low-resolution version of the IBCSO data is created. A semi-regular grid of constraint points is used to sampled the IBCSO values inside of a synthetic ice shelf border. Outside of this ice shelf border the full-resolution IBCSO grid is used. The entire region is then interpolated creating the starting bathymetry model.

## Run the inversion
`Ross_Sea_synthetic_inversion.ipynb` uses this starting bathymetry and the synthetic airborne gravity survey data to recover the Ross Sea bathymetry with an inversion. First the regional component of gravity (resulting from the basement surface) is estimated and removed. Then a series of cross-validated inversions are performed. These include with the full-resolution gravity data (not sampled onto flight lines), both with and without noise, and the synthetic airborne survey data, both with and without noise.

## Run ensembles of inversions
`Ross_Sea_ensemble.ipynb` runs two ensemble of inversions. The first tests the effects on the inverted bathymetry resulting from flight line spacing (resolution) and the level of noise in the gravity data. This runs 100 inversions from the combination of 10 levels of noise in the gravity data and 10 values of flight line spacing. The inverted bathymetries are then compared to the full resolution IBCSO bathymetry to assess their performance. The second ensemble tests the effects of the spacing of constraint points within the ice shelf. For this, 10 inversions were performed using various numbers of equally spaced constraints within the ice shelf.

## Run the Monte Carlo simulations
`Ross_Sea_montecarlo_uncertainties.ipynb` performs several Monte Carlo simulations. The "full" simulation involves the random sampling of the gridding gravity disturbance data and constraint point depths, and the Latin Hypercube sampling of the density contrast of the seafloor. With these sampled inversion inputs, 20 full inversion workflows are run (create starting bed, remove the regional component, perform cross-validated inversion) and the resulting 20 bathymetry models are used to calculate cell-specific statistics. This Monte Carlo simulation is repeated with the individual sampling of 1) the gravity data, 2) the constraint points, and 3) the density contrast value. These additional simulation reveal the components of the total uncertainties which results from these aspects of the inversion.
