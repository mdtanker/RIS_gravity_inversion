# Bathymetry gravity inversion

Here we present a gravity inversion algorithm for modelling bathymetry. This is a non-linear geometric regularized least-squares inversion. Pre-existing bathymetry measurements can be used to constrain the inversion, and a Bayesian approach, via Monte Carlo simulation, is used to estimate uncertainties and sensitivity of the inversion to the various input data and parameters.

The inversion code in `RIS_gravity_inversion` has mostly been migrated to a separate Python package, [Invert4Geom](https://github.com/mdtanker/invert4geom), while some specific functions for the synthetic tests and specific bathymetry applications are retained here.

This inversion was developed as part of my PhD thesis. Chapter 3 of the thesis tests the inversion on a suite of synthetic and semi-synthetic models. The relevant Jupyter notebooks for this are in `notebooks/synthetic_inversion` and `notebooks/Ross_Sea_inversion`.

Chapter 4 of the thesis uses the inversion to model the bathymetry beneath Antarctica's Ross Ice Shelf. The relevant Jupyter notebooks for this are in `notebooks/Ross_Ice_Shelf_inversion`. This includes notebooks for levelling and reducing the airborne gravity data.

Below are instructions for using this repository.

## Getting the code

You can download a copy of all the files for this project by cloning the GitHub repository:

    git clone https://github.com/mdtanker/RIS_gravity_inversion

## Dependencies

These instructions assume you have `Make` installed. If you don't you can just open up the `Makefile` file and copy and paste the commands into your terminal. This also assumes you have Python installed.

Install the required dependencies with either `conda` or `mamba`:

    cd RIS_gravity_inversion

    make conda_install

Activate the newly created environment:

    conda activate RIS_gravity_inversion

Install the local project

    make install


## Run the inversion

The various Jupyter notebooks and `README` files in the folder `notebooks` should explain how to use this inversion.
