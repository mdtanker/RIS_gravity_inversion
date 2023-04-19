import contextlib
import copy
import os
import pathlib
import pickle
import string
import sys
import warnings
from getpass import getpass
from typing import Union

import geopandas as gpd
import harmonica as hm
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pygmt
import seaborn as sns
import verde as vd
import xarray as xr
import xrft
import zarr
from antarctic_plots import fetch, maps, profile, utils
from dotenv import load_dotenv
from optuna.storages import JournalFileStorage, JournalStorage
from pykdtree.kdtree import KDTree
from requests import get
from sklearn.metrics import mean_squared_error
from tqdm.autonotebook import tqdm

from RIS_gravity_inversion import gravity_processing as grav
from RIS_gravity_inversion import inversion as inv
from RIS_gravity_inversion import optimization, plotting, regional

load_dotenv()


def bedmachine_buffer_points(
    buffer_width=10e3,
    bedmachine_bed=None,
    plot=False,
):
    # get RIS mask
    measures_shelves = fetch.measures_boundaries(version="IceShelf")
    ice_shelves = gpd.read_file(measures_shelves)
    RIS_seperate = ice_shelves[ice_shelves.NAME.isin(["Ross_West", "Ross_East"])]
    RIS = RIS_seperate.dissolve()

    # get buffer RIS mask
    RIS_buffer = RIS.buffer(buffer_width)

    # mask bedmachine bed inside of RIS
    bedmachine_masked = (
        utils.mask_from_shp(
            shapefile=RIS,
            xr_grid=bedmachine_bed,
            masked=True,
        )
        .rename({"x": "easting", "y": "northing"})
        .rename("upward")
    )

    # mask bedmachine bed outside of buffered RIS
    bedmachine_masked_buffer = utils.mask_from_shp(
        shapefile=RIS_buffer,
        xr_grid=bedmachine_masked,
        masked=True,
        invert=False,
    ).rename("upward")

    # create dataframes from grids
    bedmachine_df_outside = vd.grid_to_table(bedmachine_masked).dropna()
    bedmachine_df_buffer = vd.grid_to_table(bedmachine_masked_buffer).dropna()

    if plot is True:
        grav.plotly_points(
            bedmachine_df_buffer,
            color_col="upward",
            robust=True,
            point_size=4,
        )

    return bedmachine_df_buffer, bedmachine_df_outside, bedmachine_masked


def create_starting_bed(
    buffer_points,
    inside_constraints,
    masked_bed,
    region,
    spacing,
    icebase,
    plot=False,
):
    points = pd.concat((buffer_points, inside_constraints))

    grd = vd.Spline(mindist=1e-5, damping=None)
    coords = (points.easting, points.northing)

    grd.fit(coords, points.upward)

    inner_bed = grd.grid(
        region=region,
        spacing=spacing,
    ).scalars

    bed_from_constraints = xr.where(
        masked_bed.isnull(),
        inner_bed,
        masked_bed,
    )

    # ensure bed doesn't cross ice base
    bed_from_constraints = xr.where(
        bed_from_constraints > icebase,
        icebase,
        bed_from_constraints,
    )

    if plot is True:
        fig = maps.plot_grd(
            bed_from_constraints,
            points=inside_constraints.rename(columns={"easting": "x", "northing": "y"}),
        )
        fig.show()

    return bed_from_constraints


def inversion_varying_gravity(
    gravity_df,
    anomaly_args,
    inversion_args,
):
    """
    Run all portions of the inversion workflow which rely on the gravity data and
    return the resulting inverted bathymetry. The workflow components include correcting
      the observed gravity with the partial bouguer corrections, removing the forward
      gravity of the bed, seperating the regional/residual fields, and running the
      inversion. Supplied gravity dataframe should alrady contain the forward gravities
      of the surface, icebase, and starting bed.
    """
    grav = gravity_df.copy()

    # apply partial bouguer corrections
    grav["Gobs_corr"] = grav.Gobs - grav.surface_forward - grav.icebase_forward

    # apply Gobs shift, calculate misfit and seperate the regional field
    anomalies = regional.regional_seperation(
        input_grav=grav,
        input_forward_column="bed_forward",
        input_grav_column="Gobs_corr",
        regional_method="constraints",
        tension_factor=0.25,
        **anomaly_args,
    )

    # run inversion
    result = inv.geo_inversion(input_grav=anomalies, **inversion_args)

    # get final topography
    final_topo = (
        result[0].set_index(["northing", "easting"]).to_xarray()[result[0].columns[-1]]
    )

    return final_topo


def inversion_varying_constraints(
    constraints,
    gravity_df,
    weights_grid,
    masked_bed,
    buffer_points,
    starting_bed_args,
    anomaly_args,
    inversion_args,
):
    """
    Run all portions of the inversion workflow which rely on constraints and return the
    resulting inverted bathymetry. The workflow components include create the starting
    bed model from the constraints, calculating the forward gravity of the bed,
    seperating the regional/residual fields, and running the inversion. Supplied
    gravity dataframe should have been preprocessed, with partial bouguer corrections
    and DC shift.
    """
    grav = gravity_df.copy()

    # create starting bathymetry
    starting_bed = create_starting_bed(
        inside_constraints=constraints[constraints.inside],
        masked_bed=masked_bed,
        buffer_points=buffer_points,
        **starting_bed_args,
    )

    # calculate and remove forward gravity of starting bed
    water_density = 1024  # +/- 5, from Griggs and Bamber 2009/2011
    sediment_density = 2300
    density = sediment_density - water_density
    starting_bed_prisms = grids_to_prisms(
        surface=starting_bed,
        reference=starting_bed.values.mean(),
        density=xr.where(starting_bed >= starting_bed.values.mean(), density, -density),
        input_coord_names=["easting", "northing"],
    )
    bed_grav_grid, bed_grav_df = forward_grav_of_prismlayer(
        [starting_bed_prisms],
        grav[grav.Gobs.notna()],
        names=["bed_prisms"],
        progressbar=False,
        plot=False,
    )
    grav["bed_forward"] = bed_grav_df.forward_total

    # apply Gobs shift, calculate misfit and seperate the regional field
    anomalies = regional.regional_seperation(
        constraints=constraints,
        input_grav=grav,
        input_forward_column="bed_forward",
        input_grav_column="Gobs_corr",
        regional_method="constraints",
        tension_factor=0.25,
        **anomaly_args,
    )

    starting_bed_prisms["weights"] = weights_grid

    # run inversion
    result = inv.geo_inversion(
        input_grav=anomalies, prism_layer=starting_bed_prisms, **inversion_args
    )

    # get final topography
    final_topo = (
        result[0].set_index(["northing", "easting"]).to_xarray()[result[0].columns[-1]]
    )

    return final_topo


def recalculate_partial_bouguer_anomalies(
    gravity,
    air_density,
    ice_density,
    water_density,
    sediment_density,
    surface_grid,
    icebase_grid,
):
    grav = gravity.copy()

    # surface
    density = ice_density - air_density
    surface_prisms = grids_to_prisms(
        surface=surface_grid,
        reference=surface_grid.values.mean(),
        density=xr.where(surface_grid >= surface_grid.values.mean(), density, -density),
        input_coord_names=["x", "y"],
    )
    surface_grav_grid, surface_grav_df = forward_grav_of_prismlayer(
        [surface_prisms],
        grav[grav.Gobs.notna()],
        names=["surface_prisms"],
        progressbar=False,
        plot=False,
    )
    grav["surface_forward"] = surface_grav_df.forward_total

    # icebase
    density = water_density - ice_density
    icebase_prisms = grids_to_prisms(
        surface=icebase_grid,
        reference=icebase_grid.values.mean(),
        density=xr.where(icebase_grid >= icebase_grid.values.mean(), density, -density),
        input_coord_names=["x", "y"],
    )
    icebase_grav_grid, icebase_grav_df = forward_grav_of_prismlayer(
        [icebase_prisms],
        grav[grav.Gobs.notna()],
        names=["icebase_prisms"],
        progressbar=False,
        plot=False,
    )
    grav["icebase_forward"] = icebase_grav_df.forward_total

    # apply partial bouguer corrections
    grav["Gobs_corr"] = grav.Gobs - grav.surface_forward - grav.icebase_forward

    return grav


def recreate_bed(
    constraints,
    bed_buffer_points,
    region,
    spacing,
    icebase,
):
    # bed inside RIS masked
    masked_bed = (
        constraints[~constraints.inside]
        .set_index(["northing", "easting"])
        .to_xarray()
        .upward
    )

    # points within buffer zone
    buffer_points = pd.merge(
        constraints.reset_index(),
        bed_buffer_points[["northing", "easting"]],
        how="inner",
    ).set_index("index")

    # create starting bathymetry
    starting_bed = create_starting_bed(
        buffer_points=buffer_points,
        inside_constraints=constraints[constraints.inside],
        masked_bed=masked_bed,
        region=region,
        spacing=spacing,
        icebase=icebase,
    )

    return starting_bed


def recalc_bed_grav(
    gravity,
    starting_bed,
    water_density,
    sediment_density,
):
    grav = gravity.copy()

    # calculate and remove forward gravity of starting bed
    density = sediment_density - water_density
    starting_bed_prisms = grids_to_prisms(
        surface=starting_bed,
        reference=starting_bed.values.mean(),
        density=xr.where(starting_bed >= starting_bed.values.mean(), density, -density),
        input_coord_names=["easting", "northing"],
    )
    bed_grav_grid, bed_grav_df = forward_grav_of_prismlayer(
        [starting_bed_prisms],
        grav[grav.Gobs.notna()],
        names=["bed_prisms"],
        progressbar=False,
        plot=False,
    )
    grav["bed_forward"] = bed_grav_df.forward_total

    return grav, starting_bed_prisms


def monte_carlo_inversion_uncertainty_loop(
    fname,
    runs,
    grav,
    constraints,
    sample_grav=False,
    sample_constraints=False,
    sample_tension_factor=False,
    sample_ice_density=False,
    sample_water_density=False,
    sample_sediment_density=False,
    sample_inner_bound=False,
    sample_outer_bound=False,
    sample_solver_damping=False,
    inversion_args=None,
    sampled_args=None,
    plot=False,
    mask=None,
    title="Monte Carlo simulation",
    **kwargs,
):
    # if file exists, start and next run, else start at 0
    try:
        topos = xr.open_zarr(fname)
        starting_run = len(topos)
    except FileNotFoundError:
        print(f"File {fname} not found, creating new file\n")
        # create / overwrite pickle files
        with open(f"{fname[:-5]}_params.pickle", "wb") as _:
            pass
        with open(f"{fname[:-5]}_sampled_values.pickle", "wb") as _:
            pass
        with open(f"{fname[:-5]}_grav_dfs.pickle", "wb") as _:
            pass
        starting_run = 0

    if starting_run == runs:
        print("all runs already complete, loading results from file.")
    for i in tqdm(range(starting_run, runs)):
        if i == starting_run:
            print(
                f"starting Monte Carlo uncertainty analysis at run {starting_run} of"
                f"{runs}\nsaving results to {fname}\n"
            )

        # define unsampled values
        tension_factor = 0.25
        ice_density = 873.5  # 830 -917, firn closoff to max ice density
        water_density = 1024
        sediment_density = 2300
        inner_bound = 5e3
        outer_bound = 20e3
        solver_damping = 0.1

        # create random generator
        rand = np.random.default_rng(seed=i)

        recalculate_bouguer_anomalies = False
        recalculate_starting_bed = False
        recalculate_bed_gravity = False
        recalculate_regional = False
        recalculate_weights = False

        sampled_args = {}

        # if gravity resampled, must recalculate regional
        if sample_grav is True:
            sampled_grav = grav.copy()
            Gobs_sampled = rand.normal(sampled_grav.Gobs, sampled_grav.uncert)
            Gobs_corr_sampled = rand.normal(sampled_grav.Gobs_corr, sampled_grav.uncert)
            sampled_grav["Gobs"] = Gobs_sampled
            sampled_grav["Gobs_corr"] = Gobs_corr_sampled
            gravity_df = sampled_grav.copy()
            recalculate_regional = True
        elif sample_grav is False:
            gravity_df = grav.copy()

        # if constraints resampled, must recalculate starting bed,
        # recalculate bed gravity, and recalculate regional
        if sample_constraints is True:
            sampled_constraints = constraints.drop(columns="upward").copy()
            sampled_constraints["upward"] = rand.uniform(
                sampled_constraints.low_lim, sampled_constraints.high_lim
            )
            constraint_points = sampled_constraints.copy()
            recalculate_starting_bed = True
            recalculate_bed_gravity = True
            recalculate_regional = True
        elif sample_constraints is False:
            constraint_points = constraints.copy()

        # if tension factor resampled, must recalculate region
        if sample_tension_factor is True:
            sampled_args["tension_factor"] = rand.uniform(0, 0.99)
            recalculate_regional = True
        elif sample_tension_factor is False:
            sampled_args["tension_factor"] = tension_factor

        # if ice density resampled, must recalculate bouguer anomalies and regional
        if sample_ice_density is True:
            sampled_args["ice_density"] = rand.normal(
                873.5, 5
            )  # 5 from Hawley et al. 2004
            recalculate_bouguer_anomalies = True
            recalculate_regional = True
        elif sample_ice_density is False:
            sampled_args["ice_density"] = ice_density

        # if water density resampled, must recalculate bouguer anomalies,
        # bed gravity and regional
        if sample_water_density is True:
            sampled_args["water_density"] = rand.normal(1024, 5)
            recalculate_bouguer_anomalies = True
            recalculate_bed_gravity = True
            recalculate_regional = True
        elif sample_water_density is False:
            sampled_args["water_density"] = water_density

        # if sediment density resampled, must recalculate bed gravity and regional
        if sample_sediment_density is True:
            sampled_args["sediment_density"] = rand.normal(2300, 400)
            recalculate_bed_gravity = True
            recalculate_regional = True
        elif sample_sediment_density is False:
            sampled_args["sediment_density"] = sediment_density

        # if inner bounds are resampled, must recalculate weights grid
        if sample_inner_bound is True:
            sampled_args["inner_bound"] = rand.uniform(1e3, 10e3)
            recalculate_weights = True
        elif sample_inner_bound is False:
            sampled_args["inner_bound"] = inner_bound

        # if outer bounds are resampled, must recalculate weights grid
        if sample_outer_bound is True:
            sampled_args["outer_bound"] = rand.uniform(10e3, 30e3)
            recalculate_weights = True
        elif sample_outer_bound is False:
            sampled_args["outer_bound"] = outer_bound

        # if solver damping is resampled, include in inversion args
        if sample_solver_damping is True:
            sampled_args["solver_damping"] = rand.uniform(0, 1)
        elif sample_solver_damping is False:
            sampled_args["solver_damping"] = solver_damping

        if recalculate_starting_bed is True:
            recalculate_bed_gravity = True
            recalculate_regional = True
        if recalculate_bouguer_anomalies is True:
            recalculate_regional = True
        if recalculate_bed_gravity is True:
            recalculate_regional = True

        if i == starting_run:
            verbose = None
        else:
            verbose = "q"

        # run inversion
        topo, results = monte_carlo_inversion_uncertainty(
            constraints=constraint_points,
            gravity_df=gravity_df,
            recalculate_bouguer_anomalies=recalculate_bouguer_anomalies,
            recalculate_starting_bed=recalculate_starting_bed,
            recalculate_bed_gravity=recalculate_bed_gravity,
            recalculate_regional=recalculate_regional,
            recalculate_weights=recalculate_weights,
            inversion_args=inversion_args,
            sampled_args=sampled_args,
            verbose=verbose,
            **kwargs,
        )

        # rename grid with run number
        topo = topo.to_dataset(name=f"run_{i}")

        if i == 0:
            mode = "w"
        else:
            mode = "a"

        enc = {x: {"compressor": zarr.Blosc()} for x in topo}
        topo.to_zarr(
            fname,
            encoding=enc,
            mode=mode,
        )

        # get gravity data
        grav_df = results[1]

        # get parameters
        params = results[2]

        # save parameters
        with open(f"{fname[:-5]}_params.pickle", "ab") as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{fname[:-5]}_sampled_values.pickle", "ab") as file:
            pickle.dump(sampled_args, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{fname[:-5]}_grav_dfs.pickle", "ab") as file:
            pickle.dump(grav_df, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Finished inversion {i+1} of {runs}")

    # load pickle file
    params = []
    with open(f"{fname[:-5]}_params.pickle", "rb") as file:
        while 1:
            try:
                params.append(pickle.load(file))
            except EOFError:
                break
    values = []
    with open(f"{fname[:-5]}_sampled_values.pickle", "rb") as file:
        while 1:
            try:
                values.append(pickle.load(file))
            except EOFError:
                break
    grav_dfs = []
    with open(f"{fname[:-5]}_grav_dfs.pickle", "rb") as file:
        while 1:
            try:
                grav_dfs.append(pickle.load(file))
            except EOFError:
                break

    # get final gravity residual RMSE of each model
    residuals = [utils.RMSE(df[list(df.columns)[-1]]) for df in grav_dfs]

    # load dataset with all the topo models
    topos = xr.open_zarr(fname)

    # get stats on the ensemble
    stats = model_ensemble_stats(topos, weights=residuals)
    merged, z_mean, z_stdev, weighted_mean, weighted_stdev = stats

    if plot is True:
        grd = weighted_stdev
        if mask is not None:
            grd = utils.mask_from_shp(
                shapefile=mask, xr_grid=grd, masked=True, invert=False
            )
        fig = maps.plot_grd(
            grd,
            cmap="inferno",
            reverse_cpt=True,
            robust=True,
            hist=True,
            cbar_label="weighted standard deviation (m)",
            title=title,
        )
        fig.plot(
            x=constraints.easting,
            y=constraints.northing,
            fill="black",
            style="c.01c",
        )
        fig.plot(
            x=constraints[constraints.inside].easting,
            y=constraints[constraints.inside].northing,
            fill="white",
            pen=".3p,black",
            style="c.08c",
            label="Bed constraints",
        )
        fig.legend()
        fig.show()

    return topos, params, values, grav_dfs


def model_ensemble_stats(
    dataset,
    weights=None,
):
    da_list = [dataset[i] for i in list(dataset)]

    merged = (
        xr.concat(da_list, dim="runs")
        .assign_coords({"runs": list(dataset)})
        .rename("run_num")
        .to_dataset()
    )

    z_mean = merged["run_num"].mean("runs").rename("z_mean")

    z_stdev = merged["run_num"].std("runs").rename("z_std")

    if weights is not None:
        assert len(da_list) == len(weights)

        weighted_mean = sum(g * w for g, w in zip(da_list, weights)) / sum(weights)

        # from https://stackoverflow.com/questions/30383270/how-do-i-calculate-the-standard-deviation-between-weighted-measurements # noqa
        weighted_var = (
            sum(w * (g - weighted_mean) ** 2 for g, w in zip(da_list, weights))
        ) / sum(weights)
        weighted_stdev = np.sqrt(weighted_var)

    else:
        weighted_mean = None
        weighted_stdev = None

    return merged, z_mean, z_stdev, weighted_mean, weighted_stdev


def monte_carlo_inversion_uncertainty(
    constraints,
    gravity_df,
    recalculate_bouguer_anomalies=False,
    recalculate_starting_bed=False,
    recalculate_bed_gravity=False,
    recalculate_regional=False,
    recalculate_weights=False,
    inversion_args=None,
    sampled_args=None,
    verbose=None,
    **kwargs,
):
    """
    Run all portions of the inversion workflow which rely on constraints or gravity and
    return the resulting inverted bathymetry.

    to include layer densities in the MC analysis provide densities and grids in
    `forward_grav_args` else, column `Gobs_corr` must be in gravity dataframe

    to include constraints in the MC analysis, provide "constraint_args", else,
    starting_bed must be provided and "bef_forward" must be in the gravity dataframe

    to include gravity data in the MC analysis, provide "gravity_args", else,


    """

    grav = gravity_df.copy()

    # re-calculate surface / icebase gravities
    if recalculate_bouguer_anomalies is False:
        assert all(item in grav.columns for item in ["Gobs_corr"]), (
            "if recalculate_bouguer_anomalies is False,"
            " 'Gobs_corr' must be in gravity dataframe"
        )
    else:
        if verbose != "q":
            print("Recalculating bouguer anomalies")
        grav.drop(
            columns=[
                "surface_forward",
                "icebase_forward",
                "Gobs_corr",
                "misfit",
                "reg",
                "res",
            ],
            inplace=True,
            errors="ignore",
        )
        with HiddenPrints():
            grav = recalculate_partial_bouguer_anomalies(
                gravity=grav,
                air_density=1,
                ice_density=sampled_args.get("ice_density"),
                water_density=sampled_args.get("water_density"),
                sediment_density=sampled_args.get("sediment_density"),
                surface_grid=kwargs.get("surface_fullres"),
                icebase_grid=kwargs.get("icebase_fullres"),
            )

    # re-create starting bed
    if recalculate_starting_bed is False:
        starting_bed = kwargs.get("starting_bed", None)
    else:
        # recalculate the starting bed with the provided constraints
        if verbose != "q":
            print("Recalculating starting bed")
        with HiddenPrints():
            starting_bed = recreate_bed(
                constraints=constraints,
                bed_buffer_points=kwargs.get("bed_buffer_points"),
                region=kwargs.get("buffer_region"),
                spacing=kwargs.get("layer_spacing"),
                icebase=kwargs.get("icebase_lowres"),
            )
        recalculate_bed_gravity = True

    # recalculate bed gravity
    if recalculate_bed_gravity is False:
        assert all(item in grav.columns for item in ["bed_forward"]), (
            "if recalculate_bed_gravity is False, 'bed_forward' must be in gravity"
            "dataframe"
        )
        starting_bed_prisms = kwargs.get("starting_bed_prisms", None)
        if starting_bed_prisms is None:
            raise TypeError(
                (
                    "If recalculate_bed_gravity is False, must "
                    "provide prisms layer with `starting_bed_prisms`"
                )
            )
    else:
        if verbose != "q":
            print("Recalculating bed gravity")
        grav.drop(
            columns=["bed_forward", "misfit", "reg", "res"],
            inplace=True,
            errors="ignore",
        )
        with HiddenPrints():
            grav, starting_bed_prisms = recalc_bed_grav(
                gravity=grav,
                starting_bed=starting_bed,
                water_density=sampled_args.get("water_density"),
                sediment_density=sampled_args.get("sediment_density"),
            )
    # recalculate regional seperation
    if recalculate_regional is False:
        assert all(item in grav.columns for item in ["Gobs_corr_shift", "res"]), (
            "if recalculate_regional is False, 'Gobs_corr_shift'"
            " and 'res' must be in gravity dataframe"
        )
        anomalies = grav
    else:
        if verbose != "q":
            print("Recalculating regional field")
        grav.drop(columns=["res", "reg", "misfit"], inplace=True, errors="ignore")
        with HiddenPrints():
            anomalies = regional.regional_seperation(
                input_grav=grav,
                grav_spacing=kwargs.get("grav_spacing"),
                regional_method="constraints",
                constraints=constraints,
                input_forward_column="bed_forward",
                input_grav_column="Gobs_corr",
                inversion_region=kwargs.get("inversion_region"),
                tension_factor=sampled_args.get("tension_factor"),
            )

    # recalculate weights grid
    if recalculate_weights is False:
        weights_grid = kwargs.get("weights_grid", None)
        if (inversion_args.get("apply_weights") is True) & (weights_grid is None):
            raise TypeError(
                "If recalculate_weights is False, must provide grid with kwarg"
                "`weights_grids`"
            )
    else:
        if verbose != "q":
            print("Recalculating weights grid")
        with HiddenPrints():
            weights_grid, _ = constraints_grid(
                constraint_points=constraints,
                grid=starting_bed_prisms,
                inner_bound=sampled_args.get("inner_bound"),
                outer_bound=sampled_args.get("outer_bound"),
                low=0,
                high=1,
                between=np.nan,
                region=kwargs.get("inversion_region"),
                interp_type="spline",
                efficient_interp=True,
                efficient_interp_dist=kwargs.get("layer_spacing"),
                epsg=None,
                plot=False,
            )

    # add weights grid to starting prisms
    if inversion_args.get("apply_weights") is True:
        starting_bed_prisms["weights"] = weights_grid
    if inversion_args.get("weights_after_solving") is True:
        starting_bed_prisms["weights"] = weights_grid

    # run inversion
    with HiddenPrints():
        results = inv.geo_inversion(
            input_grav=anomalies,
            prism_layer=starting_bed_prisms,
            solver_damping=sampled_args.get("solver_damping"),
            **inversion_args,
        )

    # get final topography
    final_topo = (
        results[0]
        .set_index(["northing", "easting"])
        .to_xarray()[results[0].columns[-1]]
    )

    return final_topo, results


def split_gravity_test_train(
    grav_df,
    spacing=None,
    shape=None,
    n_splits=6,
    test_size=0.1,
    random_state=10,
    plot=False,
    number_to_plot=None,
):
    df = grav_df.copy()

    # kfold = vd.BlockKFold(
    #     spacing=spacing,
    #     shape=shape,
    #     n_splits=n_splits,
    #     shuffle=True,
    #     random_state=random_state,
    # )
    kfold = vd.BlockShuffleSplit(
        spacing=spacing,
        shape=shape,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
    )

    coords = (df.easting, df.northing)
    feature_matrix = np.transpose(coords)
    shape = coords[0].shape
    mask = np.full(shape=shape, fill_value="     ")

    for iteration, (train, test) in enumerate(kfold.split(feature_matrix)):
        mask[np.unravel_index(train, shape)] = "train"
        mask[np.unravel_index(test, shape)] = " test"
        df = pd.concat(
            [df, pd.DataFrame({f"fold_{iteration}": mask}, index=df.index)], axis=1
        )

    df_test_train = df.copy()

    if plot is True:
        if number_to_plot is None:
            _, ncols = utils.square_subplots(n_splits)
            df = df_test_train.copy()
        else:
            n_splits = number_to_plot
            _, ncols = utils.square_subplots(n_splits)
            fold_cols = list(
                df_test_train.columns[df_test_train.columns.str.startswith("fold_")]
            )
            df = df_test_train.drop(columns=fold_cols[number_to_plot:]).copy()

        for i in range(n_splits):
            if i == 0:
                fig = (None,)
                origin_shift = "initialize"
                xshift_amount = None
                yshift_amount = None
            elif i % ncols == 0:
                fig = fig
                origin_shift = "both_shift"
                xshift_amount = -ncols + 1
                yshift_amount = -1
            else:
                fig = fig
                origin_shift = "xshift"
                xshift_amount = 1
                yshift_amount = 1

            df_test = df[df[f"fold_{i}"] == " test"]
            df_train = df[df[f"fold_{i}"] == "train"]

            fig = maps.basemap(
                region=vd.get_region((df.easting, df.northing)),
                title=f"Fold {i} ({len(df_test)} testing points)",
                coast=True,
                origin_shift=origin_shift,
                xshift_amount=xshift_amount,
                yshift_amount=yshift_amount,
                fig=fig,
            )

            fig.plot(
                x=df_train.easting,
                y=df_train.northing,
                style="c.05c",
                fill="blue",
                label="Train",
            )
            fig.plot(
                x=df_test.easting,
                y=df_test.northing,
                style="c.1c",
                fill="red",
                label="Test",
            )

            fig.legend()
        fig.show()

    return df_test_train


def split_constraints_test_train(
    constraints,
    spacing=None,
    shape=None,
    n_splits=6,
    test_size=0.1,
    random_state=10,
    plot=False,
    number_to_plot=None,
):
    df = constraints[constraints.inside].copy()

    kfold = vd.BlockKFold(
        spacing=spacing,
        shape=shape,
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    # kfold = vd.BlockShuffleSplit(
    #     spacing=spacing,
    #     shape=shape,
    #     n_splits=n_splits,
    #     test_size=test_size,
    #     random_state = random_state,
    # )

    coords = (df.easting, df.northing)
    feature_matrix = np.transpose(coords)
    shape = coords[0].shape
    mask = np.full(shape=shape, fill_value="     ")

    for iteration, (train, test) in enumerate(kfold.split(feature_matrix)):
        mask[np.unravel_index(train, shape)] = "train"
        mask[np.unravel_index(test, shape)] = " test"
        df = pd.concat(
            [df, pd.DataFrame({f"fold_{iteration}": mask}, index=df.index)], axis=1
        )

    merged_df = merge_test_train_to_outside(df, constraints)

    if plot is True:
        if number_to_plot is None:
            _, ncols = utils.square_subplots(n_splits)
            df = merged_df.copy()
        else:
            n_splits = number_to_plot
            _, ncols = utils.square_subplots(n_splits)
            fold_cols = list(
                merged_df.columns[merged_df.columns.str.startswith("fold_")]
            )
            df = merged_df.drop(columns=fold_cols[number_to_plot:]).copy()

        for i in range(n_splits):
            if i == 0:
                fig = (None,)
                origin_shift = "initialize"
                xshift_amount = None
                yshift_amount = None
            elif i % ncols == 0:
                fig = fig
                origin_shift = "both_shift"
                xshift_amount = -ncols + 1
                yshift_amount = -1
            else:
                fig = fig
                origin_shift = "xshift"
                xshift_amount = 1
                yshift_amount = 1

            df_test = merged_df[merged_df[f"fold_{i}"] == " test"]
            df_train = merged_df[merged_df[f"fold_{i}"] == "train"]

            fig = maps.basemap(
                region=vd.get_region((merged_df.easting, merged_df.northing)),
                title=f"Fold {i} ({len(df_test)} testing points)",
                coast=True,
                origin_shift=origin_shift,
                xshift_amount=xshift_amount,
                yshift_amount=yshift_amount,
                fig=fig,
            )
            # fig.plot(
            #     x=df_train[~df_train.inside].easting,
            #     y=df_train[~df_train.inside].northing,
            #     style = "c.04c",
            #     fill = "blue",
            # )
            fig.plot(
                x=df_train[df_train.inside].easting,
                y=df_train[df_train.inside].northing,
                style="c.1c",
                fill="blue",
                label="Train",
            )
            fig.plot(
                x=df_test.easting,
                y=df_test.northing,
                style="t.3c",
                fill="red",
                label="Test",
            )

            fig.legend()
        fig.show()

    return merged_df


def merge_test_train_to_outside(
    test_train_df,
    constraints,
):
    df = constraints.copy()
    outside_constraints = df[~df.inside].copy()

    fold_col_names = list(
        test_train_df.columns[test_train_df.columns.str.startswith("fold_")]
    )

    # make all outside constaints training points
    cols = {col: "train" for col in fold_col_names}
    new_cols = pd.DataFrame(cols, index=outside_constraints.index)
    outside_constraints = pd.concat([outside_constraints, new_cols], axis=1)

    # merge the outside constraints with the test/train constraints
    merged_df = pd.concat([test_train_df, outside_constraints])

    return merged_df


def inversion_gravity_cross_val_score(
    training_gravity,
    anomaly_args,
    inversion_args,
    calculate_anomalies=False,
):
    """
    Run a full inversion with `training` gravity data and return the resulting
    bathymety model
    """
    # calculate misfit and seperate regional
    if calculate_anomalies is True:
        with HiddenPrints():
            anomalies = regional.regional_seperation(
                input_grav=training_gravity, **anomaly_args
            )
    else:
        anomalies = training_gravity.copy()

    # run inversion
    with HiddenPrints():
        result = inv.geo_inversion(input_grav=anomalies, **inversion_args)

    # get final topography
    final_topo = (
        result[0].set_index(["northing", "easting"]).to_xarray()[result[0].columns[-1]]
    )

    # fig = maps.plot_grd(final_topo)
    # fig.show()

    return final_topo


def inversion_cross_val_score(
    test_constraints,
    train_constraints,
    starting_prisms,
    anomaly_args,
    weights_args,
    inversion_args,
):
    """
    Run a full inversion with `train` constraints and return the score of the `test`
    constraints.
    """

    # apply Gobs shift, calculate misfit and seperate the regional field
    anomalies = regional.regional_seperation(
        constraints=train_constraints,
        **anomaly_args,
    )

    # create constraints weighting grid
    weights, min_dist = constraints_grid(
        train_constraints, starting_prisms, **weights_args
    )

    starting_prisms["weights"] = weights
    starting_prisms["min_dist"] = min_dist

    # run inversion
    result = inv.geo_inversion(
        input_grav=anomalies, prism_layer=starting_prisms, **inversion_args
    )

    # get final topography
    final_topo = (
        result[0].set_index(["northing", "easting"]).to_xarray()[result[0].columns[-1]]
    )

    score = inversion_constraint_score(test_constraints, final_topo)

    return score, result, final_topo


def inversion_constraint_score(
    constraints,
    final_topo,
):
    """
    Find the difference between the final topography and the constraint point depths.
    """
    # sample the final topography into the constraints df
    df = profile.sample_grids(
        constraints,
        final_topo,
        "predicted",
        coord_names=("easting", "northing"),
    )

    # calculate the RMSE
    score = mean_squared_error(df.upward, df.predicted, squared=False)

    return score


def fetch_private_github_file(
    fname, username="mdtanker", fpath="RIS_grav_bath_data/main", output_dir="/data/"
):
    token = os.environ.get("GITHUB_TOKEN")
    if token is None:
        token = getpass("GITHUB_TOKEN: ")

    res = get(
        f"https://{username}:{token}@raw.githubusercontent.com/{username}/{fpath}/"
        f"{fname}"
    )

    out_file = f"{output_dir}/{fname}"
    with open(out_file, "wb+") as f:
        f.write(res.content)

    return os.path.abspath(out_file)


def RMSE(data):
    """calculate the RMSE of data."""
    return np.sqrt(np.nanmedian(data**2).item())
    # return np.sqrt(np.nanmean(data**2).item())


def nearest_grid_fill(
    grid,
    method="verde",
):
    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    if method == "pygmt":
        filled = pygmt.grdfill(grid, mode="n", verbose="q").rename(original_name)
    elif method == "rioxarray":
        filled = (
            grid.rio.write_crs("epsg:3031")
            .rio.set_spatial_dims(original_dims[1], original_dims[0])
            .rio.write_nodata(np.nan)
            .rio.interpolate_na(method="nearest")
            .rename(original_name)
        )
    elif method == "verde":
        df = vd.grid_to_table(grid)
        df_dropped = df[df[grid.name].notna()]
        coords = (df_dropped[grid.dims[1]], df_dropped[grid.dims[0]])
        region = vd.get_region((df[grid.dims[1]], df[grid.dims[0]]))
        filled = (
            vd.KNeighbors()
            .fit(coords, df_dropped[grid.name])
            .grid(region=region, shape=grid.shape, data_names=original_name)[
                original_name
            ]
        )

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        filled = filled.rename(
            {
                list(filled.dims)[0]: original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )

    return filled


def filter_grid(
    grid,
    filter_width,
    filt_type="lowpass",
    change_spacing=False,
):
    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # get original grid name
    original_name = grid.name

    # if there are nan's, fill them with nearest neighbor
    if grid.isnull().any():
        filled = nearest_grid_fill(grid, method="verde")
        print("filling NaN's with nearest neighbor")
    else:
        filled = grid.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        filled = filled.rename(
            {
                list(filled.dims)[0]: original_dims[0],
                list(filled.dims)[1]: original_dims[1],
            }
        )

    # define width of padding in each direction
    pad_width = {
        original_dims[1]: grid[original_dims[1]].size // 3,
        original_dims[0]: grid[original_dims[0]].size // 3,
    }

    # apply padding
    padded = xrft.pad(filled, pad_width)

    if filt_type == "lowpass":
        filt = hm.gaussian_lowpass(padded, wavelength=filter_width).rename("filt")
    elif filt_type == "highpass":
        filt = hm.gaussian_highpass(padded, wavelength=filter_width).rename("filt")
    else:
        raise ValueError("filt_type must be 'lowpass' or 'highpass'")

    unpadded = xrft.unpad(filt, pad_width)

    # reset coordinate values to original (avoid rounding errors)
    unpadded = unpadded.assign_coords(
        {
            original_dims[0]: grid[original_dims[0]].values,
            original_dims[1]: grid[original_dims[1]].values,
        }
    )

    if change_spacing is True:
        # region = utils.get_grid_info(grid)[1]
        grid = fetch.resample_grid(
            grid,
            spacing=filter_width,
            verbose="q",
        )  # , region = region)
        unpadded = fetch.resample_grid(
            unpadded,
            spacing=filter_width,
            verbose="q",
        )  # , region = region)
    else:
        pass
    if grid.isnull().any():
        result = xr.where(grid.notnull(), unpadded, grid)
    else:
        result = unpadded.copy()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        result = result.rename(
            {
                list(result.dims)[0]: original_dims[0],
                list(result.dims)[1]: original_dims[1],
            }
        )

    return result.rename(original_name)


def dist_nearest_points(
    targets: pd.DataFrame,
    data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    coord_names=["easting", "northing"],
):
    """
    calculate to the distance to the nearest target for a series of points.
    """

    df_targets = targets[[coord_names[0], coord_names[1]]].copy()

    if isinstance(data, pd.DataFrame):
        df_data = data[coord_names].copy()
    elif isinstance(data, xr.DataArray):
        df_grid = vd.grid_to_table(data).dropna()
        df_data = df_grid[[coord_names[0], coord_names[1]]].copy()
    elif isinstance(data, xr.Dataset):
        try:
            df_grid = vd.grid_to_table(data[list(data.variables)[0]]).dropna()
        except IndexError:
            df_grid = vd.grid_to_table(data).dropna()
        df_data = df_grid[[coord_names[0], coord_names[1]]].copy()

    min_dist, _ = KDTree(df_targets.values).query(df_data.values, 1)

    df_data["min_dist"] = min_dist

    if isinstance(data, pd.DataFrame):
        return df_data
    elif isinstance(data, (xr.DataArray, xr.Dataset)):
        return df_data.set_index([coord_names[0], coord_names[1]][::-1]).to_xarray()


def weight_grid(
    grid: xr.DataArray,
    low_lim: float,
    high_lim: float,
    low: float = 0,
    high: float = 1,
    between: float = np.nan,
    interp_type: str = "linear",
    epsg: str = None,
    efficient_interp: bool = False,
    efficient_interp_dist: float = None,
):
    """
    create a grid with values from low to high based on if z is above or below
    thresholds, low_lim and high_lim.

    interp_type can be "linear", "nearest", or "spline"
    efficient_interp: bool float, only fit spline for grid cells nearby to NaNs,
        significantly speeds up interpolation for large grids, by default False
    efficient_interp_dist: float, distance to use for efficient interp, by default is
        same as low_lim.
    """
    grid = grid.copy()

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    # set values above high_lim to high
    # below low_lim to low
    # and between to NaN
    weights = xr.where(grid > high_lim, high, low)
    weights = xr.where((grid <= high_lim) & (grid >= low_lim), np.nan, weights)
    # if NaN's between outer and inner radius, choose interpolation method
    if between is np.nan:
        if interp_type == "linear":
            weights = (
                weights.rio.write_crs(f"epsg:{epsg}")
                .rio.set_spatial_dims(original_dims[1], original_dims[0])
                .rio.write_nodata(np.nan)
                .rio.interpolate_na(method="linear")
            )
        elif interp_type == "nearest":
            weights = nearest_grid_fill(weights, method="verde")
        elif interp_type == "spline":
            # fit entire region with a spline (SLOW FOR BIG GRIDS!)
            if efficient_interp is False:
                df = vd.grid_to_table(weights)
                region = vd.get_region((df[original_dims[1]], df[original_dims[0]]))
                df.dropna(how="any", inplace=True)
                grd = vd.Spline(mindist=1e-5, damping=None)
                grd.fit((df[original_dims[1]], df[original_dims[0]]), df.min_dist)
                weights = grd.grid(region=region, shape=weights.shape).scalars

            # only use spline to fill regions close to NaNs, then merge to rest of grid
            elif efficient_interp is True:
                if efficient_interp_dist is None:
                    efficient_interp_dist = low_lim

                # subset the cells which are nearby NaNs
                weights_df = vd.grid_to_table(weights)
                weights_df = weights_df[weights_df.min_dist.isna()]
                close_to_nans_weights = vd.distance_mask(
                    data_coordinates=(
                        weights_df[original_dims[1]],
                        weights_df[original_dims[0]],
                    ),
                    maxdist=efficient_interp_dist,
                    grid=weights.to_dataset(),
                ).min_dist

                # fit a spline to the subset
                df = vd.grid_to_table(close_to_nans_weights)
                region = vd.get_region((df[original_dims[1]], df[original_dims[0]]))
                df.dropna(how="any", inplace=True)
                grd = vd.Spline(mindist=1e-5, damping=None)
                grd.fit((df[original_dims[1]], df[original_dims[0]]), df.min_dist)
                close_to_nans_weights = grd.grid(
                    region=region, shape=close_to_nans_weights.shape
                ).scalars

                # merge the spline with the rest of the grid
                weights = xr.where(weights.isnull(), close_to_nans_weights, weights)

    # ensure data is between low and high
    weights = xr.where(weights > high, high, weights)
    weights = xr.where(weights < low, low, weights)

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        weights = weights.rename(
            {
                weights.dims[0]: original_dims[0],
                weights.dims[1]: original_dims[1],
            }
        )

    return weights.rename("weights")


def constraints_grid(
    constraint_points: pd.DataFrame,
    grid: Union[xr.DataArray, xr.Dataset],
    inner_bound: float,
    outer_bound: float,
    low: float = 0,
    high: float = 1,
    between: float = np.nan,
    region: list = None,
    interp_type: str = "linear",
    efficient_interp: bool = True,
    efficient_interp_dist: float = None,
    epsg: str = None,
    plot: bool = False,
):
    """
    create a grid of weights based on distance from nearest constraint point
    """

    grid = copy.deepcopy(grid)

    # if a dataset supplied, use first variable as a dataarray
    if isinstance(grid, xr.Dataset):
        grid = grid[list(grid.variables.keys())[0]]

    # get coordinate names
    original_dims = list(grid.sizes.keys())

    constraint_points = constraint_points.copy()

    min_dist = dist_nearest_points(
        targets=constraint_points,
        data=grid,
        coord_names=(original_dims[1], original_dims[0]),
    ).min_dist

    weights = weight_grid(
        min_dist,
        low_lim=inner_bound,
        high_lim=outer_bound,
        between=between,
        interp_type=interp_type,
        low=low,
        high=high,
        epsg=epsg,
        efficient_interp=efficient_interp,
        efficient_interp_dist=efficient_interp_dist,
    )
    # set points outside of region to 0
    if region is not None:
        df = vd.grid_to_table(weights)
        df["are_inside"] = vd.inside(
            (df[original_dims[1]], df[original_dims[0]]),
            region=region,
        )
        weights = df.set_index([original_dims[0], original_dims[1]]).to_xarray()
        weights = xr.where(weights.are_inside, weights.weights, 0)

    if plot is True:
        fig = maps.plot_grd(
            weights,
            cmap="dense",
            points=constraint_points,
            points_style="c0.2c",
        )
        fig.show()

    # reset coordinate names if changed
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="rename '")
        weights = weights.rename(
            {
                weights.dims[0]: original_dims[0],
                weights.dims[1]: original_dims[1],
            }
        )
        min_dist = min_dist.rename(
            {min_dist.dims[0]: original_dims[0], min_dist.dims[1]: original_dims[1]}
        )

    return weights, min_dist


def prep_grav_data(
    grav_df: pd.DataFrame,
    input_grav_name: str,
    input_coord_names: list,
    region: list = None,
):
    """
    prep gravity dataframe to expected format for other functions. We use the same
    conventions as Harmonca, 'easting' and 'northing' for the horizontal coordinates,
    and 'upward' for the vertical coordinate, all in meters. We use 'Gobs' to denote the
     observed gravity, whether its a free-air anomaly, gravity disturbance, or raw
     measured gravity.
    """

    # set standard column names
    grav = grav_df.rename(
        columns={
            input_grav_name: "Gobs",
            input_coord_names[0]: "easting",
            input_coord_names[1]: "northing",
            input_coord_names[2]: "upward",
        },
    )

    # remove points outside of region of interest
    if region is not None:
        grav = utils.points_inside_region(
            grav, region=region, names=["easting", "northing"]
        )

    # center gravity around 0
    # grav.Gobs -= np.median(grav.Gobs)

    grav.Gobs = grav.Gobs.astype(np.float64)
    grav.easting = grav.easting.astype(np.float64)
    grav.northing = grav.northing.astype(np.float64)
    grav.upward = grav.upward.astype(np.float64)

    return grav


def block_reduce_gravity(
    grav_df,
    spacing,
    registration="g",
    method="verde",
):
    # copy dataframe
    grav = grav_df.copy()

    # get number of grav points before reduction
    prior_len = len(grav.Gobs)

    # block reduce gravity data
    if method == "verde":
        grav = utils.block_reduce(
            grav,
            np.median,
            spacing=spacing,
            input_coord_names=["easting", "northing", "upward"],
            input_data_names=["Gobs"],
            center_coordinates=False,
            adjust="spacing",
            drop_coords=False,
        )
    #         reducer = vd.BlockReduce(
    #             reduction=np.median,
    #             spacing=spacing,
    #             center_coordinates=False,  # center of block, or blockmedian of coords
    #             adjust="spacing",  # adjust "spacing" or "region"
    #             drop_coords=False,
    #         )

    #         coordinates, data = reducer.filter(
    #             coordinates=(grav.easting, grav.northing, grav.upward),
    #             data=(grav.Gobs,),
    #         )

    #         grav = pd.DataFrame(
    #             data={
    #                 "easting": coordinates[0],
    #                 "northing": coordinates[1],
    #                 "upward": coordinates[2],
    #                 "Gobs": data[0],
    #             },
    #         )

    elif method == "pygmt":
        blocked = pygmt.blockmedian(
            grav[["easting", "northing", "upward"]],
            spacing=spacing,
            registration=registration,
        )

        blocked["Gobs"] = pygmt.blockmedian(
            grav[["easting", "northing", "Gobs"]],
            spacing=spacing,
            registration=registration,
        ).Gobs

        grav = blocked.copy()

    else:
        raise ValueError("invalid string for Block Reduction type")

    # get number of grav points after reduction
    post_len = len(grav.Gobs)

    print(f"Block-reduced the gravity data at {int(spacing)}m spacing")
    print(f"from {prior_len} points to {post_len} points")

    return grav


def eq_sources_score(params, coordinates, data, delayed=False, **kwargs):
    eqs = hm.EquivalentSources(
        damping=params.get("damping"),
        depth=params.get("depth"),
        **kwargs,
    )
    score = np.mean(
        vd.cross_val_score(
            eqs,
            coordinates,
            data,
            delayed=delayed,
        )
    )

    return score


def eq_sources_GB_score(
    params, coordinates, data, window_size, delayed=False, **kwargs
):
    eqs = hm.EquivalentSourcesGB(
        damping=params.get("damping"),
        depth=params.get("depth"),
        **kwargs,
    )
    score = np.mean(
        vd.cross_val_score(
            eqs,
            coordinates,
            data,
            delayed=delayed,
        )
    )
    return score


def parallel_eq_sources_scores(parameter_sets, coordinates, data, **kwargs):
    #     n_jobs = len(psutil.Process().cpu_affinity())
    #     with tqdm_joblib(
    #         desc="Calculating scores", total=len(parameter_sets)
    #     ) as progress_bar:  # noqa
    #         scores = joblib.Parallel(n_jobs=n_jobs)(
    #             joblib.delayed(eq_sources_score)(i, coordinates, data, **kwargs)
    #             for i in parameter_sets
    #         )
    scores = []
    for i in tqdm(parameter_sets, leave=False, desc="Parameters"):
        score = eq_sources_score(
            i,
            coordinates=coordinates,
            data=data,
            **kwargs,
        )
        scores.append(score)

    return scores


def parallel_eq_sources_GB_scores(parameter_sets, coordinates, data, **kwargs):
    scores = []
    for i in tqdm(parameter_sets, leave=False, desc="Parameters"):
        score = eq_sources_GB_score(
            i,
            coordinates=coordinates,
            data=data,
            **kwargs,
        )
        scores.append(score)

    return scores


def eq_sources_best_param(parameter_sets, coordinates, data, **kwargs):
    scores = parallel_eq_sources_scores(parameter_sets, coordinates, data, **kwargs)

    best = np.argmax(scores)
    print("Best score:", scores[best])
    print("Best parameters:", parameter_sets[best])

    eqs_best = hm.EquivalentSources(**parameter_sets[best], **kwargs).fit(
        coordinates, data
    )

    return eqs_best


def eq_sources_GB_best_param(parameter_sets, coordinates, data, **kwargs):
    scores = parallel_eq_sources_GB_scores(parameter_sets, coordinates, data, **kwargs)

    best = np.argmax(scores)
    print("Best score:", scores[best])
    print("Best parameters:", parameter_sets[best])

    eqs_best = hm.EquivalentSourcesGB(**parameter_sets[best], **kwargs).fit(
        coordinates, data
    )

    return eqs_best


def eq_sources_best(
    parameter_sets, coordinates, data, region, spacing, height=None, **kwargs
):
    """
    Test a suite of damping and depth parameters, pick the best resulting parameters,
    and predict the gravity on a regular grid. Set the observation height to upwards
    continue to, or use the max height of the original data (default).
    """
    eqs_best = eq_sources_best_param(parameter_sets, coordinates, data, **kwargs)

    if height is None:
        height = coordinates[2].max()

    grid_coords = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        extra_coords=height,
    )

    grid = eqs_best.grid(grid_coords, data_names=["predicted_grav"]).predicted_grav

    df = vd.grid_to_table(grid)
    df["upward"] = height

    return eqs_best, grid, df


def eq_sources_GB_best(
    parameter_sets, coordinates, data, region, spacing, height=None, **kwargs
):
    """
    Test a suite of damping and depth parameters, pick the best resulting parameters,
    and predict the gravity on a regular grid. Set the observation height to upwards
    continue to, or use the max height of the orignal data (default).
    """
    eqs_best = eq_sources_GB_best_param(parameter_sets, coordinates, data, **kwargs)

    if height is None:
        height = coordinates[2].max()

    grid_coords = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        extra_coords=height,
    )

    grid = eqs_best.grid(grid_coords, data_names=["predicted_grav"]).predicted_grav

    df = vd.grid_to_table(grid)
    df["upward"] = height

    return eqs_best, grid, df


def optimize_eq_source_params(
    coordinates,
    data,
    n_trials=0,
    damping_limits=[0, 10**3],
    depth_limits=[0, 10e6],
    sampler=None,
    parallel=False,
    fname="tmp",
    use_existing=False,
    plot=False,
    **eq_kwargs,
):
    # set name and storage for the optimization
    study_name = fname
    fname = f"{study_name}.log"

    # remove if exists
    if use_existing is True:
        pass
    else:
        pathlib.Path(fname).unlink(missing_ok=True)
        pathlib.Path(f"{fname}.lock").unlink(missing_ok=True)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="JournalStorage is experimental")
        storage = JournalStorage(JournalFileStorage(fname))

    # if sampler not provided, used BoTorch as default
    if sampler is None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="BoTorch")
            sampler = (
                optuna.integration.BoTorchSampler(n_startup_trials=int(n_trials / 3)),
            )
            # sampler=optuna.samplers.TPESampler(n_startup_trials=int(n_trials/3)),
            # sampler=optuna.samplers.GridSampler(search_space),

    # create study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # define the objective function
    objective = optimization.optimal_eq_source_params(
        coordinates=coordinates,
        data=data,
        damping_limits=damping_limits,
        depth_limits=depth_limits,
        **eq_kwargs,
    )

    if n_trials == 0:  # (use_existing is True) & (n_trials is None):
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
        )
        study_df = study.trials_dataframe()
    else:
        # run the optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with HiddenPrints():
                study, study_df = optimization.optuna_parallel(
                    study_name=study_name,
                    study_storage=storage,
                    objective=objective,
                    n_trials=n_trials,
                    maximize_cpus=True,
                    parallel=parallel,
                )

    print(study.best_trial)

    eqs = hm.EquivalentSources(
        damping=study.best_params.get("damping"),
        depth=study.best_params.get("depth"),
        **eq_kwargs,
    ).fit(coordinates, data)

    if plot is True:
        plotting.plot_optuna_inversion_figures(
            study,
            target_names=["score"],
            # include_duration=True,
        )

    return study_df.sort_values("value", ascending=False), eqs


def normalize_xarray(da, low=0, high=1):
    # min_val = da.values.min()
    # max_val = da.values.max()

    min_val = da.quantile(0)
    max_val = da.quantile(1)

    return (high - low) * (((da - min_val) / (max_val - min_val)).clip(0, 1)) + low


def grids_to_prisms(
    surface: xr.DataArray,
    reference: Union[float, xr.DataArray],
    density: Union[float, int, xr.DataArray],
    input_coord_names=["easting", "northing"],
):
    """
    create a Harmonica layer of prisms with assigned densities.

    Parameters
    ----------
    surface : xr.DataArray
        data to use for prism surface
    reference : xr.DataArray
        data or constant to use for prism reference, if value is below surface, prism
        will be inverted
    density : Union[float, int, xr.DataArray]
        data or constant to use for prism densities.

    Returns
    -------
    hm.prism_layer
       a prisms layer with assigned densities
    """
    # if density provided as a single number, use it for all prisms
    if isinstance(density, (float, int)):
        dens = density * np.ones_like(surface)
    # if density provided as a dataarray, map each density to the correct prisms
    elif isinstance(density, xr.DataArray):
        dens = density
    else:
        raise ValueError("invalid density type, should be a number or DataArray")

    # create layer of prisms based off input dataarrays
    prisms = hm.prism_layer(
        coordinates=(
            surface[input_coord_names[0]].values,
            surface[input_coord_names[1]].values,
        ),
        surface=surface,
        reference=reference,
        properties={
            "density": dens,
            "thickness": surface - reference,
        },
    )

    return prisms


def forward_grav_of_prismlayer(
    prisms: list,
    observation_points: pd.DataFrame,
    names: list,
    remove_median=False,
    DC_shift=None,
    progressbar=False,
    plot: bool = True,
    **kwargs,
):
    df = copy.deepcopy(observation_points)

    for i, p in enumerate(prisms):
        grav = p.prism_layer.gravity(
            coordinates=(df.easting, df.northing, df.upward),
            field="g_z",
            progressbar=progressbar,
        )

        # center around 0
        if remove_median is True:
            grav -= np.median(grav)

        # add to dataframe
        df[names[i]] = grav

    # add all together
    df["forward_total"] = df[names].sum(axis=1, skipna=True)

    # apply a DC shift
    if DC_shift is not None:
        df.forward_total += DC_shift

    if remove_median is True:
        df.forward_total -= df.forward_total.median()

    # grid each column into a common dataset
    grids = df.set_index(["northing", "easting"]).to_xarray()

    if plot is True:
        for i, n in enumerate(names + ["forward_total"]):
            if i == 0:
                fig = None
                origin_shift = "initialize"
            else:
                origin_shift = "xshift"
                fig = fig

            fig = maps.plot_grd(
                grids[n],
                fig_height=8,
                cmap="vik",
                cbar_label="mGal",
                title=n.capitalize(),
                coast=False,
                hist=True,
                cbar_yoffset=3,
                fig=fig,
                origin_shift=origin_shift,
            )
            fig.text(
                position="TL",
                justify="BL",
                text=f"{string.ascii_lowercase[i]})",
                font="20p,Helvetica,black",
                offset="j0/.3",
                no_clip=True,
            )
        fig.show()

    return grids, df


def make_surface(
    spacing,
    region,
    top,
    checkerboard=True,
    amplitude=100,
    wavelength=10e3,
    plot=True,
):
    """
    Function to create either a flat or checkboard surface.

    Parameters
    ----------
    spacing : float
        grid spacing
    region : list
        region string [e,w,n,s]
    top : float
        top for flat surface, or baselevel for checkboard
    checkerboard : bool, optional
        choose whether to return a checkboard or flat surface, by default True
    amplitude : int, optional
        amplitude of checkboard, by default 100
    wavelength : float, optional
        checkboard wavelength in same units as grid, by default 10,000
    plot : bool, optional
        plot the results, by default True

    Returns
    -------
    xarray.DataArray
        the generated surface
    """
    # create a surface with repeating checkerboard pattern
    if checkerboard is True:
        synth = vd.synthetic.CheckerBoard(
            amplitude=amplitude,
            region=region,
            w_east=wavelength,
            w_north=wavelength,
        )

        surface = synth.grid(
            spacing=spacing, data_names="upward", dims=("northing", "easting")
        ).upward

        surface += top

    # create grid of coordinates
    else:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
        )

        # create xarray dataarray from coords with a constant value as defined by 'top'
        surface = vd.make_xarray_grid(
            coords,
            np.ones_like(coords[0]) * top,
            data_names="upward",
            dims=("northing", "easting"),
        ).upward

    if plot is True:
        # plot gravity and percentage contours
        fig, ax = plt.subplots()
        surface.plot(ax=ax, robust=True)
        ax.set_aspect("equal")

    return surface


def gravity_decay_buffer(
    buffer_perc,
    spacing=1e3,
    interest_region=[-5e3, 5e3, -10e3, 15e3],
    top=2e3,
    checkerboard=False,
    density_contrast=False,
    reference=-4e3,
    obs_height=1200,
    density=2300,
    plot=False,
    percentages=[0.99, 0.95, 0.90],
    **kwargs,
):
    """
    For a given buffer zone width (as percentage of x or y range) and domain parameters,
    calculate the max percent decay of the gravity anomaly within the region of
    interest.
    Decay is defined as the (max-min)/max.
    """
    # get x and y range of interest region
    x_diff = np.abs(interest_region[0] - interest_region[1])
    y_diff = np.abs(interest_region[2] - interest_region[3])

    # pick the bigger range
    max_diff = max(x_diff, y_diff)

    # calc buffer as percentage of width
    buffer_width = max_diff * (buffer_perc / 100)

    # round to nearest multiple of spacing
    def round_to_input(num, multiple):
        return round(num / multiple) * multiple

    # round buffer width to nearest spacing interval
    buffer_width = round_to_input(buffer_width, spacing)

    # define buffer region
    buffer_region = utils.alter_region(interest_region, buffer=buffer_width)[1]

    # calculate buffer width in terms of number of cells
    buffer_cells = buffer_width / spacing

    # create surface
    surface = make_surface(
        spacing=spacing,
        region=buffer_region,
        top=top,
        checkerboard=checkerboard,
        amplitude=kwargs.get("amplitude", 100),
        wavelength=kwargs.get("wavelength", 10e3),
        plot=False,
    )

    # define what the reference is
    if density_contrast is True:
        # prism reference is mean value of surface
        reference = surface.values.mean()

        # positive densities above, negative below
        dens = surface.copy()
        dens.values[:] = density
        dens = dens.where(surface >= reference, -density)
    else:
        dens = density

    # create prism layer
    flat_prisms = grids_to_prisms(
        surface,
        reference,
        density=dens,
    )

    # create set of observation points
    data = vd.grid_coordinates(
        interest_region,
        spacing=spacing,
        extra_coords=obs_height,
    )

    observation_points = pd.DataFrame(
        {
            "easting": data[0].ravel(),
            "northing": data[1].ravel(),
            "upward": data[2].ravel(),
        }
    )

    # calculate forward gravity of prism layer
    forward_grid, forward_df = forward_grav_of_prismlayer(
        [flat_prisms],
        observation_points,
        plot=False,
        names=["Flat prisms"],
    )

    grav = forward_grid["forward_total"]

    # get max decay value inside the region of interest
    if density_contrast is True:
        # max_decay = (abs(grav.values.min())-grav.values.max())/(grav.values.max()*2)
        max_decay = 10
    else:
        max_decay = (grav.values.max() - grav.values.min()) / grav.values.max()

    if plot is True:
        results = (
            f"maximum decay: {int(max_decay*100)}% \n"
            f"buffer: {buffer_perc}% / {buffer_width}m / {int(buffer_cells)} cells"
        )

        print(results)

        # plot diagonal profile
        if kwargs.get("plot_profile", False) is True:
            data_dict = profile.make_data_dict(
                ["Forward gravity"],
                [grav],
                ["black"],
            )
            # profile.plot_data(
            #     "points",
            #     start=(interest_region[0], interest_region[2]),
            #     stop=(interest_region[1], interest_region[3]),
            #     data_dict=data_dict,
            # )
            layers_dict = profile.make_data_dict(
                ["Surface"],
                [surface],
                ["black"],
            )
            profile.plot_profile(
                "points",
                start=(buffer_region[0], (buffer_region[3] - buffer_region[2]) / 2),
                stop=(buffer_region[1], (buffer_region[3] - buffer_region[2]) / 2),
                layers_dict=layers_dict,
                data_dict=data_dict,
                add_map=True,
                map_background=surface,
                inset=False,
                gridlines=False,
            )

        # plot histogram of gravity decay values
        sns.displot(forward_df.forward_total, kde=True)

        # add lines for various decay percentiles
        col = ["r", "cyan", "k"]
        for i, p in enumerate(percentages):
            plt.axvline(
                forward_df.forward_total.max() * p,
                color=col[i],
                label=f"{p*100}%",
            )
        plt.xlabel("grav")
        plt.ylabel("count")
        plt.title("Gravity decay within region of interest")
        plt.legend()

        # plot gravity and percentage contours
        fig, ax = plt.subplots()
        grav.plot(ax=ax, robust=True)
        ax.set_aspect("equal")

        grav.plot.contour(
            levels=[
                forward_df.forward_total.max() * percentages[0],
                forward_df.forward_total.max() * percentages[1],
                forward_df.forward_total.max() * percentages[2],
            ],
            colors=["k", "cyan", "r"],
        )

        ax.set_title("Forward gravity with decay contours")

    return max_decay, buffer_width, buffer_cells, grav


def regional_seperation_quality(
    comparison_method,
    input_grav,
    input_forward_column,
    input_grav_column,
    true_regional,
    grav_spacing,
    inversion_region,
    constraints,
    regional_method,
    param,
    **kwargs,
):
    """
    Evaluate the effectiveness of the regional-residual seperation of the gravity
    misfit. 2 methods of evaluations, by comparing to the know regional field, or by
    minimizing the misfit at the constraint points.

    The returned RMSE value is used in the below optimization function as the object to
    minimize.

    Returns
    -------
    rms, df_anomalies
        returns a single RMSE value, and a dataframe
    """

    df_anomalies = regional.regional_seperation(
        input_grav=input_grav,
        input_forward_column=input_forward_column,
        input_grav_column=input_grav_column,
        grav_spacing=grav_spacing,
        regional_method=regional_method,
        inversion_region=inversion_region,
        filter=f"g{param}",
        trend=param,
        constraints=constraints,
        tension_factor=param,
        eq_sources=param,
        block_size=kwargs.get("block_size", grav_spacing),
    )

    if comparison_method == "regional_comparison":
        # compare the true regional gravity with the calculated regional
        df = profile.sample_grids(
            df_anomalies,
            true_regional,
            "true_regional",
            coord_names=("easting", "northing"),
        )
        rmse = RMSE(df.true_regional - df.reg)
    elif comparison_method == "minimize_constraints":
        # grid the residuls
        residuals = pygmt.xyz2grd(
            data=df_anomalies[["easting", "northing", "res"]],
            region=inversion_region,
            spacing=grav_spacing,
            registration="g",
            verbose="q",
        )
        # sample the residuals at the constraint points
        df = profile.sample_grids(
            constraints,
            residuals,
            "residuals",
            coord_names=("easting", "northing"),
        )
        rmse = RMSE(df.residuals)
    else:
        raise ValueError(
            "comparison method must be either `regional_comparison` or"
            "`minimize_constraints`"
        )

    return rmse, df_anomalies


def sample_bounding_surfaces(
    prisms_df: pd.DataFrame,
    upper_confining_layer: xr.DataArray = None,
    lower_confining_layer: xr.DataArray = None,
):
    """
    sample upper and/or lower confining layers into prisms dataframe
    """
    df = prisms_df.copy()

    if upper_confining_layer is not None:
        df = profile.sample_grids(
            df=df,
            grid=upper_confining_layer,
            name="upper_bounds",
            coord_names=["easting", "northing"],
        )
        assert len(df.upper_bounds) != 0
    if lower_confining_layer is not None:
        df = profile.sample_grids(
            df=df,
            grid=lower_confining_layer,
            name="lower_bounds",
            coord_names=["easting", "northing"],
        )
        assert len(df.lower_bounds) != 0
    return df


def apply_max_change_per_iter(
    Surface_correction: np.array,
    max_layer_change_per_iter: float = None,
):
    """
    optionally constrain the surface correction to a max allowed change per iteration
    """
    correction = Surface_correction.copy()

    for i, j in enumerate(correction):
        if j > max_layer_change_per_iter:
            correction[i] = max_layer_change_per_iter
        elif j < -max_layer_change_per_iter:
            correction[i] = -max_layer_change_per_iter
    print(
        "Layer correction (after clipped) median:",
        f"{int(np.median(correction))}m,",
        f"RMSE:{int(RMSE(correction))} m",
    )

    return correction


def enforce_confining_surface(
    prisms_df: pd.DataFrame,
    iteration_number: int,
):
    """
    ensure the surface correction doesn't move the prisms above or below optional
    confining surfaces.
    """
    df = prisms_df.copy()

    if "upper_bounds" in df:
        # get max change in upwards direction for each prism
        df["max_change_above"] = df.upper_bounds - df.surface

        number_enforced = 0
        for i, j in enumerate(df[f"iter_{iteration_number}_correction"]):
            if j > df.max_change_above[i]:
                number_enforced += 1
                df[f"iter_{iteration_number}_correction"][i] = df.max_change_above[i]
        print(f"enforced upper confining surface at {number_enforced} prisms")

    if "lower_bounds" in df:
        # get max change in downwards direction for each prism
        df["max_change_below"] = df.surface - df.lower_bounds

        number_enforced = 0
        for i, j in enumerate(df[f"iter_{iteration_number}_correction"]):
            if j < df.max_change_below[i]:
                number_enforced += 1
                df[f"iter_{iteration_number}_correction"][i] = df.max_change_above[i]
        print(f"enforced lower confining surface at {number_enforced} prisms")

    return df


def apply_surface_correction(
    prisms_df: pd.DataFrame,
    iteration_number: int,
):
    """
    update the prisms dataframe and dataset with the surface correction. Ensure that the
    updated surface doesn't intersect the optional confining surfaces.
    """
    df = prisms_df.copy()

    # for negative densities, negate the correction
    df.loc[df.density < 0, f"iter_{iteration_number}_correction"] *= -1

    # grid the corrections
    # correction_grid_before = (
    #     df.rename(columns={f"iter_{iteration_number}_correction": "z"})
    #     .set_index(["northing", "easting"])
    #     .to_xarray()
    #     .z
    # )

    # optionally constrain the surface correction with bounding surfaces
    df = enforce_confining_surface(df, iteration_number)

    # grid the corrections
    correction_grid = (
        df.rename(columns={f"iter_{iteration_number}_correction": "z"})
        .set_index(["northing", "easting"])
        .to_xarray()
        .z
    )

    # (correction_grid-correction_grid_before).plot(robust=True)

    return df, correction_grid


def update_prisms_ds(
    prisms_ds: xr.Dataset,
    correction_grid: xr.DataArray,
    zref: float,
):
    """
    apply the corrections grid and update the prism tops, bottoms, surface, and
    densities.
    """
    ds = prisms_ds.copy()

    density_contrast = ds.density.values.max()

    # create surface from top and bottom
    surface_grid = xr.where(ds.density > 0, ds.top, ds.bottom)

    # apply correction to surface
    surface_grid += correction_grid

    # update the prism layer
    ds.prism_layer.update_top_bottom(surface=surface_grid, reference=zref)

    # update the density
    ds["density"] = xr.where(ds.top > zref, density_contrast, -density_contrast)

    # update the surface
    ds["surface"] = surface_grid

    return ds


def add_updated_prism_properties(
    prisms_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    iteration_number: int,
):
    """
    update the prisms dataframe the the new prism tops, bottoms, surface, and densities
    """
    df = prisms_df.copy()
    ds = prisms_ds.copy()

    # turn back into dataframe
    prisms_iter = ds.to_dataframe().reset_index().dropna().astype(float)

    # add new cols to dict
    dict_of_cols = {
        f"iter_{iteration_number}_top": prisms_iter.top,
        f"iter_{iteration_number}_bottom": prisms_iter.bottom,
        f"iter_{iteration_number}_density": prisms_iter.density,
        f"iter_{iteration_number}_layer": prisms_iter.surface,
    }

    df = pd.concat([df, pd.DataFrame(dict_of_cols)], axis=1)

    return df


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)

    return wrapper
