import pathlib
import pickle

import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import xarray as xr
import zarr
from antarctic_plots import maps, profile, utils
from scipy.special import erf
from scipy.stats import pearsonr
from tqdm.autonotebook import tqdm
from UQpy.distributions import Normal, Uniform
from UQpy.sampling import LatinHypercubeSampling as LHS
from UQpy.sampling.stratified_sampling.latin_hypercube_criteria import Centered

from RIS_gravity_inversion import inversion as inv
from RIS_gravity_inversion import regional
from RIS_gravity_inversion import utils as inv_utils


def monte_carlo_full_workflow_uncertainty_loop(
    fname,
    runs,
    grav,
    constraints,
    sample_grav=False,
    sample_constraints=False,
    sample_starting_bed_tension=False,
    sample_starting_bed_damping=False,
    sample_regional_tension_factor=False,
    sample_regional_damping=False,
    sample_ice_density=False,
    sample_water_density=False,
    sample_sediment_density=False,
    run_damping_CV=True,
    damping_CV_fname=None,
    inversion_args=None,
    starting_args=None,
    sampling="LHC",
    **kwargs,
):
    """
    Run a series of Monte Carlo simulations (N=runs), and save results of
    each inversion to pickle files starting with `fname`. If files already
    exist, just return the loaded results instead of re-running the inversion.
    Choose which variables to include in the sampling and whether or not to
    run a damping value cross-validation for each inversion.

    Feed returned values into function `merged_stats` to compute
    cell-wise stats on the resulting ensemble of starting bed models,
    inverted bed models, and gravity anomalies.
    """
    # if file exists, start and next run, else start at 0
    try:
        # load pickle files
        params = []
        with open(f"{fname}_params.pickle", "rb") as file:
            while 1:
                try:
                    params.append(pickle.load(file))
                except EOFError:
                    break
        starting_run = len(params)
    except FileNotFoundError:
        print(f"No pickle files starting with '{fname}' found, creating new files\n")
        # create / overwrite pickle files
        with open(f"{fname}_params.pickle", "wb") as _:
            pass
        with open(f"{fname}_sampled_values.pickle", "wb") as _:
            pass
        with open(f"{fname}_grav_dfs.pickle", "wb") as _:
            pass
        with open(f"{fname}_prism_dfs.pickle", "wb") as _:
            pass
        starting_run = 0
    if starting_run == runs:
        print(f"all {runs} runs already complete, loading results from files.")
    for i in tqdm(range(starting_run, runs)):
        if i == starting_run:
            print(
                f"starting Monte Carlo uncertainty analysis at run {starting_run} of "
                f"{runs}\nsaving results to pickle files with prefix: '{fname}'\n"
            )

        recalculate_starting_bed = False
        recalculate_ice_gravity = False
        recalculate_water_gravity = False
        recalculate_bed_gravity = False
        recalculate_regional = False

        if sampling == "LHC":
            if i == 0:
                # gather distributions (excluding grav and constraints)
                dists = {}
                if sample_starting_bed_tension is True:
                    dists["starting_bed_tension"] = Uniform(loc=0, scale=1)
                if sample_starting_bed_damping is True:
                    dists["starting_bed_damping"] = Uniform(
                        loc=10**-40, scale=10**-10
                    )
                if sample_regional_tension_factor is True:
                    dists["regional_tension_factor"] = Uniform(loc=0, scale=1)
                if sample_regional_damping is True:
                    dists["regional_damping"] = Uniform(loc=10**-45, scale=10**-5)
                if sample_ice_density is True:
                    dists["ice_density"] = Normal(
                        loc=starting_args.get("ice_density"), scale=5
                    )
                if sample_water_density is True:
                    dists["water_density"] = Normal(
                        loc=starting_args.get("water_density"), scale=5
                    )
                if sample_sediment_density is True:
                    dists["sediment_density"] = Normal(
                        loc=starting_args.get("sediment_density"), scale=400
                    )

                # make latin hyper cube
                lhs = LHS(
                    distributions=[v for k, v in dists.items()],
                    criterion=Centered(),
                    # criterion=Random(),
                    random_state=np.random.RandomState(1),
                    nsamples=runs,
                )
                # make dict of sampled values
                samples = {}
                for j, (k, v) in enumerate(dists.items()):
                    samples[k] = lhs._samples[:, j]

            # get ith element of each sampled list
            sampled_args = {}
            for j, (k, v) in enumerate(samples.items()):
                sampled_args[k] = v[i]

            # sample grav and constraints with random sampling
            # create random generator
            rand = np.random.default_rng(seed=i)

            # if gravity resampled, must recalculate misfit and regional
            if sample_grav is True:
                sampled_grav = grav.copy()
                Gobs_sampled = rand.normal(sampled_grav.Gobs, sampled_grav.uncert)
                sampled_grav["Gobs"] = Gobs_sampled
                gravity_df = sampled_grav.copy()
            elif sample_grav is False:
                gravity_df = grav.copy()
            # if constraints resampled, must recalculate starting bed,
            # recalculate bed gravity, and recalculate regional
            if sample_constraints is True:
                sampled_constraints = constraints.copy()
                sampled_constraints["upward"] = rand.normal(
                    sampled_constraints.upward, sampled_constraints.z_error
                )
                constraint_points = sampled_constraints.copy()
            elif sample_constraints is False:
                constraint_points = constraints.copy()

        elif sampling == "random":
            # create random generator
            rand = np.random.default_rng(seed=i)

            # if gravity resampled, must recalculate misfit and regional
            if sample_grav is True:
                sampled_grav = grav.copy()
                Gobs_sampled = rand.normal(sampled_grav.Gobs, sampled_grav.uncert)
                sampled_grav["Gobs"] = Gobs_sampled
                gravity_df = sampled_grav.copy()
            elif sample_grav is False:
                gravity_df = grav.copy()
            # if constraints resampled, must recalculate starting bed,
            # recalculate bed gravity, and recalculate regional
            if sample_constraints is True:
                sampled_constraints = constraints.copy()
                sampled_constraints["upward"] = rand.normal(
                    sampled_constraints.upward, sampled_constraints.z_error
                )
                constraint_points = sampled_constraints.copy()
            elif sample_constraints is False:
                constraint_points = constraints.copy()

            # make dict of sampled values
            sampled_args = {}
            if sample_starting_bed_tension is True:
                sampled_args["starting_bed_tension"] = rand.uniform(0, 1)
            if sample_starting_bed_damping is True:
                deg = rand.uniform(-40, -10)
                sampled_args["starting_bed_damping"] = 10**deg
            if sample_regional_tension_factor is True:
                sampled_args["regional_tension_factor"] = rand.uniform(0, 1)
            if sample_regional_damping is True:
                deg = rand.uniform(-45, -5)
                sampled_args["regional_damping"] = 10**deg
            if sample_ice_density is True:
                sampled_args["ice_density"] = rand.normal(
                    starting_args.get("ice_density"), 5
                )  # 5 from Hawley et al. 2004
            if sample_water_density is True:
                sampled_args["water_density"] = rand.normal(
                    starting_args.get("water_density"), 5
                )
            if sample_sediment_density is True:
                sampled_args["sediment_density"] = rand.normal(
                    starting_args.get("sediment_density"), 400
                )

        # if values not sampled, add default values from starting args
        if sample_starting_bed_tension is False:
            sampled_args["starting_bed_tension"] = starting_args.get(
                "starting_bed_tension"
            )
        if sample_starting_bed_damping is False:
            sampled_args["starting_bed_damping"] = starting_args.get(
                "starting_bed_damping"
            )
        if sample_regional_tension_factor is False:
            sampled_args["regional_tension_factor"] = starting_args.get(
                "regional_tension_factor"
            )
        if sample_regional_damping is False:
            sampled_args["regional_damping"] = starting_args.get("regional_damping")
        if sample_ice_density is False:
            sampled_args["ice_density"] = starting_args.get("ice_density")
        if sample_water_density is False:
            sampled_args["water_density"] = starting_args.get("water_density")
        if sample_sediment_density is False:
            sampled_args["sediment_density"] = starting_args.get("sediment_density")

        if sample_starting_bed_tension & sample_starting_bed_damping:
            raise ValueError(
                "cant resample both starting bed damping and tension factor"
            )
        elif sample_starting_bed_tension:
            starting_bed_method = "surface"
        elif sample_starting_bed_damping:
            starting_bed_method = "spline"
        else:
            starting_bed_method = starting_args.get("starting_bed_method")

        if sample_regional_damping & sample_regional_tension_factor:
            raise ValueError(
                "cant resample both regional damping and regional tension factor"
            )
        elif sample_regional_damping:
            regional_grid_method = "verde"
        elif sample_regional_tension_factor:
            regional_grid_method = "pygmt"
        else:
            regional_grid_method = starting_args.get("regional_grid_method")

        # define what needs to be done depending on what parameters are sampled
        if sample_grav is True:
            recalculate_regional = True
        if sample_constraints is True:
            recalculate_starting_bed = True
            recalculate_bed_gravity = True
            recalculate_regional = True
        if sample_starting_bed_tension is True:
            recalculate_starting_bed = True
        if sample_starting_bed_damping is True:
            recalculate_starting_bed = True
        if sample_regional_tension_factor is True:
            recalculate_regional = True
        if sample_regional_damping is True:
            recalculate_regional = True
        if sample_ice_density is True:
            recalculate_ice_gravity = True
            recalculate_water_gravity = True
        if sample_water_density is True:
            recalculate_water_gravity = True
            recalculate_bed_gravity = True
        if sample_sediment_density is True:
            recalculate_bed_gravity = True

        # if certain things are recalculated, other must be as well
        if recalculate_starting_bed is True:
            recalculate_bed_gravity = True
        if recalculate_ice_gravity is True:
            recalculate_regional = True
        if recalculate_water_gravity is True:
            recalculate_regional = True
        if recalculate_bed_gravity is True:
            recalculate_regional = True

        if i == starting_run:
            verbose = None
        else:
            verbose = "q"

        # run inversion
        inv_results = monte_carlo_full_workflow(
            constraints=constraint_points,
            gravity_df=gravity_df,
            recalculate_starting_bed=recalculate_starting_bed,
            recalculate_ice_gravity=recalculate_ice_gravity,
            recalculate_water_gravity=recalculate_water_gravity,
            recalculate_bed_gravity=recalculate_bed_gravity,
            recalculate_regional=recalculate_regional,
            run_damping_CV=run_damping_CV,
            damping_CV_fname=f"{damping_CV_fname}_run_{i}",
            inversion_args=inversion_args,
            sampled_args=sampled_args,
            verbose=verbose,
            regional_grid_method=regional_grid_method,
            starting_bed_method=starting_bed_method,
            **kwargs,
        )

        # get prism layer data
        prism_df = inv_results[0]

        # get gravity data
        grav_df = inv_results[1]

        # get parameters
        params = inv_results[2]

        # add run number to the parameter values
        params["run_num"] = i

        # save results
        with open(f"{fname}_params.pickle", "ab") as file:
            pickle.dump(params, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{fname}_sampled_values.pickle", "ab") as file:
            pickle.dump(sampled_args, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{fname}_grav_dfs.pickle", "ab") as file:
            pickle.dump(grav_df, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f"{fname}_prism_dfs.pickle", "ab") as file:
            pickle.dump(prism_df, file, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Finished inversion {i+1} of {runs}")

    # load pickle files
    params = []
    with open(f"{fname}_params.pickle", "rb") as file:
        while 1:
            try:
                params.append(pickle.load(file))
            except EOFError:
                break
    values = []
    with open(f"{fname}_sampled_values.pickle", "rb") as file:
        while 1:
            try:
                values.append(pickle.load(file))
            except EOFError:
                break
    grav_dfs = []
    with open(f"{fname}_grav_dfs.pickle", "rb") as file:
        while 1:
            try:
                grav_dfs.append(pickle.load(file))
            except EOFError:
                break
    prism_dfs = []
    with open(f"{fname}_prism_dfs.pickle", "rb") as file:
        while 1:
            try:
                prism_dfs.append(pickle.load(file))
            except EOFError:
                break

    return (params, values, grav_dfs, prism_dfs)


def merge_simulation_results(
    params,
    results_dfs,
    col_name,
):
    """
    supply list of dataframe from Monte Carlo simulation, specify a column with either
    a string or the index position of the column, turn into a dataarray, rename with the
     run number of the simulation (from params), and merge into a dataset.
    prism_dfs col names; -1 for inverted bathymetry, "starting_bed" for starting
    bathymetry
    grav_dfs col names; -1 for final residual, "Gobs" for disturbance,
    "starting_bed_grav", "ice_surface_grav", "water_surface_grav",
    "partial_topo_corrected_disturbance", "misfit", "reg", "res",
    """
    # get list of run numbers
    runs = [p["run_num"] for p in params]

    # get specified grids, rename with run numbers and merge into dataset
    # if given column is int, get that index of the column names
    if isinstance(col_name, int):
        grids = [
            df.set_index(["northing", "easting"]).to_xarray()[df.columns[col_name]]
            for df in results_dfs
        ]
    # else use col_name as string
    else:
        grids = [
            df.set_index(["northing", "easting"]).to_xarray()[col_name]
            for df in results_dfs
        ]

    renamed_grids = []
    for i, j in enumerate(grids):
        da = j.rename(f"run_{runs[i]}")
        renamed_grids.append(da)
    merged = xr.merge(renamed_grids)

    return merged


def merged_stats(
    results,
    grav_or_topo,
    col_name,
    plot=True,
    mask=None,
    title="Monte Carlo simulation",
    constraints=None,
    weight_by="constraints",
):
    # unpack results
    params, values, grav_dfs, prism_dfs = results

    # pick either gravity dataframe or topography dataframe
    if grav_or_topo == "grav":
        results_dfs = grav_dfs
    elif grav_or_topo == "topo":
        results_dfs = prism_dfs

    # get merged dataset
    merged = merge_simulation_results(
        params,
        results_dfs,
        col_name,
    )

    # get final gravity residual RMS of each model
    if weight_by == "residual":
        weight_vals = [utils.RMSE(df[list(df.columns)[-1]]) for df in grav_dfs]
    # get constraint point RMSE of each model
    elif weight_by == "constraints":
        weight_vals = []
        for df in prism_dfs:
            ds = df.set_index(["northing", "easting"]).to_xarray()
            bed = ds[df.columns[-1]]
            points = profile.sample_grids(
                constraints[constraints.inside],
                bed,
                name="sampled_bed",
                coord_names=["easting", "northing"],
            )
            points["dif"] = points.upward - points.sampled_bed
            weight_vals.append(utils.RMSE(points.dif))

    # convert residuals into weights
    weights = [1 / (x**2) for x in weight_vals]

    # get stats and weighted stats on the merged dataset
    stats_ds = model_ensemble_stats(merged, weights=weights)

    if plot is True:
        if grav_or_topo == "grav":
            cmap = "viridis"
            unit = "mGal"
            if col_name == -1:
                col_name = "final residual gravity"
        elif grav_or_topo == "topo":
            cmap = "batlowW"
            unit = "m"
            if col_name == -1:
                col_name = "inverted bed"

        if isinstance(col_name, int):
            col_name = results_dfs[0][results_dfs[0].columns[col_name]].name

        if mask is not None:
            weighted_stdev = utils.mask_from_shp(
                shapefile=mask,
                xr_grid=stats_ds.weighted_stdev,
                masked=True,
                invert=False,
            )
        else:
            weighted_stdev = stats_ds.weighted_stdev

        fig = maps.plot_grd(
            weighted_stdev,
            cmap="inferno",
            coast=True,
            coast_version="measures-v2",
            reverse_cpt=True,
            robust=True,
            hist=True,
            cbar_label=f"{col_name}: weighted standard deviation, {unit}",
            title=title,
        )
        if constraints is not None:
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
                pen=".4p,black",
                style="c.08c",
                label="Bed constraints",
            )
        fig.legend()

        if mask is not None:
            weighted_mean = utils.mask_from_shp(
                shapefile=mask,
                xr_grid=stats_ds.weighted_mean,
                masked=True,
                invert=False,
            )
        else:
            weighted_mean = stats_ds.weighted_mean

        fig = maps.plot_grd(
            weighted_mean,
            cmap=cmap,
            coast=True,
            coast_version="measures-v2",
            robust=True,
            hist=True,
            cbar_label=f"{col_name}: weighted mean, {unit}",
            title=title,
            fig=fig,
            origin_shift="xshift",
        )
        if constraints is not None:
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
                pen=".4p,black",
                style="c.08c",
                label="Bed constraints",
            )
        fig.legend()

        fig.show()

    return stats_ds


def monte_carlo_inversion_uncertainty_loop(
    fname,
    runs,
    grav,
    constraints,
    sample_grav=False,
    sample_constraints=False,
    sample_water_density=False,
    sample_sediment_density=False,
    inversion_args=None,
    sampled_args=None,
    starting_args=None,
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
        print(f"all {runs} runs already complete, loading results from file.")
    for i in tqdm(range(starting_run, runs)):
        if i == starting_run:
            print(
                f"starting Monte Carlo uncertainty analysis at run {starting_run} of "
                f"{runs}\nsaving results to {fname}\n"
            )

        # create random generator
        rand = np.random.default_rng(seed=i)

        recalculate_starting_bed = False
        recalculate_bed_gravity = False
        recalculate_regional = False

        sampled_args = {}

        # if gravity resampled, must recalculate misfit and regional
        if sample_grav is True:
            sampled_grav = grav.copy()
            Gobs_sampled = rand.normal(sampled_grav.Gobs, sampled_grav.uncert)
            sampled_grav["Gobs"] = Gobs_sampled
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

        # if water density resampled, must recalculate bed gravity and regional
        if sample_water_density is True:
            sampled_args["water_density"] = rand.normal(
                starting_args.get("water_density"), 5
            )
            recalculate_bed_gravity = True
            recalculate_regional = True
        elif sample_water_density is False:
            sampled_args["water_density"] = starting_args.get("water_density")

        # if sediment density resampled, must recalculate bed gravity and regional
        if sample_sediment_density is True:
            sampled_args["sediment_density"] = rand.normal(
                starting_args.get("sediment_density"), 400
            )
            recalculate_bed_gravity = True
            recalculate_regional = True
        elif sample_sediment_density is False:
            sampled_args["sediment_density"] = starting_args.get("sediment_density")

        if recalculate_starting_bed is True:
            recalculate_bed_gravity = True
            recalculate_regional = True
        if recalculate_bed_gravity is True:
            recalculate_regional = True

        if i == starting_run:
            verbose = None
        else:
            verbose = "q"

        # run inversion
        topo, results = monte_carlo_inversion(
            constraints=constraint_points,
            gravity_df=gravity_df,
            recalculate_starting_bed=recalculate_starting_bed,
            recalculate_bed_gravity=recalculate_bed_gravity,
            recalculate_regional=recalculate_regional,
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

    # load pickle files
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

    # convert residuals into weights
    weights = [1 / (x**2) for x in residuals]

    # load dataset with all the topo models
    topos = xr.open_zarr(fname)

    # get stats on the ensemble
    stats = model_ensemble_stats(topos, weights=weights)

    if plot is True:
        grd = stats.weighted_stdev
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


def monte_carlo_full_workflow(
    constraints,
    gravity_df,
    recalculate_starting_bed=False,
    recalculate_ice_gravity=False,
    recalculate_water_gravity=False,
    recalculate_bed_gravity=False,
    recalculate_regional=False,
    starting_bed_method=None,
    regional_grid_method=None,
    run_damping_CV=True,
    damping_CV_fname=None,
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

    # re-create starting bed
    if recalculate_starting_bed is False:
        starting_bed = kwargs.get("starting_bed", None)
        starting_bed_prisms = kwargs.get("starting_bed_prisms", None)
    else:
        # recalculate the starting bed with the provided constraints
        if verbose != "q":
            print("Recalculating starting bed")
        with inv_utils.HiddenPrints():
            starting_bed = inv_utils.recreate_bed(
                inside_points=constraints[constraints.inside],
                buffer_points=kwargs.get("buffer_points"),
                outside_grid=kwargs.get("bed_outside"),
                region=kwargs.get("buffer_region"),
                fullres_spacing=kwargs.get("fullres_spacing"),
                layer_spacing=kwargs.get("layer_spacing"),
                method=starting_bed_method,
                damping=sampled_args.get("starting_bed_damping"),
                tension=sampled_args.get("starting_bed_tension"),
                icebase=kwargs.get("icebase_layer_spacing"),
                surface=kwargs.get("surface_layer_spacing"),
                icebase_fullres=kwargs.get("icebase_fullres"),
                surface_fullres=kwargs.get("surface_fullres"),
            )
        recalculate_bed_gravity = True

    # re-calculate ice mass effect
    if recalculate_ice_gravity is False:
        assert all(item in grav.columns for item in ["ice_surface_grav"]), (
            "if recalculate_ice_gravity is False,"
            " 'ice_surface_grav' must be in gravity dataframe"
        )
    else:
        if verbose != "q":
            print("Recalculating ice gravity")
        grav.drop(
            columns=[
                "ice_surface_grav",
                "partial_topo_corrected_disturbance",
                "misfit",
                "reg",
                "res",
            ],
            inplace=True,
            errors="ignore",
        )
        with inv_utils.HiddenPrints():
            grav = inv_utils.recalculate_ice_gravity(
                gravity=grav,
                air_density=1,
                ice_density=sampled_args.get("ice_density"),
                grid=kwargs.get("surface_layer_spacing"),
            )

    # re-calculate water mass effect
    if recalculate_water_gravity is False:
        assert all(item in grav.columns for item in ["water_surface_grav"]), (
            "if recalculate_water_gravity is False,"
            " 'water_surface_grav' must be in gravity dataframe"
        )
    else:
        if verbose != "q":
            print("Recalculating water gravity")
        grav.drop(
            columns=[
                "water_surface_grav",
                "partial_topo_corrected_disturbance",
                "misfit",
                "reg",
                "res",
            ],
            inplace=True,
            errors="ignore",
        )
        with inv_utils.HiddenPrints():
            grav = inv_utils.recalculate_water_gravity(
                gravity=grav,
                ice_density=sampled_args.get("ice_density"),
                water_density=sampled_args.get("water_density"),
                grid=kwargs.get("icebase_layer_spacing"),
            )

    # re-calculate starting bed gravity
    if recalculate_bed_gravity is False:
        assert all(item in grav.columns for item in ["starting_bed_grav"]), (
            "if recalculate_bed_gravity is False,"
            " 'starting_bed_grav' must be in gravity dataframe"
        )
    else:
        if verbose != "q":
            print("Recalculating starting bed gravity")
        grav.drop(
            columns=[
                "starting_bed_grav",
                "partial_topo_corrected_disturbance",
                "misfit",
                "reg",
                "res",
            ],
            inplace=True,
            errors="ignore",
        )
        with inv_utils.HiddenPrints():
            grav, starting_bed_prisms = inv_utils.recalculate_bed_gravity(
                gravity=grav,
                water_density=sampled_args.get("water_density"),
                bed_density=sampled_args.get("sediment_density"),
                grid=starting_bed,
            )

    # recalculate partial topo correction
    grav["partial_topo_corrected_disturbance"] = (
        grav.Gobs - grav.ice_surface_grav - grav.water_surface_grav
    )

    # recalculate misfit
    grav["misfit"] = grav.partial_topo_corrected_disturbance - grav.starting_bed_grav

    # recalculate regional seperation
    if recalculate_regional is False:
        assert all(item in grav.columns for item in ["res", "reg"]), (
            "if recalculate_regional is False, 'reg'"
            " and 'res' must be in gravity dataframe"
        )
        anomalies = grav
    else:
        if verbose != "q":
            print("Recalculating regional field")
        grav.drop(columns=["res", "reg"], inplace=True, errors="ignore")
        with inv_utils.HiddenPrints():
            anomalies = regional.regional_seperation(
                input_grav=grav,
                constraints=constraints,
                grav_spacing=kwargs.get("grav_spacing"),
                regional_method="constraints",
                inversion_region=kwargs.get("inversion_region"),
                grid_method=regional_grid_method,
                tension_factor=sampled_args.get("regional_tension_factor"),
                dampings=sampled_args.get("regional_damping"),
                constraint_weights_col="weights",
            )

    # add weights grid to starting prisms
    if inversion_args.get("weights_after_solving") is True:
        starting_bed_prisms["weights"] = kwargs.get("weights_grid")

    # run inversion with damping parameter cross validation
    if run_damping_CV is True:
        dampings = kwargs.get("CV_damping_values")
        scores, rmses = inv.inversion_optimal_parameters(
            training_data=anomalies[~anomalies.test],
            testing_data=anomalies[anomalies.test],
            parameter_values=dampings,
            function=inv.inversion_damping_MSE,
            results_fname=damping_CV_fname,
            progressbar=True,
            inversion_region=kwargs.get("inversion_region"),
            prism_layer=starting_bed_prisms,
            plot=False,
            **inversion_args,
        )

        # put scores and damping values into dict
        CV_results = dict(scores=scores, dampings=dampings, rmses=rmses)

        # remove if exists
        pathlib.Path(damping_CV_fname).unlink(missing_ok=True)

        # save scores and dampings to pickle
        with open(f"{damping_CV_fname}.pickle", "wb") as f:
            pickle.dump(CV_results, f)

        # load scores and dampings from pickle
        with open(f"{damping_CV_fname}.pickle", "rb") as f:
            CV_results = pickle.load(f)

        best = np.argmin(CV_results["scores"])
        print("Best score:", CV_results["scores"][best])
        print("Best damping:", CV_results["dampings"][best])

        # get best inversion result of each set
        with open(f"{damping_CV_fname}_trial_{best}.pickle", "rb") as f:
            inv_result = pickle.load(f)

        # delete other results to save space
        fnames = []
        for i in range(len(dampings)):
            if i == best:
                pass
            else:
                fnames.append(f"{damping_CV_fname}_trial_{i}.pickle")
        for f in fnames:
            pathlib.Path(f).unlink(missing_ok=True)

    # run inversion with supplied damping parameter
    elif run_damping_CV is False:
        with inv_utils.HiddenPrints():
            inv_result = inv.geo_inversion(
                input_grav=anomalies[~anomalies.test],
                prism_layer=starting_bed_prisms,
                solver_damping=kwargs.get("solver_damping"),
                **inversion_args,
            )

    return inv_result


def monte_carlo_inversion(
    constraints,
    gravity_df,
    recalculate_starting_bed=False,
    recalculate_bed_gravity=False,
    recalculate_regional=False,
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

    # re-create starting bed
    if recalculate_starting_bed is False:
        starting_bed = kwargs.get("starting_bed", None)
    else:
        # recalculate the starting bed with the provided constraints
        if verbose != "q":
            print("Recalculating starting bed")
        with inv_utils.HiddenPrints():
            starting_bed = inv_utils.recreate_bed(
                constraints=constraints,
                bed_buffer_points=kwargs.get("bed_buffer_points", None),
                region=kwargs.get("buffer_region"),
                spacing=kwargs.get("layer_spacing"),
                icebase=kwargs.get("icebase_lowres", None),
                weights=kwargs.get("constraint_weights_col", None),
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
        with inv_utils.HiddenPrints():
            grav, starting_bed_prisms = inv_utils.recalc_bed_grav(
                gravity=grav,
                starting_bed=starting_bed,
                water_density=sampled_args.get("water_density"),
                sediment_density=sampled_args.get("sediment_density"),
            )
    # recalculate regional seperation
    if recalculate_regional is False:
        assert all(item in grav.columns for item in ["Gobs_shift", "res"]), (
            "if recalculate_regional is False, 'Gobs_shift'"
            " and 'res' must be in gravity dataframe"
        )
        anomalies = grav
    else:
        if verbose != "q":
            print("Recalculating regional field")
        grav.drop(columns=["res", "reg", "misfit"], inplace=True, errors="ignore")
        with inv_utils.HiddenPrints():
            anomalies = regional.regional_seperation(
                input_grav=grav,
                grav_spacing=kwargs.get("grav_spacing"),
                regional_method="constraints",
                constraints=constraints,
                input_forward_column="bed_forward",
                input_grav_column="Gobs",
                inversion_region=kwargs.get("inversion_region"),
                grid_method="verde",
                dampings=kwargs.get("dampings"),
                constraint_weights_col=kwargs.get("constraints_weights_col"),
            )
    # add weights grid to starting prisms
    starting_bed_prisms["weights"] = kwargs.get("weights_grid")

    # run inversion
    with inv_utils.HiddenPrints():
        results = inv.geo_inversion(
            input_grav=anomalies,
            prism_layer=starting_bed_prisms,
            **inversion_args,
        )

    # get final topography
    final_topo = (
        results[0]
        .set_index(["northing", "easting"])
        .to_xarray()[results[0].columns[-1]]
    )

    return final_topo, results


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
    z_min = merged["run_num"].min("runs").rename("z_min")
    z_max = merged["run_num"].max("runs").rename("z_max")
    z_stdev = merged["run_num"].std("runs").rename("z_std")
    # z_var = merged["run_num"].var("runs").rename("z_var")

    if weights is not None:
        assert len(da_list) == len(weights)

        weighted_mean = sum(g * w for g, w in zip(da_list, weights)) / sum(weights)
        weighted_mean = weighted_mean.rename("weighted_mean")

        # from https://stackoverflow.com/questions/30383270/how-do-i-calculate-the-standard-deviation-between-weighted-measurements # noqa
        weighted_var = (
            sum(w * (g - weighted_mean) ** 2 for g, w in zip(da_list, weights))
        ) / sum(weights)
        weighted_stdev = np.sqrt(weighted_var)
        weighted_stdev = weighted_stdev.rename("weighted_stdev")

    else:
        weighted_mean = None
        weighted_stdev = None
        # weighted_var = None

    grids = [
        merged,
        z_mean,
        z_stdev,
        weighted_mean,
        weighted_stdev,
        z_min,
        z_max,
        # z_var, weighted_var,
    ]
    stats = []
    for g in grids:
        if g is not None:
            stats.append(g)

    stats = xr.merge(stats)

    return stats


def plot_stats(
    title,
    grav_dfs,
    topos,
    mask=None,
    constraints=None,
    region=None,
):
    residuals = [utils.RMSE(df[list(df.columns)[-1]]) for df in grav_dfs]

    # convert residuals into weights
    weights = [1 / (x**2) for x in residuals]

    stats = model_ensemble_stats(topos, weights=weights)

    grids = [stats[g] for g in list(stats)]

    grids = [
        utils.mask_from_shp(shapefile=mask, xr_grid=i, masked=True, invert=False)
        for i in grids
    ]

    subplot_titles = [
        "z_mean",
        "z_stdev",
        "z_min",
        "weighted_mean",
        "weighted_stdev",
        "z_max",
    ]

    fig = maps.subplots(
        grids=grids,
        region=region,
        dims=(2, 3),
        fig_title=title,
        subplot_titles=subplot_titles,
        cbar_labels=["m" for x in grids],
        autolabel="a)+JTL",
        fig_height=10,
        margins="1c",
        # grd2cpt=True,
        cmaps=[
            "rain",
            "inferno",
            "rain",
            "rain",
            "inferno",
            "rain",
        ],
        reverse_cpt=True,
        points=constraints[constraints.inside].rename(
            columns={"easting": "x", "northing": "y"}
        ),
        points_style="x.2c",
        points_pen="1.2p",
        # hist=True,
        # cbar_yoffset=1.5,
    )
    fig.show()


# elif sampling == "random":
# # create random generator
# rand = np.random.default_rng(seed=i)

# sampled_args = {}

# # if gravity resampled, must recalculate misfit and regional
# if sample_grav is True:
#     sampled_grav = grav.copy()
#     Gobs_sampled = rand.normal(sampled_grav.Gobs, sampled_grav.uncert)
#     sampled_grav["Gobs"] = Gobs_sampled
#     gravity_df = sampled_grav.copy()
# elif sample_grav is False:
#     gravity_df = grav.copy()

# # if constraints resampled, must recalculate starting bed,
# # recalculate bed gravity, and recalculate regional
# if sample_constraints is True:
#     # sampled_constraints = constraints.drop(columns="upward").copy()
#     sampled_constraints = constraints.copy()
#     # sampled_constraints["upward"] = rand.uniform(
#     #     sampled_constraints.low_lim, sampled_constraints.high_lim
#     # )
#     sampled_constraints["upward"] = rand.normal(
#       sampled_constraints.upward, sampled_constraints.z_error)
#     constraint_points = sampled_constraints.copy()
# elif sample_constraints is False:
#     constraint_points = constraints.copy()

# # if starting bed tension resampled, must recalculate starting bed, bed gravity,
# and regional
# if sample_starting_bed_tension is True:
#     sampled_args["starting_bed_tension"] = rand.uniform(0, 1)
# elif sample_starting_bed_tension is False:
#     sampled_args["starting_bed_tension"] = starting_args.get("starting_bed_tension")

# # if starting bed damping resampled, must recalculate starting bed, bed gravity,
# and regional
# if sample_starting_bed_damping is True:
#     deg = rand.uniform(-40, -10)
#     sampled_args["starting_bed_damping"] = 10**deg
# elif sample_starting_bed_damping is False:
#     sampled_args["starting_bed_damping"] = starting_args.get("starting_bed_damping")

# if sample_starting_bed_tension & sample_starting_bed_damping:
#     raise ValueError("cant resample both starting bed damping and tension factor")
# elif sample_starting_bed_tension:
#     starting_bed_method = "surface"
# elif sample_starting_bed_damping:
#     starting_bed_method = "spline"
# else:
#     starting_bed_method = starting_args.get("starting_bed_method")

# # if regional tension factor resampled, must recalculate regional
# if sample_regional_tension_factor is True:
#     sampled_args["regional_tension_factor"] = rand.uniform(0, 1)
# elif sample_regional_tension_factor is False:
#     sampled_args["regional_tension_factor"] = starting_args.get(
#           "regional_tension_factor")

# # if regional damping resampled, must recalculate regional
# if sample_regional_damping is True:
#     deg = rand.uniform(-45, -5)
#     sampled_args["regional_damping"] = 10**deg
# elif sample_regional_damping is False:
#     sampled_args["regional_damping"] = starting_args.get("regional_damping")

# if sample_regional_damping & sample_regional_tension_factor:
#     raise ValueError(
#           "cant resample both regional damping and regional tension factor")
# elif sample_regional_damping:
#     regional_grid_method = "verde"
# elif sample_regional_tension_factor:
#     regional_grid_method = "pygmt"
# else:
#     regional_grid_method = starting_args.get("regional_grid_method")

# # if ice density resampled, must recalculate ice and water effect, and regional
# if sample_ice_density is True:
#     sampled_args["ice_density"] = rand.normal(
#         starting_args.get("ice_density"), 5
#     )  # 5 from Hawley et al. 2004
# elif sample_ice_density is False:
#     sampled_args["ice_density"] = starting_args.get("ice_density")

# # if water density resampled, must recalculate water and bed effect,
# # bed gravity and regional
# if sample_water_density is True:
#     sampled_args["water_density"] = rand.normal(
#         starting_args.get("water_density"), 5)
# elif sample_water_density is False:
#     sampled_args["water_density"] = starting_args.get("water_density")

# # if sediment density resampled, must recalculate bed gravity and regional
# if sample_sediment_density is True:
#     sampled_args["sediment_density"] = rand.normal(
#         starting_args.get("sediment_density"), 400)
# elif sample_sediment_density is False:
#     sampled_args["sediment_density"] = starting_args.get("sediment_density")


"""

The below functions are adapted from the GitHub repository
"https://github.com/charlesrouge/SampleVis"

"""


def scale_normalized(sample, bounds):
    """
    Rescales the sample space into the unit hypercube, bounds = [0,1]
    """
    scaled_sample = np.zeros(sample.shape)

    for j in range(sample.shape[1]):
        scaled_sample[:, j] = (sample[:, j] - bounds[j][0]) / (
            bounds[j][1] - bounds[j][0]
        )

    return scaled_sample


# Rescales a sample defined in the unit hypercube, to its bounds
def scale_to_bounds(scaled_sample, bounds):
    sample = np.zeros(scaled_sample.shape)

    for j in range(sample.shape[1]):
        sample[:, j] = (
            scaled_sample[:, j] * (bounds[j][1] - bounds[j][0]) + bounds[j][0]
        )

    return sample


def binning(tab, vect):
    """
    Discretizes value from tab depending on how they fall on a scale defined by vec
    Returns binned_tab, with the same shape as tab
    Example if vec = [0,1], binned_tab[i,j]=0 if tab[i,j]<=0, =1 if 0<tab[i,j]<=1,
    =2 otherwise
    """
    binned_tab = np.zeros(tab.shape)

    for i in range(len(vect)):
        binned_tab = binned_tab + 1 * (tab > vect[i] * np.ones(tab.shape))

    return binned_tab


def correlation_plots(z_test, p_val, test_name, var_names):
    """
    Get and plot results for within-sample correlation test, based on
    1) test results z_test (Figure 1)
    2) statistical significance pval (Figure 2)
    other inputs are the test_name and var_names
    """
    # Local variables
    nvar = len(var_names)
    pval = 1 - (
        p_val + np.matlib.eye(nvar)
    )  # Transformation convenient for plotting below

    ###################################################
    # Figure 1: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))
    res_mat[:, 0:-1] = z_test

    # Center the color scale on 0
    res_mat[0, nvar] = max(np.amax(z_test), -np.amin(z_test))
    res_mat[1, nvar] = -res_mat[0, nvar]

    # Plotting Pearson test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=plt.cm.bwr)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.set_xticks(np.linspace(0.5, nvar - 0.5, num=nvar))
    ax.set_xticklabels(var_names, rotation=20, ha="right")
    ax.set_yticks(np.linspace(0.5, nvar - 0.5, num=nvar))
    ax.set_yticklabels(var_names[::-1])
    ax.tick_params(axis="x", top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=False)
    plt.title("Rank correlation between variables' sampled values", size=13, y=1.07)
    plt.colorbar()
    plt.show()
    # plt.savefig(test_name + '_cross_correlation.png')
    # plt.clf()

    ###################################################
    # Figure 2: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))

    # Set the thresholds at +-95%, 99%, and 99.9% significance levels
    bin_thresholds = [0.9, 0.95, 0.99, 0.999]
    n_sig = len(bin_thresholds)
    res_mat[:, 0:-1] = binning(pval, bin_thresholds)

    # Set the color scale
    res_mat[0, nvar] = n_sig

    # Common color map
    cmap = plt.cm.Greys
    cmaplist = [cmap(0)]
    for i in range(n_sig):
        cmaplist.append(cmap(int(255 * (i + 1) / n_sig)))
    mycmap = cmap.from_list("Custom cmap", cmaplist, n_sig + 1)

    # Plot background mesh
    mesh_points = np.linspace(0.5, nvar - 0.5, num=nvar)
    for i in range(nvar):
        plt.plot(
            np.arange(0, nvar + 1),
            mesh_points[i] * np.ones(nvar + 1),
            c="k",
            linewidth=0.3,
            linestyle=":",
        )
        plt.plot(
            mesh_points[i] * np.ones(nvar + 1),
            np.arange(0, nvar + 1),
            c="k",
            linewidth=0.3,
            linestyle=":",
        )

    # Plotting MK test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=mycmap)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.set_xticks(mesh_points)
    ax.set_xticklabels(var_names, rotation=20, ha="right")
    ax.set_yticks(mesh_points)
    ax.set_yticklabels(var_names[::-1])
    ax.tick_params(axis="x", top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=False)
    plt.title("Significance of the rank correlations", size=13, y=1.07)
    colorbar = plt.colorbar()
    colorbar.set_ticks(
        np.linspace(res_mat[0, nvar] / 10, 9 * res_mat[0, nvar] / 10, num=n_sig + 1)
    )
    cb_labels = ["None"]
    for i in range(n_sig):
        cb_labels.append(str(bin_thresholds[i] * 100) + "%")
    colorbar.set_ticklabels(cb_labels)
    plt.show()
    # plt.savefig(test_name + '_significance.png')
    # plt.clf()

    return None


def pearson_test_sample(sample):
    """
    Correlation Pearson test for whole sample. Outputs are:
    the Pearson statistic rho
    the p-value pval
    """
    # Local variables
    var = sample.shape[1]
    rho = np.zeros((var, var))
    pval = np.zeros((var, var))

    # Pearson test results
    for i in range(var):
        for v in np.arange(i + 1, var):
            [rho[i, v], pval[i, v]] = pearsonr(sample[:, i], sample[:, v])
            [rho[v, i], pval[v, i]] = [rho[i, v], pval[i, v]]

    return [rho, pval]


def mann_kendall_test(y, prec):
    """
    Mann-Kendall test (precision is the number of decimals)
    Outputs are the normalized statistic Z and the associated p-value
    """
    n = len(y)
    x = np.int_(y * (10**prec))

    # Sign matrix and ties
    sm = np.zeros((n - 1, n - 1))
    for i in range(n - 1):
        sm[i, i:n] = np.sign(x[i + 1 : n] - x[0 : n - 1 - i])  # noqa E203

    # Compute MK statistic
    s = np.sum(sm)

    # Count ties and their c
    # appel Mimiontributions to variance of the MK statistic
    [val, count] = np.unique(x, return_counts=True)
    [extent, ties] = np.unique(count, return_counts=True)
    tie_contribution = np.zeros(len(ties))
    for i in range(len(ties)):
        tie_contribution[i] = (
            ties[i] * extent[i] * (extent[i] - 1) * (2 * extent[i] + 5)
        )

    # Compute the variance
    vs = (n * (n - 1) * (2 * n + 5) - np.sum(tie_contribution)) / 18
    if vs < 0:
        print("WARNING: negative variance!!!")

    # Compute standard normal statistic
    z = (s - np.sign(s)) / np.sqrt(max(vs, 1))

    # Associated p-value
    pval = 1 - erf(abs(z) / np.sqrt(2))

    return [z, pval]


def mann_kendall_test_sample(sample):
    """
    Same as above, but for whole sample
    Outputs are the normalized statistic Z and the associated p-value
    """
    # Local variables
    n = sample.shape[0]
    var = sample.shape[1]
    x = np.argsort(
        sample, axis=0
    )  # Ranks of the values in the ensemble, for each variable
    mk_res = np.zeros((var, var))
    pval = np.zeros((var, var))

    # MK test results
    for i in range(var):
        reorder_sample = np.zeros((n, var))
        for j in range(n):
            reorder_sample[j, :] = sample[x[j, i], :]
        for v in np.arange(i + 1, var):
            [mk_res[i, v], pval[i, v]] = mann_kendall_test(reorder_sample[:, v], 5)
            [mk_res[v, i], pval[v, i]] = [mk_res[i, v], pval[i, v]]

    return [mk_res, pval]


def projection_1D(sample, var_names):
    """
    Assess the uniformity of each 1D projection of the sample
    Assumes bounds of sample are [0,1]**n
    """
    [n, dim] = sample.shape
    y = np.zeros(sample.shape)

    z_int = np.linspace(0, 1, num=n + 1)
    binned_sample = binning(sample, z_int)

    for i in range(n):
        y[i, :] = 1 * (np.sum(1 * (binned_sample == i + 1), axis=0) > 0)

    proj = np.sum(y, axis=0) / n

    plt.bar(np.arange(dim), proj)
    plt.ylim(0, max(1, 1.01 * np.amax(proj)))
    plt.xticks(np.arange(dim), var_names)
    plt.ylabel("Coverage of axis")
    plt.show()
    # plt.savefig('1D_coverage_index.png')
    # plt.clf()

    # Return a single index: the average of values for all the variables
    return np.mean(proj)


def projection_2D(sample, var_names):
    """
    Plots the sample projected on each 2D plane
    """
    dim = sample.shape[1]

    for i in range(dim):
        for j in range(dim):
            plt.subplot(dim, dim, i * dim + j + 1)
            plt.scatter(
                sample[:, j],
                sample[:, i],
                s=2,
            )
            if j == 0:
                plt.ylabel(var_names[i], rotation=0, ha="right")
            if i == dim - 1:
                plt.xlabel(var_names[j], rotation=20, ha="right")

            plt.xticks([])
            plt.yticks([])
    plt.show()
    # plt.savefig('2D-projections.png')
    # plt.clf()

    return None


def space_filling_measures_discrepancy(sample):
    """
    Assumes the samle has N points (lines) in p dimensions (columns)
    Assumes sample drawn from unit hypercube of dimension p
    L2-star discrepancy formula from

    """

    [n, p] = sample.shape

    # First term of the L2-star discrepancy formula
    dl2 = 1.0 / (3.0**p)

    # Second term of the L2-star discrepancy formula
    sum_1 = 0.0
    for k in range(n):
        for i in range(n):
            prod = 1.0
            for j in range(p):
                prod = prod * (1 - max(sample[k, j], sample[i, j]))
            sum_1 = sum_1 + prod
    dl2 += sum_1 / n**2

    # Third term of the L2-star discrepancy formula
    sum_2 = 0.0
    for i in range(n):
        prod = 1
        for j in range(p):
            prod *= 1 - sample[i, j] ** 2
        sum_2 = sum_2 + prod
    dl2 -= sum_2 * (2 ** (1 - p)) / n

    return dl2


def space_filling_measures_min_distance(sample, show):
    """
    Returns the minimal distance between two points
    Assumes sample drawn from unit hypercube of dimension p
    """
    n = sample.shape[0]
    dist = np.ones(
        (n, n)
    )  # ones and not zeros because we are interested in abstracting min distance

    # Finding distance between points
    for i in range(n):
        for j in np.arange(i + 1, n):
            dist[i, j] = np.sqrt(np.sum((sample[i, :] - sample[j, :]) ** 2))
            dist[j, i] = dist[i, j]  # For the plot

    # If wanted: plots the distribution of minimal distances from all points
    if show == 1:
        plt.plot(np.arange(1, n + 1), np.sort(np.amin(dist, axis=1)))
        plt.xlim(1, n)
        plt.xlabel("Sorted ensemble members")
        plt.ylabel("Min Euclidean distance to any other point")
        plt.show()
        # plt.savefig('Distances.png')
        # plt.clf()

    # Space-filling index is minimal distance between any two points
    return np.amin(dist)
