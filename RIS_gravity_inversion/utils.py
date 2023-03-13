import contextlib
import copy
import os
import sys
from typing import Union

import harmonica as hm
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import pygmt
import seaborn as sns
import verde as vd
import xarray as xr
from antarctic_plots import maps, profile, utils
from scipy.spatial import cKDTree
from tqdm_joblib import tqdm_joblib

from RIS_gravity_inversion import regional


def RMSE(data):
    """calculate the RMSE of data."""
    return np.sqrt(np.nanmedian(data**2).item())
    # return np.sqrt(np.nanmean(data**2).item())


def dist_nearest_points(
    targets: pd.DataFrame,
    data: Union[pd.DataFrame, xr.DataArray, xr.Dataset],
    coord_names=["easting", "northing"],
):
    """
    calculate the distance to the nearest target for a series of points.
    """

    df_targets = targets[coord_names].copy()

    if isinstance(data, pd.DataFrame):
        df_data = data[coord_names].copy()
    elif isinstance(data, xr.DataArray):
        df_data = vd.grid_to_table(data)[coord_names]
    elif isinstance(data, xr.Dataset):
        try:
            df_grid = vd.grid_to_table(data[list(data.variables)[0]])
        except IndexError:
            df_grid = vd.grid_to_table(data)
        df_data = df_grid[coord_names].copy()

    min_dist, min_dist_idx = cKDTree(df_targets.values).query(df_data.values, 1)

    df_data["min_dist"] = min_dist

    if isinstance(data, pd.DataFrame):
        return df_data
    elif isinstance(data, (xr.DataArray, xr.Dataset)):
        return df_data.set_index(coord_names[::-1]).to_xarray()


def weight_grid(
    grid: xr.DataArray,
    low_lim: float,
    high_lim: float,
    low: float = 0,
    high: float = 1,
    between: Union[str, float] = "NaN",
    interp_type: str = "linear",
    tension_factor: float = None,
    radius: float = None,
    epsg: str = None,
):
    """
    create a grid with values from low to high based on if z is above or below
    thresholds, low_lim and high_lim.

    interp_type can be "linear", "nearest", or "spline"
    """
    grid = grid.copy()

    weights = pygmt.grdclip(
        grid,
        above=[high_lim, high],  # if z > high_lim, set to high (1 by default)
        below=[low_lim, low],  # if z < low_lim, set to low (0 by default)
        between=[low_lim, high_lim, between],  # if z is between, set to NaN
    )

    if between == "NaN":
        if interp_type == "linear":
            weights = weights.rio.write_crs(f"epsg:{epsg}").rio.interpolate_na(
                method=interp_type
            )
        elif interp_type == "nearest":
            if radius is None:
                mode = "n"
            else:
                mode = f"n{radius}"
            weights = pygmt.grdfill(weights, mode=mode, verbose="q")
        elif interp_type == "spline":
            if tension_factor is None:
                mode = "s"
            else:
                mode = f"s{tension_factor}"
            weights = pygmt.grdfill(weights, mode=mode, verbose="q")

    # ensure data is between low and high
    weights = pygmt.grdclip(weights, above=[high, high], below=[low, low])

    df = vd.grid_to_table(weights)

    grid = df.set_index(["y", "x"]).to_xarray()

    return grid


def constraints_grid(
    constraint_points: pd.DataFrame,
    grid: xr.DataArray,
    inner_bound: float,
    outer_bound: float,
    low: float = 0,
    high: float = 1,
    between: Union[str, float] = "NaN",
    coord_names=["easting", "northing"],
    interp_type: str = "linear",
    tension_factor: float = None,
    radius: float = None,
    epsg: str = None,
    plot: bool = False,
):
    """
    create a grid of weights based on distance from nearest constraint point
    """
    if coord_names == ["x", "y"]:
        grid = grid.copy()
    else:
        constraint_points = constraint_points.rename(
            columns={coord_names[0]: "x", coord_names[1]: "y"}
        ).copy()
        grid = grid.rename({coord_names[0]: "x", coord_names[1]: "y"}).copy()

    min_dist = dist_nearest_points(
        targets=constraint_points,
        data=grid,
        coord_names=["x", "y"],
    )

    weights = weight_grid(
        min_dist.min_dist,
        low_lim=inner_bound,
        high_lim=outer_bound,
        interp_type=interp_type,
        low=low,
        high=high,
        radius=radius,
        tension_factor=tension_factor,
        epsg=epsg,
    )

    if plot is True:
        fig = maps.plot_grd(
            grid.weights,
            cmap="dense",
            points=constraint_points,
            points_style="c0.2c",
        )
        fig.show()
    weights = weights.rename({"x": coord_names[0], "y": coord_names[1]})
    min_dist = min_dist.rename({"x": coord_names[0], "y": coord_names[1]})

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
        reducer = vd.BlockReduce(
            reduction=np.median,
            spacing=spacing,
            center_coordinates=False,  # center of block, or block median of coords
            adjust="spacing",  # adjust "spacing" or "region"
            drop_coords=False,
        )

        coordinates, data = reducer.filter(
            coordinates=(grav.easting, grav.northing, grav.upward),
            data=(grav.Gobs,),
        )

        grav = pd.DataFrame(
            data={
                "easting": coordinates[0],
                "northing": coordinates[1],
                "upward": coordinates[2],
                "Gobs": data[0],
            },
        )

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


def eq_sources_score(params, coordinates, data, **kwargs):
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
        )
    )
    return score


def parallel_eq_sources_scores(parameter_sets, coordinates, data, **kwargs):
    n_jobs = len(psutil.Process().cpu_affinity())

    with tqdm_joblib(
        desc="Calculating scores", total=len(parameter_sets)
    ) as progress_bar:  # noqa
        scores = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(eq_sources_score)(i, coordinates, data, **kwargs)
            for i in parameter_sets
        )
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


def eq_sources_best(
    parameter_sets, coordinates, data, region, spacing, height=None, **kwargs
):
    """
    Test a suite of damping and depth parameters, pick the best resulting parameters,
    and predict the gravity on a regular grid. Set the observation height to upwards
    continue to, or use the max height of the orignal data (default).
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
        coordinates=(surface.easting.values, surface.northing.values),
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
                fig = maps.plot_grd(
                    grids[n],
                    cmap="vik",
                    cbar_label="mGal",
                    title=n,
                    coast=False,
                    hist=True,
                    cbar_yoffset=3,
                )
            else:
                fig = maps.plot_grd(
                    grids[n],
                    cmap="vik",
                    cbar_label="mGal",
                    title=n,
                    coast=False,
                    hist=True,
                    cbar_yoffset=3,
                    fig=fig,
                    origin_shift="xshift",
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
    if lower_confining_layer is not None:
        df = profile.sample_grids(
            df=df,
            grid=lower_confining_layer,
            name="lower_bounds",
            coord_names=["easting", "northing"],
        )
    return df


def enforce_confining_surface(
    prisms_df: pd.DataFrame,
    Surface_correction: np.array,
    upper_confining_layer: xr.DataArray = None,
    lower_confining_layer: xr.DataArray = None,
):
    """
    ensure the surface correction doesn't move the prisms above or below optional
    confining surfaces
    """
    correction = Surface_correction.copy()

    if upper_confining_layer is not None:
        # get max change in positive direction for each prism
        prisms_df["max_allowed_change_above"] = prisms_df.upper_bounds - prisms_df.top
        for i, j in enumerate(prisms_df.max_allowed_change_above):
            if correction[i] > j:
                correction[i] = j
    if lower_confining_layer is not None:
        # get max change in negative direction for each prism
        prisms_df["max_allowed_change_below"] = (
            prisms_df.lower_bounds - prisms_df.bottom
        )
        for i, j in enumerate(prisms_df.max_allowed_change_below):
            if correction[i] < j:
                correction[i] = j

    return correction


def constrain_surface_correction(
    prisms_df: pd.DataFrame,
    Surface_correction: np.array,
    upper_confining_layer: xr.DataArray = None,
    lower_confining_layer: xr.DataArray = None,
    max_layer_change_per_iter: float = None,
):
    """
    optionally constrain the surface correction and add to the prisms dataframe.
    """

    correction = Surface_correction.copy()
    prisms = prisms_df.copy()

    # set maximum allowed change of each prism per iteration
    if max_layer_change_per_iter is not None:
        for i, j in enumerate(correction):
            if j > max_layer_change_per_iter:
                j = max_layer_change_per_iter
            elif j < -max_layer_change_per_iter:
                j = -max_layer_change_per_iter
        print(
            "Layer correction (after clipped) median:",
            f"{int(np.median(correction))}m,",
            f"RMSE:{int(RMSE(correction))} m",
        )

    # ensure prisms don't extend above or below confining surfaces.
    correction = enforce_confining_surface(
        prisms,
        correction,
        upper_confining_layer,
        lower_confining_layer,
    )

    return correction


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
