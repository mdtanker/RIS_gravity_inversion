import warnings

import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from antarctic_plots import fetch, maps, profile, utils
from scipy.sparse.linalg import lsqr
import copy
import functools
import time

warnings.filterwarnings("ignore", message="pandas.Int64Index")
warnings.filterwarnings("ignore", message="pandas.Float64Index")


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

def import_layers(
    layers_list: list,
    spacing_list: list,
    rho_list: list,
    fname_list: list,
    grav_spacing: float,
    active_layer: str,
    buffer_region: list,
    inversion_region: list,
    grav_file: str,
    **kwargs,
):
    """
    Read zarrs and csvs, resample/rename to be uniform, and add to layers dictionary

    Parameters
    ----------
    layers_list : list
        list of names of layers
    spacing_list : list
        list of grid spacing of layers, in meters
    rho_list : list
        list of densities to use for layers, in kg/m3
    fname_list : list
        list of zarr file names for input grids
    grav_spacing : float
        spacing to resample gravity grid at
    active_layer : str
        layer which will be inverted for
    buffer_region : list
        inversion region with a buffer zone included
    inversion_region : list
       inversion region
    grav_file : str
        csv file with gravity point observation data

    Other Parameters
    ----------------
    constraint_grid: str
        .nc file of a constraint grid, 0-1.
    constraint_points: str
        .csv file with position of constraints
    input_grav_name: str
        column name of observed gravity / freeair / disturbance
    input_obs_height_name: str
        column name of gravity station elevation
    block_reduction: str
        choose type of block reduction to apply to gravity data
    registration: str
        choose grid registration type for input layers and constraints grid
    Returns
    -------
    tuple
        layers (dict),
        grav (pd.DataFrame),
        constraint_grid (xr.DataArray),
        constraint_points (pd.DataFrame),
        constraint_points_RIS (pd.DataFrame)
    """

    constraint_grid = kwargs.get("constraint_grid", None)
    constraint_points = kwargs.get("constraint_points", None)

    input_grav_name = kwargs.get("input_grav_name", "gravity_disturbance")
    input_obs_height_name = kwargs.get("input_obs_height_name", "ellipsoidal_elevation")

    layers_list = pd.Series(layers_list)
    spacing_list = pd.Series(spacing_list)
    rho_list = pd.Series(rho_list)
    fname_list = pd.Series(fname_list)

    # read gravity csv file
    df = pd.read_csv(
        grav_file,
        sep=",",
        header="infer",
        index_col=None,
        compression="gzip",
    )

    # remove other columns
    df = df[["x", "y", input_grav_name, input_obs_height_name]]

    # get number of grav points before reduction

    prior_len = len(df[input_grav_name])
    # block reduce gravity data
    if kwargs.get("block_reduction", None) is None:
        grav = df.copy()
    elif kwargs.get("block_reduction", None) == "pygmt":
        grav = pygmt.blockmedian(
            df[["x", "y", input_grav_name]],
            spacing=grav_spacing,
            region=inversion_region,
            registration="p",
        )
        grav[input_obs_height_name] = pygmt.blockmedian(
            df[["x", "y", input_obs_height_name]],
            spacing=grav_spacing,
            region=inversion_region,
            registration="p",
        )[input_obs_height_name]
    elif kwargs.get("block_reduction", None) == "verde":
        reducer_mean = vd.BlockReduce(
            reduction=np.median,
            spacing=grav_spacing,
            center_coordinates=False,
            adjust="region",
        )
        coordinates, data = reducer_mean.filter(
            coordinates=(df.x, df.y),
            data=(
                df[input_grav_name],
                df[input_obs_height_name],
            ),
        )
        blocked = pd.DataFrame(
            data={
                "x": coordinates[0],
                "y": coordinates[1],
                input_grav_name: data[0],
                input_obs_height_name: data[1],
            },
        )
        grav = blocked[vd.inside((blocked.x, blocked.y), inversion_region)].copy()

    post_len = len(grav[input_grav_name])

    if kwargs.get("block_reduction", None) is None:
        print(f"{prior_len} gravity observation points")
    else:
        print(f"Block-reduced the gravity data at {int(grav_spacing)}m spacing")
        print(f"from {prior_len} points to {post_len} points")

    # center gravity around 0
    grav[input_grav_name] -= grav[input_grav_name].mean()

    # set standard names
    grav.rename(
        columns={input_grav_name: "Gobs", input_obs_height_name: "z"},
        inplace=True,
    )

    # make nested dictionary for layers and properties
    layers = {
        j: {"spacing": spacing_list[i], "fname": fname_list[i], "rho": rho_list[i]}
        for i, j in enumerate(layers_list)
    }

    # read and resample layer grids, convert to dataframes
    for k, v in layers.items():
        tmp_grid = xr.open_zarr(v["fname"]).to_array().squeeze()
        print(f"\n{'':*<20}Resampling {k} layer {'':*>20}")
        v["grid"] = fetch.resample_grid(
            tmp_grid,
            spacing=v["spacing"],
            region=buffer_region,
            registration=kwargs.get("registration", "p"),
            verbose="q",
        )

        # print spacing, region, max,min, and registration of layers
        print(f"{k} info: {utils.get_grid_info(v['grid'])}")

        v["df"] = v["grid"].to_dataframe().reset_index()
        v["df"]["rho"] = v["rho"]
        v["df"].dropna(how="any", inplace=True)
        v["len"] = len(v["df"].x)

    if constraint_grid is not None:
        # open zarr file
        tmp_grid = xr.open_zarr(constraint_grid).to_array().squeeze()
        # resample constraint grid
        print(f"\n{'':*<20}Resampling constraints grid {'':*>20}")
        constraint_grid = fetch.resample_grid(
            tmp_grid,
            spacing=layers[active_layer]["spacing"],
            region=buffer_region,
            registration=kwargs.get("registration", "p"),
            verbose="q",
        )
        # print spacing, region, max,min, and registration of constraint grid
        print(f"constraint grid: {utils.get_grid_info(constraint_grid)}")

    if constraint_points is not None:
        # load constraint points into a dataframe, and create a subset within region
        constraint_points_all = pd.read_csv(
            constraint_points,
            sep=",",
            header="infer",
            index_col=None,
            compression="gzip",
        )

        constraint_points = utils.points_inside_region(
            constraint_points_all,
            inversion_region,
        )

        # additional subset within RIS polygon
        # mask = utils.mask_from_shp(
        #     "plotting/RIS_outline.shp",
        #     masked=True,
        #     invert=False,
        #     region=buffer_region,
        #     spacing=1e3,
        # )
        # mask.to_netcdf("tmp_outputs/tmp_mask.nc")

        constraint_points_RIS = pygmt.select(
            data=constraint_points,
            # gridmask="tmp_outputs/tmp_mask.nc",
            F="plotting/RIS_outline.shp",
            coltypes="o",
        )

        # constraint_points_RIS = constraint_points_RIS.astype(np.float64)

    print(f"gravity: {len(grav)} points")
    print(f"gravity avg. elevation: {int(np.nanmean(grav.z))}")

    if constraint_points is not None:
        print(f"bathymetry control points:{len(constraint_points)}")
        print(f"bathymetry control points within RIS:{len(constraint_points_RIS)}")

    grav.Gobs = grav.Gobs.astype(np.float64)
    grav.z = grav.z.astype(np.float64)
    grav.x = grav.x.astype(np.float64)
    grav.y = grav.y.astype(np.float64)

    outputs = [layers, grav, None, None, None]

    if constraint_grid is not None:
        outputs[2] = constraint_grid
    if constraint_points is not None:
        outputs[3] = constraint_points
        outputs[4] = constraint_points_RIS

    return outputs


def grids_to_prism_layers(
    layers: dict,
    thickness_threshold: float = 1,
):
    """
    Turn nested dictionary of grids into series of vertical prisms between each layer.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
    thickness_threshold : float, optional
        remove prisms with thickness less than this threshold, in meters, by default 1
    """

    # buffer region defaults to first layer's extent
    buffer_region = utils.get_grid_info(list(layers.values())[0]["grid"])[1]

    # add density variable to datasets
    for k, v in layers.items():
        v["grid"]["density"] = v["grid"].copy()
        v["grid"].density.values[:] = v["rho"]

    # list of layers, bottom up
    # reversed_layers_list = layers_list.iloc[::-1]
    reversed_layers_list = pd.Series([k for k, v in layers.items()]).iloc[::-1]

    # create prisms layers from input grids
    for i, j in enumerate(reversed_layers_list):
        # bottom-most prism layer
        if i == 0:
            # tops of prisms are from current grid
            surface = layers[j]["grid"]
            # base of prisms
            # reference=-50e3,
            reference = np.nanmin(layers[j]["grid"].values)
            layers[j]["prisms"] = hm.prism_layer(
                coordinates=(layers[j]["grid"].x.values, layers[j]["grid"].y.values),
                surface=surface,
                reference=reference,
                properties={
                    "density": layers[j]["grid"].density,
                    "thickness": surface - reference,
                },
            )
            print(
                f"{'':*<10} {j} top: {int(np.nanmean(layers[j]['prisms'].top.values))}m"
                f" and bottom: {int(np.nanmean(layers[j]['prisms'].bottom.values))}m "
                f"{'':*>10}"
            )
        # 2nd to bottom layer, moving upwards to surface layer
        else:
            # if spacing of current layer doesn't match below layer's spacing, sample
            # lower layer to get values for bottoms of prisms.
            if (
                layers[j]["spacing"]
                != layers[reversed_layers_list.iloc[i - 1]]["spacing"]
            ):
                print(
                    f"resolutions don't match for {j} ({layers[j]['spacing']}m) and "
                    f"{reversed_layers_list.iloc[i-1]} "
                    f"({layers[reversed_layers_list.iloc[i-1]]['spacing']}m)"
                )
                print(
                    f"sampling {reversed_layers_list.iloc[i-1]} at"
                    f" {j} prism locations"
                )
                tmp = layers[j]["grid"].to_dataframe().reset_index()
                tmp_regrid = pygmt.grdtrack(
                    points=tmp[["x", "y"]],
                    grid=layers[reversed_layers_list.iloc[i - 1]]["grid"],
                    newcolname="z_regrid",
                    verbose="q",
                )
                tmp["z_low"] = tmp.merge(tmp_regrid, how="left", on=["x", "y"]).z_regrid
                tmp_grd = pygmt.xyz2grd(
                    tmp[["x", "y", "z_low"]],
                    region=buffer_region,
                    registration="p",
                    spacing=layers[j]["spacing"],
                )
                surface = layers[j]["grid"]
                reference = tmp_grd
                layers[j]["prisms"] = hm.prism_layer(
                    coordinates=(
                        layers[j]["grid"].x.values,
                        layers[j]["grid"].y.values,
                    ),
                    surface=surface,
                    reference=reference,
                    properties={
                        "density": layers[j]["grid"].density,
                        "thickness": surface - reference,
                    },
                )
                print(
                    f"{'':*<10} {j} top: {int(np.nanmean(layers[j]['prisms'].top.values))}"  # noqa
                    f"m and bottom: {int(np.nanmean(layers[j]['prisms'].bottom.values))}"  # noqa
                    f"m {'':*>10}\n"
                )
            else:
                surface = layers[j]["grid"]
                reference = layers[reversed_layers_list.iloc[i - 1]]["grid"]
                layers[j]["prisms"] = hm.prism_layer(
                    coordinates=(
                        layers[j]["grid"].x.values,
                        layers[j]["grid"].y.values,
                    ),
                    surface=surface,
                    reference=reference,
                    properties={
                        "density": layers[j]["grid"].density,
                        "thickness": surface - reference,
                    },
                )
                print(
                    f"{'':*<10} {j} top: {int(np.nanmean(layers[j]['prisms'].top.values))}"  # noqa
                    f"m and bottom: {int(np.nanmean(layers[j]['prisms'].bottom.values))}"  # noqa
                    f"m {'':*>10}\n"
                )

    # # drop prisms with thicknesses < threshold
    # total_before = []
    # total_after = []

    # for k, v in layers.items():
    #     print(f"\n{'':*<10}{k} layer{'':*>10}")

    #     # get number of prisms
    #     before = np.count_nonzero(~np.isnan(v['prisms'].thickness.values))

    #     # drop prisms
    #     v['prisms'] = drop_thin_prisms(v['prisms'], thickness_threshold)

    #     # get number of prisms
    #     after = np.count_nonzero(~np.isnan(v['prisms'].thickness.values))

    #     total_before.append(before)
    #     total_after.append(after)

    # print(f"\ntotal number of prisms: {sum(total_before)}")
    # print(f"prisms with thickness > {thickness_threshold}m: {sum(total_after)}")


def drop_thin_prisms(
    prism_layer: xr.Dataset,
    threshold: float = 1,
):
    """
    drop prisms from dataset which have a thickness smaller than a set threshold

    Parameters
    ----------
    prism_layer : xr.Dataset
        prism layer to drop thin prisms for
    threshold : float, optional
       max thickness in meters, by default 1

    Returns
    -------
    xr.Dataset
        returns a dataset where prisms which were < threshold are have NaN's
    """
    # get number of prisms
    num_before = np.count_nonzero(~np.isnan(prism_layer.thickness.values))

    # prism_layer=prism_layer.where(
    #         prism_layer.thickness >= threshold,
    #         )
    # drop prisms with thicknesses < threshold
    for k in list(prism_layer.variables):
        if k in ("easting", "northing"):
            pass
        else:
            prism_layer[k] = prism_layer[k].where(
                prism_layer.thickness >= threshold,
            )

    # get number of prisms
    num_after = np.count_nonzero(~np.isnan(prism_layer.thickness.values))

    print(f"removed {num_before-num_after} prisms with thickness < {threshold}.")

    return prism_layer


def forward_grav_layers(
    layers: dict,
    gravity: pd.DataFrame,
    exclude_layers: list = None,
    **kwargs,
):
    """
    Calculate forward gravity of layers of prisms.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
    gravity : pd.DataFrame
       locations to calculate forward gravity at. needs variables 'x', 'y', and 'z'.
    exclude_layers : list, optional
        list of layers to exclude from total forward gravity, useful if applying a
        Bouguer correction with a layer's calculated forward gravity, by default None

    Other Parameters
    ----------------
    progressbar: bool
        display a progress bar for the calculation
    parallel: bool
        use parallel processing to speed up calculation

    Returns
    -------
    pd.DataFrame
        Returns the input dataframe with forward gravity of individual and combined
        layers
    """

    df_forward = gravity.copy()

    # calculate forward gravity of each layer of prisms
    for k, v in layers.items():
        df_forward[f"{k}_forward_grav"] = v["prisms"].prism_layer.gravity(
            coordinates=(df_forward.x, df_forward.y, df_forward.z),
            field="g_z",
            progressbar=kwargs.get("progressbar", True),
            density_name="density",
            parallel=kwargs.get("parallel", True),
        )
        # subtract mean from each layer
        df_forward[f"{k}_forward_grav"] -= df_forward[f"{k}_forward_grav"].mean()
        prism_count = np.count_nonzero(~np.isnan(v["prisms"].thickness.values))
        print(f"{prism_count} prisms in {k} layer")
        print(f"finished {k} layer")

    # make list of layers to include (not layers used in Bouguer correction)
    if exclude_layers is not None:
        include_forward_layers = pd.Series(
            [k for k, v in layers.items() if k not in exclude_layers]
        )
    else:
        include_forward_layers = pd.Series(k for k, v in layers.items())

    # add gravity effects of all input layers
    grav_layers_list = [f"{i}_forward_grav" for i in include_forward_layers]
    df_forward["forward_total"] = df_forward[grav_layers_list].sum(axis=1, skipna=True)

    return df_forward


def anomalies(
    layers: dict,
    input_grav: pd.DataFrame,
    grav_spacing: int,
    regional_method: str,
    crs: str = "3031",
    **kwargs,
):
    """
    Calculate the residual gravity anomaly from 1 of 4 methods. Starting with the
    misfit between observed and forward gravity, remove the regional misfit field to get
    the residual misfit. The regional misfit is calculated from either:
    1) a 2D polynomial trend of the misfit, with a user-input degree order,
    2) filtering the misfit with a user-defined filter,
    3) sampling the misfit at known constraint points, and grid the misfit at these
        points with a user-defined tension factor, or
    4) use the equivalent source technique to predict the regional field.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
            'len': int, number of prisms in the layer
            'prisms': xarray.DataSet, harmonica.prism_layer
    input_grav : pd.DataFrame
        input gravity data
    grav_spacing : int
        spacing of gravity data, to use for make misfit grid and plotting
    regional_method : {'trend', 'filter', 'constraints', 'eq_sources'}
        choose a method to determine the regional gravity misfit.
    crs : str, optional
        if fill_method = 'rioxarray', set the coordinate reference system to be used
            in rioxarray, by default "3031"
    Returns
    -------
    _type_
        _description_
    """
    """

    Other Parameters
    ----------------
    input_grav_column: str,
        name of the column which contains the observed gravity, by default is 'Gobs'
    input_forward_column: str,
        name of the column which contains the total forward gravity, by default is
        'forward_total'
    corrections: np.ndarray,
        list of layers to include in partial Bouguer correction of Observed
        gravity data.
    inversion_region: np.ndarray or str,
        GMT format region for the inverion, by default is extent of gravity data
    trend: int,
        trend order used from calculating regional misfit if
        regional_method = 'trend'.
    filter: str,
        input string for pygmt.grdfilter() for calculating regional misfit if
        regional_method = 'filter', ex. "g200e3" gives a 200km Gaussian filter.
    constraints: pd.DataFrame,
        Locations of constraint points to interpolate between for calculating
        regional misfit if regional_method = 'constraints'.
    fill_method: {'pygmt', 'rioxarray'},
        Choose method to fill nans, by default is 'pygmt'
    tension_factor: float,
    damping: float
        smoothness to impose on estimated coefficients
    block_size: float
        block reduce the data to speed up
    depth_type: str
        "relative" or "constant" for depth for eq_sources
    Returns
    -------
    pd.DataFrame
        Returns the input gravity dataframe with 4 additional columns:
        'grav_corrected': Observed gravity, or if 'corrections' is not None, then
            Observed gravity with the forward gravity of the specified layers removed.
        'misfit': 'gravity_corrected' - total forward gravity
        'reg': regional gravity misfit
        'res': residual graivty misift
    """
    # must provide kwargs with same name as method type.
    # for example, if regional_method='trend', must provide a trend order via the kwarg
    # 'trend'=6.
    # for constraints, kwarg is a dataframe of constraint points
    # for filter, kwarg is a pygmt filter string, such as "g150e3" for a 150km gaussian
    # for eq_sources, kwargs is depth of sources in meters
    if kwargs.get(regional_method) is None:
        raise ValueError(
            f"Must provide keyword argument '{regional_method}' if regional_method ="
            f" {regional_method}."
        )

    input_forward_column = kwargs.get("input_forward_column", "forward_total")
    input_grav_column = kwargs.get("input_grav_column", "Gobs")
    corrections = kwargs.get("corrections", None)

    # if inversion region not supplied, extract from dataframe
    inversion_region = kwargs.get(
        "inversion_region", vd.get_region((input_grav.x, input_grav.y))
    )

    # get kwargs associated with the various methods
    trend = kwargs.get("trend", None)
    filter = kwargs.get("filter", None)
    df = kwargs.get("constraints", None)
    if df is not None:
        constraints = df.copy()
    eq_sources = kwargs.get("eq_sources", None)

    crs = kwargs.get("crs", None)
    fill_method = kwargs.get("fill_method", "pygmt")

    # if fill_method == "rioxarray" and crs is None:
    if crs is None:
        # raise ValueError(
        print("'crs' not provided for rioxarray fill_method, defaulting to EPSG:3031\n")
        crs = 3031

    anomalies = input_grav.copy()

    # if anomalies already calculated, drop the columns
    try:
        anomalies.drop(columns=["misfit", "reg", "res"], inplace=True)
    except KeyError:
        pass

    # apply partial Bouguer correction of layers in 'corrections'
    if corrections is not None:
        print(f"applying bouguer correction for layers: {corrections}")
        col_list = []
        for i, j in enumerate(corrections):
            col_list.append(f"{j}_forward_grav")
        anomalies["boug_corr"] = anomalies[col_list].sum(axis=1)
        anomalies["grav_corrected"] = (
            anomalies[input_grav_column] - anomalies["boug_corr"]
        )
    else:
        print("no bouguer corrections applied")
        anomalies["grav_corrected"] = anomalies[input_grav_column]

    # get obs-forward misfit
    anomalies["misfit"] = anomalies.grav_corrected - anomalies[input_forward_column]

    # grid the misfits, used in trend, filter, and constraints, not in Eq. Sources
    misfit = pygmt.xyz2grd(
        data=anomalies[["x", "y", "misfit"]],
        region=inversion_region,
        spacing=grav_spacing,
        registration="p",
    )

    # Trend method
    if regional_method == "trend":
        # fill misfit nans with 1 of 2 methods
        if fill_method == "pygmt":
            # option 1) with pygmt.grdfill(), needs grav_spacing and inversion_region
            misfit_filled = pygmt.grdfill(misfit, mode="n")
        elif fill_method == "rioxarray":
            # option 1) with rio.interpolate(), needs crs set.
            # misfit = misfit.rio.write_crs(crs) # noqa
            misfit_filled = (
                misfit.rio.write_nodata(np.nan).rio.interpolate_na().rename("z")
            )  # noqa

        df = vd.grid_to_table(misfit_filled).astype("float64")

        trend = vd.Trend(degree=trend).fit((df.x, df.y.values), df.z)
        anomalies["reg"] = trend.predict((anomalies.x, anomalies.y))
        anomalies["res"] = anomalies.misfit - anomalies.reg

    # Filter method
    elif regional_method == "filter":
        # filter the observed-forward misfit with the provided filter in meters
        regional_misfit = pygmt.grdfilter(misfit, filter=filter, distance="0")
        # sample the results and merge into the anomalies dataframe
        tmp_regrid = pygmt.grdtrack(
            points=anomalies[["x", "y"]],
            grid=regional_misfit,
            newcolname="reg",
            verbose="q",
        )
        anomalies = anomalies.merge(tmp_regrid, on=["x", "y"], how="left")
        anomalies["res"] = anomalies.misfit - anomalies.reg

    # Constraints method
    elif regional_method == "constraints":
        # sample observed-forward misfit at constraint points
        constraints = profile.sample_grids(
            df=constraints,
            grid=misfit,
            name="misfit",
        )

        # get median misfit of constraint points in each 1km cell
        blocked = pygmt.blockmedian(
            data=constraints[["x", "y", "misfit"]],
            spacing=grav_spacing,
            region=inversion_region,
            registration="p",
        )

        # grid the entire region misfit based just on the misfit at the constraints
        regional_misfit = pygmt.surface(
            data=blocked,
            region=inversion_region,
            spacing=grav_spacing,
            registration="p",
            T=kwargs.get("tension_factor", 0.25),
            verbose="q",
        )

        # sample the resulting grid and add to anomalies dataframe
        anomalies = profile.sample_grids(
            df=anomalies,
            grid=regional_misfit,
            name="reg",
        )

        # calculate the residual
        anomalies["res"] = anomalies.misfit - anomalies.reg

    # Equivalent sources method
    elif regional_method == "eq_sources":
        # create set of deep sources
        equivalent_sources = hm.EquivalentSources(
            depth=eq_sources,
            damping=kwargs.get(
                "damping", None
            ),  # float: smoothness to impose on estimated coefficients
            block_size=kwargs.get(
                "block_size", None
            ),  # block reduce the data to speed up
            depth_type=kwargs.get(
                "depth_type", "relative"
            ),  # constant depths, not relative to observation heights
        )
        # fit the source coefficients to the data
        coordinates = (anomalies.x, anomalies.y, anomalies.z)
        equivalent_sources.fit(coordinates, anomalies.misfit)
        # use sources to predict the regional field at the observation points
        anomalies["reg"] = equivalent_sources.predict(coordinates)
        # calculate the residual field
        anomalies["res"] = anomalies.misfit - anomalies.reg

    RMS = round(np.sqrt((anomalies.res**2).mean(skipna=True)), 2)
    print(f"Root mean squared residual: {RMS}mGal")

    return anomalies


def density_inversion(
    density_layer,
    df_grav,
    layers,
    max_density_change=2000,
    grav_spacing=None,
    input_grav=None,
    buffer_reg=None,
    inv_reg=None,
    buffer_proj=None,
    plot=True,
):
    """
    Function to invert gravity anomaly to update a prism layer's density.
    density_layer: str; layer to perform inversion on
    max_density_change: int, maximum amount to change each prisms density by, in kg/m^3
    input_grav: xarray.DataSet
    plot: bool; defaults to True
    """
    if input_grav is None:
        input_grav = df_grav.Gobs_shift_filt
    # density in kg/m3
    forward_grav_layers(layers=layers, plot=False)
    # spacing = layers[density_layer]["spacing"]

    # df_grav['inv_misfit']=df_grav.Gobs_shift-df_grav[f'forward_grav_total']
    df_grav["inv_misfit"] = input_grav - df_grav["forward_grav_total"]

    # get prisms' coordinates from active layer
    prisms = layers[density_layer]["prisms"].to_dataframe().reset_index().dropna()

    print(f"active layer average density: {int(prisms.density.mean())}kg/m3")

    MAT_DENS = np.zeros([len(input_grav), len(prisms)])

    initial_RMS = round(np.sqrt((df_grav["inv_misfit"] ** 2).mean()), 2)
    print(f"initial RMS = {initial_RMS}mGal")
    print("calculating sensitivity matrix to determine density correction")

    prisms_n = []
    for x in range(len(layers[density_layer]["prisms"].easting.values)):
        for y in range(len(layers[density_layer]["prisms"].northing.values)):
            prisms_n.append(
                layers[density_layer]["prisms"].prism_layer.get_prism((x, y))
            )
    for col, prism in enumerate(prisms_n):
        MAT_DENS[:, col] = hm.prism_gravity(
            coordinates=(df_grav.x, df_grav.y, df_grav.z),
            prisms=prism,
            density=1,  # unit density
            field="g_z",
        )
    # Calculate shift to prism's densities to minimize misfit
    Density_correction = lsqr(MAT_DENS, df_grav.inv_misfit, show=False)[0]

    # for i,j in enumerate((input_grav)): #add tqdm for progressbar
    # MAT_DENS[i,:] = gravbox(
    #     df_grav.y.iloc[i],
    #     df_grav.x.iloc[i],
    #     df_grav.z.iloc[i],
    #     prisms.northing-spacing/2,
    #     prisms.northing+spacing/2,
    #     prisms.easting-spacing/2,
    #     prisms.easting+spacing/2,
    #     prisms.top,
    #     prisms.bottom,
    #     np.ones_like(prisms.density),
    # )  # unit density, list of ones
    # # Calculate shift to prism's densities to minimize misfit
    # Density_correction=lsqr(MAT_DENS,df_grav.inv_misfit,show=False)[0]*1000

    # apply max density change
    for i in range(0, len(prisms)):
        if Density_correction[i] > max_density_change:
            Density_correction[i] = max_density_change
        elif Density_correction[i] < -max_density_change:
            Density_correction[i] = -max_density_change

    # resetting the rho values with the above correction
    prisms["density_correction"] = Density_correction
    prisms["updated_density"] = prisms.density + prisms.density_correction
    dens_correction = pygmt.xyz2grd(
        x=prisms.easting,
        y=prisms.northing,
        z=prisms.density_correction,
        registration="p",
        region=buffer_reg,
        spacing=grav_spacing,
        projection=buffer_proj,
    )
    dens_update = pygmt.xyz2grd(
        x=prisms.easting,
        y=prisms.northing,
        z=prisms.updated_density,
        registration="p",
        region=buffer_reg,
        spacing=layers[density_layer]["spacing"],
        projection=buffer_proj,
    )
    initial_misfit = pygmt.xyz2grd(
        df_grav[["x", "y", "inv_misfit"]],
        region=inv_reg,
        spacing=grav_spacing,
        registration="p",
    )

    # apply the rho correction to the prism layer
    layers[density_layer]["prisms"]["density"].values = dens_update.values
    print(
        "average density:",
        f"{int(layers[density_layer]['prisms'].to_dataframe().reset_index().dropna().density.mean())}",  # noqa
        "kg/m3",
    )
    # recalculate forward gravity of active layer
    print("calculating updated forward gravity")
    df_grav[f"forward_grav_{density_layer}"] = layers[density_layer][
        "prisms"
    ].prism_layer.gravity(coordinates=(df_grav.x, df_grav.y, df_grav.z), field="g_z")

    # Recalculate of gravity misfit, i.e., the difference between calculated and
    # observed gravity
    df_grav["forward_grav_total"] = (
        df_grav.forward_grav_total
        - df_grav[f"{density_layer}_forward_grav"]
        + df_grav[f"forward_grav_{density_layer}"]
    )

    df_grav.inv_misfit = input_grav - df_grav.forward_grav_total
    final_RMS = round(np.sqrt((df_grav.inv_misfit**2).mean()), 2)
    print(f"RMSE after inversion = {final_RMS}mGal")
    final_misfit = pygmt.xyz2grd(
        df_grav[["x", "y", "inv_misfit"]],
        region=buffer_reg,
        registration="p",
        spacing=grav_spacing,
    )

    if plot is True:
        grid = initial_misfit
        fig = maps.plot_grd(
            grid=grid,
            cmap="polar+h0",
            cbar_label=f"initial misfit (mGal) [{initial_RMS}]",
        )

        grid = dens_correction
        fig = maps.plot_grd(
            grid=grid,
            cmap="polar+h0",
            cbar_label="density correction (kg/m3)",
            origin_shift="xshift",
        )

        grid = dens_update
        fig = maps.plot_grd(
            grid=grid,
            cmap="viridis",
            cbar_label="updated density (kg/m3)",
            origin_shift="xshift",
        )

        grid = final_misfit
        fig = maps.plot_grd(
            grid=grid,
            cmap="polar+h0",
            cbar_label=f"final misfit (mGal) [{final_RMS}]",
            origin_shift="xshift",
        )

        fig.show(width=1200)


def grav_column_der(x0, y0, z0, xc, yc, z1, z2, res, rho):
    """
    Function to calculate the vertical derivate of the gravitational acceleration cause
    by a right, rectangular prism.
    Approximate with Hammer's annulus approximation.
    x0, y0, z0: floats, coordinates of gravity observation points
    xc, yc, z1, z2: floats, coordinates of prism's y, x, top, and bottom, respectively.
    res: float, resolution of prism layer in meters,
    rho: float, density of prisms, in kg/m^3
    """
    r = np.sqrt((x0 - xc) ** 2 + (y0 - yc) ** 2)
    r1 = r - 0.5 * res
    r2 = r + 0.5 * res
    r1[r1 < 0] = 0  # will fail if prism is under obs point
    r2[r1 < 0] = 0.5 * res
    f = res**2 / (np.pi * (r2**2 - r1**2))  # eq 2.19 in McCubbine 2016 Thesis
    anomaly_grad = (
        0.0419
        * f
        * rho
        * (z1 - z0)
        * (
            1 / np.sqrt(r2**2 + (z1 - z0) ** 2)
            - 1 / np.sqrt(r1**2 + (z1 - z0) ** 2)
        )
    )
    return anomaly_grad


def jacobian_annular(
    gravity_data: pd.DataFrame,
    gravity_col: str,
    prisms: pd.DataFrame,
    spacing: float,
):
    """
    Function to calculate the Jacobian matrix using the annular cylinder approximation
    jacobian is matrix array with NG number of rows and NBath+NBase+NM number of columns
    uses vertical derivative of gravity to find least squares solution to minize gravity
    misfit for each grav station

    Parameters
    ----------
    gravity_data : pd.DataFrame
        dataframe containing gravity data
    gravity_col : str
        column of gravity_data with observed gravity
    prisms : pd.DataFrame
        dataframe of prisms coordinates with columns:
        easting, northing, top, and bottom in meters.
    spacing : float
        spacing of gravity data

    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
    """

    df = gravity_data
    jac = np.empty((len(df[gravity_col]), len(prisms)), dtype=np.float64)
    for i, j in enumerate((df[gravity_col])):
        jac[i, :] = grav_column_der(  # major issue here, way too slow
            df.y.iloc[i],  # coords of gravity observation points
            df.x.iloc[i],
            df.z.iloc[i],
            prisms.northing,
            prisms.easting,
            prisms.top,
            prisms.bottom,
            spacing,
            prisms.density / 1000,
        )
    return jac


def jacobian_prism(
    gravity_data: pd.DataFrame,
    gravity_col: str,
    model: xr.Dataset,
    delta: float,
    field: str,
):
    """
    Function to calculate the Jacobian matrix with the vertical gravity derivative
    as a numerical approximation with small prisms

    Parameters
    ----------
    gravity_data : pd.DataFrame
        dataframe containing gravity data
    gravity_col : str
        column of gravity_data with observed gravity
    model : xr.Dataset
        harmonica.prism_layer, with coordinates:
        easting, northing, top, and bottom, and variables: 'Density'.
    delta : float
        size of small prism to add, in meters
    field : str
        field to return, 'g_z' for gravitational acceleration.

    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
    """

    df = gravity_data

    jac = np.empty(
        (
            len(df[gravity_col]),
            # np.count_nonzero(~np.isnan(model.top.values))),
            model.top.size,
        ),
        dtype=np.float64,
    )

    # prisms_n_density = []

    # for x in range(len(model.easting.values)):
    #     for y in range(len(model.northing.values)):
    #         prism_info = (
    #             model.prism_layer.get_prism((x, y)),
    #             model.density.values[x, y]
    #         )
    #         if any([np.isnan(x).any() for x in prism_info]) is False:
    #             prisms_n_density.append(prism_info)

    # Build a generator for prisms (doesn't allocate memory, only returns at request)
    # about half of the cmp. time is here.
    prisms_n_density = (
        (model.prism_layer.get_prism((i, j)), model.density.values[i, j])
        for i in range(model.northing.size)
        for j in range(model.easting.size)
    )

    for col, (prism, density) in enumerate(prisms_n_density):
        # for prisms without any nan's (prisms with thicknesses < threshold)
        # if any([np.isnan(x).any() for x in (prism, density)]) is False:
        # Build a small prism ontop of existing prism (thickness equal to delta)
        bottom = prism[5] - delta / 2
        top = prism[5] + delta / 2
        delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)
        jac[:, col] = (
            hm.prism_gravity(  # other half of comp. time is here.
                coordinates=(df.x, df.y, df.z),
                prisms=delta_prism,
                density=density,
                field=field,
            )
            / delta
        )

    return jac


def solver(
    jacobian: np.array,
    residuals: np.array,
    solver_type: str = "least squares",
):
    """
    Calculate shift to add to prism's for each iteration of the inversion.

    Parameters
    ----------
    jacobian : np.array
        input jacobian matrix with a row per gravity observation, and a column per
        prisms.
    residuals : np.array
        array of gravity residuals
    solver_type : str, optional
        choose which solving method to use, by default "least squares"

    Returns
    -------
    np.array
        array of corrrection values to apply to each prism.
    """

    if solver_type == "least squares":
        # gives the amount that each column's Z1 needs to change by to have the smallest
        # misfit
        # finds the least-squares solution to jacobian and Grav_Misfit, assigns the
        # first value to Surface_correction
        step = lsqr(jacobian, residuals, show=False)[0]
    # elif solver_type == 'gauss newton':
    #     # doesn't currently work
    #     hessian = jacobian.T @ jacobian
    #     gradient = jacobian.T @ residuals
    #     step = np.linalg.solve(hessian, gradient)
    # elif solver_type == 'steepest descent':
    #     # doesn't currently work
    #     step = - jacobian.T @ residuals
    return step


def geo_inversion(
    active_layer: str,
    layers: dict,
    input_grav: pd.DataFrame,
    buffer_region: list,
    regional_method: str,
    grav_spacing: float,
    misfit_sq_tolerance: float = 0.00001,
    delta_misfit_squared_tolerance: float = 0.002,
    Max_Iterations: int = 3,
    deriv_type: str = "prisms",
    solver_type: str = "least squares",
    max_layer_change_per_iter: float = 100,
    save_results: bool = False,
    **kwargs,
):
    """
    Invert geometry of upper surface of prism layer based on gravity anomalies.

    Parameters
    ----------
    active_layer : str
        layer to invert.
    layers : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
            'len': int, number of prisms in the layer
            'prisms': xarray.DataSet, harmonica.prism_layer
    input_grav : pd.DataFrame
        input gravity data with anomaly columns
    buffer_region : list
        region including buffer zone, by default reads region from first grid layer
    regional_method : {'trend', 'filter', 'constraints', 'eq_sources'}
        choose a method to determine the regional gravity misfit.
    grav_spacing : float
        _description_
    misfit_sq_tolerance : float, optional
        _description_, by default 0.00001
    delta_misfit_squared_tolerance : float, optional
        _description_, by default 0.002
    Max_Iterations : int, optional
        terminate the inversion after this number of iterations, by default 3
    deriv_type : {'prisms', 'annulus'}, optional
        choose method for calculating vertical derivative of gravity, by default
        "prisms"
    solver_type : {'least squares', 'gauss newton', 'steepest descent'}, optional
        _choose method for solving for geometry correction, by default "least squares"
    max_layer_change_per_iter : float, optional
        maximum amount in meters each prism's surface can change by during each
        iteration, by default 100
    save_results : bool, optional
        choose whether to save results as a csv, by default False

    Other Parameters
    ----------------
    apply_constraints : bool,
        False
    constraints_grid : xr.DataArray,
        grid with values from 0-1 by which to multiple each iterations correction
        values by, defaults to None
    corrections : list,
        list of layers to include in partial Bouguer correction of Observed
        gravity data.
    input_grav_column : str,
        "Gobs"
    trend: int,
        trend order used from calculating regional misfit if
        regional_method = 'trend'.
    filter: str,
        input string for pygmt.grdfilter() for calculating regional misfit if
        regional_method = 'filter', ex. "g200e3" gives a 200km Gaussian filter.
    constraints: pd.DataFrame,
        Locations of constraint points to interpolate between for calculating
        regional misfit if regional_method = 'constraints'.
    fname_topo: str
        set csv filename, by default is 'topo_results'
    fname_gravity: str
        set csv filename, by default is 'gravity_results'

    Returns
    -------
    tuple
        iter_corrections: pd.DataFrame with corrections and updated geometry of the
            inversion layer for each iteration.
        gravity: pd.DataFrame with new columns of inversion results

    """

    if (
        kwargs.get("apply_constraints", False) is not False
        and kwargs.get("constraints_grid", None) is None
    ):
        raise ValueError(
            f"If apply_constraints = {kwargs.get('apply_constraints', False)}, ",
            "constraints_grid must be applied.",
        )

    include_forward_layers = pd.Series(
        [k for k, v in layers.items() if k not in kwargs.get("corrections", [])]
    )

    spacing = layers[active_layer]["spacing"]
    misfit_squared_updated = np.Inf  # positive infinity
    delta_misfit_squared = np.Inf  # positive infinity
    ind = include_forward_layers[include_forward_layers == active_layer].index[0]
    ITER = 0
    # while delta_misfit_squared (inf) is greater than 1 + least squares tolerance
    # (0.02)
    while delta_misfit_squared > 1 + delta_misfit_squared_tolerance:
        ITER += 1

        print(f"\n{'':#<60}##################################\niteration {ITER}")
        if ITER == 1:
            gravity = input_grav.copy()
        else:
            gravity["res"] = gravity[f"iter_{ITER-1}_final_misfit"]

        initial_RMS = round(np.sqrt((gravity.res**2).mean(skipna=True)), 2)
        print(f"initial RMS residual = {initial_RMS}mGal")

        # get prisms' coordinates from active layer and layer above
        prisms = layers[active_layer]["prisms"].to_dataframe().reset_index().dropna()
        prisms_above = (
            layers[include_forward_layers[ind - 1]]["prisms"]
            .to_dataframe()
            .reset_index()
            .dropna()
        )

        # prisms['index']=prisms.index
        # prisms_above['index']=prisms_above.index
        # prisms = prisms.dropna()
        # prisms_above = prisms_above.dropna()

        # calculate jacobian
        if deriv_type == "annulus":  # major issue with grav_column_der, way too slow
            jac = jacobian_annular(
                gravity,
                kwargs.get("input_grav_column", "Gobs"),
                prisms,
                spacing,
            )
        elif deriv_type == "prisms":
            jac = jacobian_prism(
                gravity,
                kwargs.get("input_grav_column", "Gobs"),
                layers[active_layer]["prisms"],
                1,
                "g_z",
            )
        else:
            print("not valid derivative type")

        # Calculate correction for each prism's surface
        # returns a 1-d array of length: number of input prisms > thickness threshold
        Surface_correction = solver(jac, gravity.res.values, solver_type=solver_type)

        print(f"Mean layer correction: {Surface_correction.mean()}")

        # for i, j in enumerate(prisms):
        for i in range(0, len(prisms)):
            if Surface_correction[i] > max_layer_change_per_iter:
                Surface_correction[i] = max_layer_change_per_iter
            elif Surface_correction[i] < -max_layer_change_per_iter:
                Surface_correction[i] = -max_layer_change_per_iter

        # add corrections to active prisms layer
        prisms["correction"] = Surface_correction

        # add same corrections to layer above active layer
        prisms_above["correction"] = Surface_correction
        # prisms_above = pd.merge(
        #     prisms_above, prisms[['correction']],
        #     how='left',
        #     left_index=True,
        #     right_index=True,
        #     )

        print(
            f"RMS layer correction {round(np.sqrt((Surface_correction**2).mean()),2)}m"
        )
        # apply above surface corrections
        if kwargs.get("apply_constraints", False) is True:
            prisms["constraints"] = (
                kwargs.get("constraints_grid", None).to_dataframe().reset_index().z
            )
            prisms_above["constraints"] = (
                kwargs.get("constraints_grid", None).to_dataframe().reset_index().z
            )
            prisms["correction"] = prisms.constraints * prisms.correction
            prisms_above["correction"] = (
                prisms_above.constraints * prisms_above.correction
            )
        else:
            print("constraints not applied")

        if ITER == 1:
            iter_corrections = prisms.rename(
                columns={"easting": "x", "northing": "y"}
            ).copy()
        iter_corrections[f"iter_{ITER}_initial_top"] = prisms.top.copy()

        prisms.top += prisms.correction
        prisms_above.bottom += prisms_above.correction
        iter_corrections[f"iter_{ITER}_final_top"] = prisms.top.copy()

        # apply the z correction to the active prism layer and the above layer
        prisms_grid = pygmt.xyz2grd(
            prisms[["easting", "northing", "top"]],
            region=buffer_region,
            registration="p",
            spacing=spacing,
        )
        prisms_above_grid = pygmt.xyz2grd(
            prisms_above[["easting", "northing", "bottom"]],
            region=buffer_region,
            registration="p",
            spacing=spacing,
        )

        layers[active_layer]["prisms"].prism_layer.update_top_bottom(
            surface=prisms_grid, reference=layers[active_layer]["prisms"].bottom
        )
        layers[include_forward_layers[ind - 1]]["prisms"].prism_layer.update_top_bottom(
            surface=layers[include_forward_layers[ind - 1]]["prisms"].top,
            reference=prisms_above_grid,
        )

        gravity[f"iter_{ITER}_initial_misfit"] = gravity.res

        iter_corrections[f"iter_{ITER}_correction"] = prisms.correction.copy()

        print("calculating updated forward gravity")
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] = layers[active_layer][
            "prisms"
        ].prism_layer.gravity(
            coordinates=(gravity.x, gravity.y, gravity.z), field="g_z"
        )
        gravity[f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"] = layers[
            include_forward_layers[ind - 1]
        ]["prisms"].prism_layer.gravity(
            coordinates=(gravity.x, gravity.y, gravity.z), field="g_z"
        )
        # center on 0
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] -= gravity[
            f"iter_{ITER}_{active_layer}_forward_grav"
        ].mean()
        gravity[f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"] -= gravity[
            f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"
        ].mean()

        # add updated layers' column names to list
        updated_layers = [active_layer, include_forward_layers[ind - 1]]
        updated_layers_list = [f"iter_{ITER}_{i}_forward_grav" for i in updated_layers]
        # add unchanged layers (excluding corrections layers) column names to list
        unchanged_layers = include_forward_layers[
            ~include_forward_layers.str.contains(
                f"{include_forward_layers[ind-1]}|{active_layer}"
            )
        ]
        unchanged_layers_list = [f"{i}_forward_grav" for i in unchanged_layers]
        # combined list of column names
        updated_forward = updated_layers_list + unchanged_layers_list
        # recalculate forward gravity with dataframe column names
        gravity[f"iter_{ITER}_forward_total"] = gravity[updated_forward].sum(
            axis=1, skipna=True
        )
        # center on 0
        gravity[f"iter_{ITER}_forward_total"] -= gravity[
            f"iter_{ITER}_forward_total"
        ].mean()

        print("updating the misfits")
        gravity[f"iter_{ITER}_final_misfit"] = anomalies(
            layers=layers,
            input_grav=gravity,
            grav_spacing=grav_spacing,
            regional_method=regional_method,
            input_forward_column=f"iter_{ITER}_forward_total",
            input_grav_column="Gobs",
            **kwargs,
        ).res

        # for first iteration, divide infinity by mean square of gravity residuals,
        # inversion will stop once this gets to delta_misfit_squared_tolerance (0.02)
        misfit_sq = (gravity[f"iter_{ITER}_final_misfit"] ** 2).mean(skipna=True).item()
        delta_misfit_squared = misfit_squared_updated / misfit_sq
        misfit_squared_updated = misfit_sq  # updated

        layers[active_layer]["inv_grid"] = (
            prisms.rename(columns={"easting": "x", "northing": "y"})
            .set_index(["y", "x"])
            .to_xarray()
            .top
        )

        # active_layer_total_difference = (
        #     layers[active_layer]["inv_grid"] - layers[active_layer]["grid"]
        # )

        if ITER == Max_Iterations:
            print(
                f"Inversion terminated after {ITER} iterations with least-squares norm",
                f"= {int(misfit_sq)} because maximum number of iterations ",
                f"({Max_Iterations}) reached",
            )
            break
        # if misfit_sq < misfit_sq_tolerance:
        #     print(f"Inversion terminated after {ITER} iterations with least-squares
        #     norm={int(misfit_sq)} because least-squares norm < {misfit_sq_tolerance}")
        #     break

    # end of inversion iteration WHILE loop
    if delta_misfit_squared < 1 + delta_misfit_squared_tolerance:
        print("terminated - no significant variation in least-squares norm ")

    if save_results is True:
        iter_corrections.to_csv(
            f"results/{kwargs.get('fname_topo', 'topo_results')}.csv", index=False
        )
        gravity.to_csv(
            f"results/{kwargs.get('fname_gravity', 'gravity_results')}.csv", index=False
        )

    return iter_corrections, gravity
