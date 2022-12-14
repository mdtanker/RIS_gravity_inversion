import copy
import functools
import time
import warnings

import harmonica as hm
import numba
import numpy as np
import pandas as pd
import pygmt
import scipy as sp
import verde as vd
import xarray as xr
from antarctic_plots import fetch, maps, profile, utils

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


# function to give RMSE of data
def RMSE(data):
    return np.sqrt(np.nanmean(data**2).item())


def inverted_prisms_to_zero(surface, reference):
    def neg_thick():
        return """Warning: portion of upper grid is below lower grid. Setting prism
        tops equal to bottoms for these prisms (thickness of 0).
        """

    thickness = surface - reference

    # check for negative thickness values
    if (thickness).values.min() < 0:
        warnings.warn(neg_thick())
        reference = np.minimum(surface, reference)
        thickness = surface - reference

    return surface, reference, thickness


def grids_to_prism_layers(
    layers: dict,
    thickness_threshold: float = 1,
    registration="g",
    lowest_bottom=None,
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
    # for k, v in layers.items():
    # density=v["grid"].copy()
    # density.values[:] = v["rho"]
    # v["grid"] = ([v["grid"], density])
    # v["grid"]["density"] = v["grid"].copy()
    # v["grid"].density.values[:] = v["rho"]
    # print(v["grid"])

    # list of layers, bottom up
    # reversed_layers_list = layers_list.iloc[::-1]
    reversed_layers_list = pd.Series([k for k, v in layers.items()]).iloc[::-1]

    # create prisms layers from input grids
    for i, j in enumerate(reversed_layers_list):
        density = np.ones_like(layers[j]["grid"].values) * layers[j]["rho"]
        # bottom-most prism layer
        if i == 0:
            # tops of prisms are from current grid
            surface = layers[j]["grid"]
            # base of prisms
            if lowest_bottom is not None:
                reference = lowest_bottom
            else:
                reference = np.nanmin(layers[j]["grid"].values)

            surface, reference, thickness = inverted_prisms_to_zero(surface, reference)

            layers[j]["prisms"] = hm.prism_layer(
                coordinates=(layers[j]["grid"].x.values, layers[j]["grid"].y.values),
                surface=surface.astype(np.float64),
                reference=reference.astype(np.float64),
                properties={
                    "density": density.astype(np.float64),
                    "thickness": thickness.astype(np.float64),
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
                    registration=registration,
                    spacing=layers[j]["spacing"],
                )
                surface = layers[j]["grid"]
                reference = tmp_grd

                surface, reference, thickness = inverted_prisms_to_zero(
                    surface, reference
                )

                layers[j]["prisms"] = hm.prism_layer(
                    coordinates=(
                        layers[j]["grid"].x.values,
                        layers[j]["grid"].y.values,
                    ),
                    surface=surface.astype(np.float64),
                    reference=reference.astype(np.float64),
                    properties={
                        "density": density.astype(np.float64),
                        "thickness": thickness.astype(np.float64),
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

                surface, reference, thickness = inverted_prisms_to_zero(
                    surface, reference
                )

                layers[j]["prisms"] = hm.prism_layer(
                    coordinates=(
                        layers[j]["grid"].x.values,
                        layers[j]["grid"].y.values,
                    ),
                    surface=surface.astype(np.float64),
                    reference=reference.astype(np.float64),
                    properties={
                        "density": density.astype(np.float64),
                        "thickness": thickness.astype(np.float64),
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
    registration="g",
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
    eq_damping: float
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
        anomalies["grav_corrected"] = anomalies[input_grav_column]

    # get obs-forward misfit
    anomalies["misfit"] = anomalies.grav_corrected - anomalies[input_forward_column]

    # grid the misfits, used in trend, filter, and constraints, not in Eq. Sources
    misfit = pygmt.xyz2grd(
        data=anomalies[["x", "y", "misfit"]],
        region=inversion_region,
        spacing=grav_spacing,
        registration=registration,
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

    # Filter method
    elif regional_method == "filter":
        # filter the observed-forward misfit with the provided filter in meters
        regional_misfit = pygmt.grdfilter(
            misfit,
            filter=filter,
            distance="0",
            registration=registration,
        )
        # sample the results and merge into the anomalies dataframe
        tmp_regrid = pygmt.grdtrack(
            points=anomalies[["x", "y"]],
            grid=regional_misfit,
            newcolname="reg",
            verbose="q",
        )
        anomalies = anomalies.merge(tmp_regrid, on=["x", "y"], how="left")

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
            registration=registration,
        )

        # grid the entire region misfit based just on the misfit at the constraints
        regional_misfit = pygmt.surface(
            data=blocked,
            region=inversion_region,
            spacing=grav_spacing,
            registration=registration,
            T=kwargs.get("tension_factor", 0.25),
            verbose="q",
        )

        # sample the resulting grid and add to anomalies dataframe
        anomalies = profile.sample_grids(
            df=anomalies,
            grid=regional_misfit,
            name="reg",
        )

    # Equivalent sources method
    elif regional_method == "eq_sources":
        # create set of deep sources
        equivalent_sources = hm.EquivalentSources(
            depth=eq_sources,
            damping=kwargs.get(
                "eq_damping", None
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

    # rmse = RMSE(anomalies.res)
    # print(f"Misfit RMSE: {round(rmse, 2)} mGal")

    return anomalies


@numba.njit(parallel=True)
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


@numba.njit(parallel=True)
def _jacobian_annular_numba(
    grav_x,
    grav_y,
    grav_z,
    prism_easting,
    prism_northing,
    prism_top,
    prism_bottom,
    prism_density,
    spacing: float,
):
    """
    Takes arrays from `jacobian_annular`, feeds them into `grav_column_der`, and returns
    the jacobian.

    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
    """

    jac = np.empty(
        (len(grav_x), len(prism_easting)),
        dtype=np.float64,
    )

    for i in numba.prange(len(grav_x)):
        jac[i, :] = grav_column_der(
            grav_y[i],
            grav_x[i],
            grav_z[i],
            prism_northing,
            prism_easting,
            prism_top,
            prism_bottom,
            spacing,
            prism_density / 1000,  # density
        )
    return jac


def jacobian_annular(
    coordinates: pd.DataFrame,
    model: xr.Dataset,
    spacing: float,
):
    """
    Function to calculate the Jacobian matrix using the annular cylinder approximation
    The resulting Jacobian is a matrix (numpy array) with a row per gravity observation
    and a column per prism. This approximates the prisms as an annulus, and calculates
    it's vertical gravity derivative.

    Parameters
    ----------
    coordinates : pd.DataFrame
        dataframe containing gravity data coordinates
    model : xr.Dataset
        harmonica.prism_layer, with coordinates:
        easting, northing, top, and bottom, and variables: 'Density'.
    spacing : float
        spacing of gravity data

    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
    """

    # convert prism model to dataframe
    prisms = model.to_dataframe().reset_index().dropna()

    # convert dataframes to numpy arrays
    coordinates_array = coordinates.to_numpy()
    prisms_array = prisms.to_numpy()

    # get various arrays based on gravity column names
    grav_x = coordinates_array[:, coordinates.columns.get_loc("x")]
    grav_y = coordinates_array[:, coordinates.columns.get_loc("y")]
    grav_z = coordinates_array[:, coordinates.columns.get_loc("z")]

    assert len(grav_x) == len(grav_y) == len(grav_z)

    # get various arrays based on prisms column names
    prism_easting = prisms_array[:, prisms.columns.get_loc("easting")]
    prism_northing = prisms_array[:, prisms.columns.get_loc("northing")]
    prism_top = prisms_array[:, prisms.columns.get_loc("top")]
    prism_bottom = prisms_array[:, prisms.columns.get_loc("bottom")]
    prism_density = prisms_array[:, prisms.columns.get_loc("density")]

    # feed above arrays into the jacobian function
    jac = _jacobian_annular_numba(
        grav_x,
        grav_y,
        grav_z,
        prism_easting,
        prism_northing,
        prism_top,
        prism_bottom,
        prism_density,
        spacing,
    )

    return jac


def _jacobian_prism_numba(
    grav_x,
    grav_y,
    grav_z,
    model,
    delta: float,
    field: str,
):
    """
    Takes arrays from `jacobian_prisms` and calculates the jacobian.

    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
    """

    jac = np.empty(
        (
            len(grav_x),
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
    #         prisms_n_density.append(prism_info)
    # if any([np.isnan(x).any() for x in prism_info]) is False:
    # prisms_n_density.append(prism_info)

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
        bottom = prism[5]  # - delta / 2
        top = prism[5] + delta  # / 2
        delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)

        jac[:, col] = (
            hm.prism_gravity(
                coordinates=(grav_x, grav_y, grav_z),
                prisms=delta_prism,
                density=density,
                field=field,
                parallel=True,
            )
            / delta
        )

    return jac


def jacobian_prism(
    coordinates: pd.DataFrame,
    model: xr.Dataset,
    delta: float,
    field: str,
):
    """
    Function to calculate the Jacobian matrix with the vertical gravity derivative
    as a numerical approximation with small prisms

    Parameters
    ----------
    coordinates : pd.DataFrame
        dataframe containing gravity observation coordinates
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

    # convert dataframes to numpy arrays
    coordinates_array = coordinates.to_numpy()

    # get various arrays based on gravity column names
    grav_x = coordinates_array[:, coordinates.columns.get_loc("x")]
    grav_y = coordinates_array[:, coordinates.columns.get_loc("y")]
    grav_z = coordinates_array[:, coordinates.columns.get_loc("z")]

    assert len(grav_x) == len(grav_y) == len(grav_z)

    # feed above arrays into the jacobian function
    jac = _jacobian_prism_numba(
        grav_x,
        grav_y,
        grav_z,
        model,
        delta,
        field,
    )

    return jac


def solver(
    jacobian: np.array,
    residuals: np.array,
    weights: np.array = None,
    damping: float = None,
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
    weights : np.array
        array of weights to assign to data, typically 1/(uncertainty**2)
    solver_damping : float
        positive damping (Tikhonov 0th order) regularization
    solver_type : {'least squares', 'gauss newton', 'steepest descent'} optional
        choose which solving method to use, by default "least squares"

    Returns
    -------
    np.array
        array of corrrection values to apply to each prism.
    """

    if solver_type == "scipy least squares":
        # gives the amount that each column's Z1 needs to change by to have the smallest
        # misfit
        # finds the least-squares solution to jacobian and the gravity residual, assigns
        # the first value to step
        if damping is None:
            damping = 0
        step = sp.sparse.linalg.lsqr(
            jacobian,
            residuals,
            show=False,
            damp=damping,
        )[0]
    elif solver_type == "scipy conjugate":
        step = sp.sparse.linalg.cg(
            jacobian,
            residuals,
        )[0]
    elif solver_type == "numpy least squares":
        step = np.linalg.lstsq(
            jacobian,
            residuals,
        )[0]
    elif solver_type == "verde least squares":
        step = vd.base.least_squares(
            jacobian,
            residuals,
            weights=None,
            damping=damping,
            copy_jacobian=False,
        )
    elif solver_type == "steepest descent":
        residuals = residuals
        step = jacobian.T @ residuals
    elif solver_type == "gauss newton":
        pass
        # from https://nbviewer.org/github/compgeolab/2020-aachen-inverse-problems/blob/main/gravity-inversion.ipynb # noqa
        # fdmatrix = finite_difference_matrix(jacobian[1].size)
        # hessian = jacobian.T @ jacobian  # + damping * fdmatrix.T @ fdmatrix
        # gradient = jacobian.T @ residuals  # - damping * fdmatrix.T @ fdmatrix @ tops
        # step = np.linalg.solve(hessian, gradient)
        # from https://github.com/peterH105/Gradient_Inversion/blob/071f3473f4655f88d3ee988255360781760b5055/code_SAM/grad_inv_functions_synthetic.py#L39 # noqa
        # gradient = jacobian_shift
        # # finite_diff matrix: same shape as jacobian
        # fdmatrix =
        # rhs = gradient.T.dot(residuals)-fdmatrix.T.dot(fdmatrix).dot(tops)
        # lhs = gradient.T.dot(gradient)+fdmatrix.T.dot(fdmatrix)
        # step = np.linalg.solve(lhs,rhs)
    return step


def finite_difference_matrix(nparams):
    """
    Create the finite difference matrix for regularization.
    """
    fdmatrix = np.zeros((nparams - 1, nparams))
    for i in range(fdmatrix.shape[0]):
        fdmatrix[i, i] = -1
        fdmatrix[i, i + 1] = 1
    return fdmatrix


def geo_inversion(
    active_layer: str,
    layers_dict: dict,
    input_grav: pd.DataFrame,
    buffer_region: list,
    regional_method: str,
    grav_spacing: float,
    l2_norm_tolerance: float = 1,
    delta_l2_norm_tolerance: float = 1.001,
    max_iterations: int = 3,
    deriv_type: str = "prisms",
    solver_type: str = "least squares",
    max_layer_change_per_iter: float = None,
    save_results: bool = False,
    registration="g",
    **kwargs,
):
    """
    Invert geometry of upper surface of prism layer based on gravity anomalies.

    Parameters
    ----------
    active_layer : str
        layer to invert.
    layers_dict : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
            'len': int, number of prisms in the layer
            'prisms': xarray.DataSet, harmonica.prism_layer
    input_grav : pd.DataFrame
        input gravity data, if contains anomaly columns will use them, if not, will
        calculate them
    buffer_region : list
        region including buffer zone, by default reads region from first grid layer
    regional_method : {'trend', 'filter', 'constraints', 'eq_sources'}
        choose a method to determine the regional gravity misfit.
    grav_spacing : float
        _description_
    l2_norm_tolerance : float, optional
        end inversion if L2-norm is below this value, by default 1
    delta_l2_norm_tolerance : float, optional
        end inversion if L2-norm-updated/L2-norm-previous is above this value, by
        default 1.001
    max_iterations : int, optional
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
    solver_damping: float,
        damping parameter for least squares solvers
    filter: str,
        input string for pygmt.grdfilter() for calculating regional misfit if
        regional_method = 'filter', ex. "g200e3" gives a 200km Gaussian filter.
    trend: int,
        trend order used from calculating regional misfit if
        regional_method = 'trend'.
    fill_method: {'pygmt', 'rioxarray'},
        Choose method to fill nans, by default is 'pygmt'
    constraints: pd.DataFrame,
        Locations of constraint points to interpolate between for calculating
        regional misfit if regional_method = 'constraints'.
    tension_factor: float,
    eq_damping: float
        smoothness to impose on estimated coefficients
    block_size: float
        block reduce the data to speed up
    depth_type: str
        "relative" or "constant" for depth for eq_sources
    fname_topo: str
        set csv filename, by default is 'topo_results'
    fname_gravity: str
        set csv filename, by default is 'gravity_results'

    Returns
    -------
    list
        iter_corrections: pd.DataFrame with corrections and updated geometry of the
            inversion layer for each iteration.
        gravity: pd.DataFrame with new columns of inversion results
        layers_update: dict with updated layer geometries

    """

    if (
        kwargs.get("apply_constraints", False) is not False
        and kwargs.get("constraints_grid", None) is None
    ):
        raise ValueError(
            f"If apply_constraints = {kwargs.get('apply_constraints', False)}, ",
            "constraints_grid must be applied.",
        )

    if set(["misfit", "res", "reg"]).issubset(input_grav.columns):
        gravity = input_grav.copy()
    else:
        gravity = anomalies(
            layers=layers_dict,
            input_grav=input_grav,
            grav_spacing=grav_spacing,
            regional_method=regional_method,
            registration=registration,
            **kwargs,
        )

    layers_update = copy.deepcopy(layers_dict)

    include_forward_layers = pd.Series(
        [k for k, v in layers_update.items() if k not in kwargs.get("corrections", [])]
    )

    spacing = layers_update[active_layer]["spacing"]
    delta_l2_norm = np.Inf  # positive infinity
    ind = include_forward_layers[include_forward_layers == active_layer].index[0]

    for ITER, _ in enumerate(range(max_iterations), start=1):
        print(f"\n{'':#<60}##################################\niteration {ITER}")
        if ITER == 1:
            pass
        else:
            gravity["res"] = gravity[f"iter_{ITER-1}_final_misfit"]

        initial_RMSE = RMSE(gravity.res)
        l2_norm = np.sqrt(initial_RMSE)
        print(f"initial misfit RMSE = {round(initial_RMSE, 2)} mGal")
        print(f"initial L2-norm : {round(l2_norm, 2)}")
        print(f"initial delta L2-norm : {round(delta_l2_norm, 2)}")

        # get prisms' coordinates from active layer and layer above
        prisms = (
            layers_update[active_layer]["prisms"]
            .to_dataframe()
            .reset_index()
            .dropna()
            .astype(float)
        )
        prisms_above = (
            layers_update[include_forward_layers[ind - 1]]["prisms"]
            .to_dataframe()
            .reset_index()
            .dropna()
            .astype(float)
        )

        # sample top of upper layer
        prisms = profile.sample_grids(
            df=prisms,
            grid=layers_update[include_forward_layers[ind - 1]]["prisms"].top,
            name="top_of_above",
            coord_names=("easting", "northing"),
        )

        # prisms['index']=prisms.index
        # prisms_above['index']=prisms_above.index
        # prisms = prisms.dropna()
        # prisms_above = prisms_above.dropna()

        # calculate jacobian
        if deriv_type == "annulus":  # major issue with grav_column_der, way too slow
            jac = jacobian_annular(
                gravity,
                layers_update[active_layer]["prisms"],
                spacing,
            )
        elif deriv_type == "prisms":
            jac = jacobian_prism(
                coordinates=gravity,
                model=layers_update[active_layer]["prisms"],
                delta=kwargs.get("delta", 1),
                field="g_z",
            )
        else:
            print("not valid derivative type")

        # sample constraints grid at gravity points
        if kwargs.get("apply_weights", False) is True:
            weights_df = profile.sample_grids(
                df=gravity,
                grid=kwargs.get("constraints_grid", None),
                name="weights",
            )
            weights = weights_df.weights

        else:
            weights = None

        # Calculate correction for each prism's surface
        # returns a 1-d array of length: number of input prisms > thickness threshold
        Surface_correction = solver(
            jacobian=jac,
            residuals=gravity.res.values,
            weights=weights,
            damping=kwargs.get("solver_damping", None),
            solver_type=solver_type,
        )

        print(
            f"Layer correction mean: {int(Surface_correction.mean())}",
            f"m, RMSE:{int(RMSE(Surface_correction))} m",
        )

        if max_layer_change_per_iter is not None:
            # for i, j in enumerate(prisms):
            for i in range(0, len(prisms)):
                if Surface_correction[i] > max_layer_change_per_iter:
                    Surface_correction[i] = max_layer_change_per_iter
                elif Surface_correction[i] < -max_layer_change_per_iter:
                    Surface_correction[i] = -max_layer_change_per_iter
            print(
                "Layer correction (after clipped) mean:",
                f"{int(Surface_correction.mean())}m,",
                f"RMSE:{int(RMSE(Surface_correction))} m",
            )

        # don't let correction bring active layer above top of above layer
        # i.e., don't let bed move above ice base/water surface
        prisms["max_allowed_change"] = prisms.top_of_above - prisms.top
        for i in range(0, len(prisms)):
            if Surface_correction[i] > prisms.max_allowed_change[i]:
                Surface_correction[i] = prisms.max_allowed_change[i]

        # add corrections to active prisms layer
        prisms["correction"] = Surface_correction

        # add same corrections to layer above active layer
        prisms_above["correction"] = Surface_correction

        # vizualize max allowed change
        # max_allowed_change = prisms.set_index(
        # ["northing", "easting"]).to_xarray().max_allowed_change
        # change = prisms.set_index(["northing", "easting"]).to_xarray().correction
        # (max_allowed_change - change).plot(vmax=50)
        # import matplotlib.pyplot as plt
        # plt.show()

        # prisms_above = pd.merge(
        #     prisms_above, prisms[['correction']],
        #     how='left',
        #     left_index=True,
        #     right_index=True,
        #     )

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

        # grid updated prism surfaces
        updated_top = prisms.set_index(["northing", "easting"]).to_xarray().top
        updated_bottom = (
            prisms_above.set_index(["northing", "easting"]).to_xarray().bottom
        )

        # apply the z correction to the active prism layer and the above layer
        # this is resulting in some minor issues where bottom of above layer doesn't
        # exactly equal top of below layer
        # (updated_top - updated_bottom).plot()
        # import matplotlib.pyplot as plt
        # plt.show()
        layers_update[active_layer]["prisms"].prism_layer.update_top_bottom(
            surface=updated_top, reference=layers_update[active_layer]["prisms"].bottom
        )
        layers_update[include_forward_layers[ind - 1]][
            "prisms"
        ].prism_layer.update_top_bottom(
            surface=layers_update[include_forward_layers[ind - 1]]["prisms"].top,
            reference=updated_bottom,
        )
        # same as above but without using df's
        # surf_corr = prisms.set_index(["northing", "easting"]).to_xarray().correction
        # layers_update[active_layer]["prisms"].prism_layer.update_top_bottom(
        #         surface=layers_update[active_layer]["prisms"].top+surf_corr,
        #         reference=layers_update[active_layer]["prisms"].bottom
        #     )
        # layers_update[
        # include_forward_layers[ind - 1]]["prisms"].prism_layer.update_top_bottom(
        #     surface=layers_update[include_forward_layers[ind - 1]]["prisms"].top,
        #     reference=layers_update[
        # include_forward_layers[ind - 1]]["prisms"].bottom+surf_corr,
        # )

        # add results to df's
        gravity[f"iter_{ITER}_initial_misfit"] = gravity.res
        iter_corrections[f"iter_{ITER}_correction"] = prisms.correction.copy()
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] = layers_update[
            active_layer
        ]["prisms"].prism_layer.gravity(
            coordinates=(gravity.x, gravity.y, gravity.z), field="g_z"
        )
        gravity[
            f"iter_{ITER}_{include_forward_layers[ind-1]}_forward_grav"
        ] = layers_update[include_forward_layers[ind - 1]][
            "prisms"
        ].prism_layer.gravity(
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
            layers=layers_update,
            input_grav=gravity,
            grav_spacing=grav_spacing,
            regional_method=regional_method,
            input_forward_column=f"iter_{ITER}_forward_total",
            input_grav_column="Gobs",
            **kwargs,
        ).res

        # calculate updated RMSE of the misfit
        updated_RMSE = RMSE(gravity[f"iter_{ITER}_final_misfit"])
        print(f"\nupdated misfit RMSE: {round(updated_RMSE, 2)}")

        # square-root of RMSE is the l-2 norm
        updated_l2_norm = np.sqrt(updated_RMSE)

        print(
            f"updated L2-norm: {round(updated_l2_norm, 2)}, ",
            f"tolerance: {l2_norm_tolerance}",
        )

        # inversion will stop once this gets to delta_l2_norm_tolerance (0.02)
        updated_delta_l2_norm = l2_norm / updated_l2_norm

        print(
            f"updated delta L2-norm : {round(updated_delta_l2_norm, 2)}, ",
            f"tolerance: {delta_l2_norm_tolerance}",
        )

        # update the l2_norm
        l2_norm = updated_l2_norm

        # updated the delta l2_norm
        delta_l2_norm = updated_delta_l2_norm

        # save the update topo in the dictionary
        layers_update[active_layer]["inv_grid"] = (
            prisms.rename(columns={"easting": "x", "northing": "y"})
            .set_index(["y", "x"])
            .to_xarray()
            .top
        )

        if delta_l2_norm < delta_l2_norm_tolerance:
            print(
                f"\nInversion terminated after {ITER} iterations because there was",
                "no significant variation in the L2-norm",
            )
            break

        if l2_norm < l2_norm_tolerance:
            print(
                f"\nInversion terminated after {ITER} iterations with L2-norm =",
                f"{round(l2_norm, 2)} because L2-norm < {l2_norm_tolerance}",
            )
            break

        if ITER == max_iterations:
            print(
                f"\nInversion terminated after {ITER} iterations with L2-norm =",
                f"{round(l2_norm, 2)} because maximum number of iterations ",
                f"({max_iterations}) reached",
            )
            break

    if save_results is True:
        iter_corrections.to_csv(
            f"results/{kwargs.get('fname_topo', 'topo_results')}.csv", index=False
        )
        gravity.to_csv(
            f"results/{kwargs.get('fname_gravity', 'gravity_results')}.csv", index=False
        )

    return iter_corrections, gravity, layers_update


def geo_inversion_dens_contrast(
    active_layer: str,
    layers_dict: dict,
    input_grav: pd.DataFrame,
    buffer_region: list,
    regional_method: str,
    grav_spacing: float,
    l2_norm_tolerance: float = 1,
    delta_l2_norm_tolerance: float = 1.001,
    max_iterations: int = 3,
    deriv_type: str = "prisms",
    solver_type: str = "least squares",
    max_layer_change_per_iter: float = None,
    save_results: bool = False,
    registration="g",
    **kwargs,
):
    """
    Invert geometry of upper surface of prism layer based on gravity anomalies.

    Parameters
    ----------
    active_layer : str
        layer to invert.
    layers_dict : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
            'len': int, number of prisms in the layer
            'prisms': xarray.DataSet, harmonica.prism_layer
    input_grav : pd.DataFrame
        input gravity data, if contains anomaly columns will use them, if not, will
        calculate them
    buffer_region : list
        region including buffer zone, by default reads region from first grid layer
    regional_method : {'trend', 'filter', 'constraints', 'eq_sources'}
        choose a method to determine the regional gravity misfit.
    grav_spacing : float
        _description_
    l2_norm_tolerance : float, optional
        end inversion if L2-norm is below this value, by default 1
    delta_l2_norm_tolerance : float, optional
        end inversion if L2-norm-updated/L2-norm-previous is above this value, by
        default 1.001
    max_iterations : int, optional
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
    solver_damping: float,
        damping parameter for least squares solvers
    filter: str,
        input string for pygmt.grdfilter() for calculating regional misfit if
        regional_method = 'filter', ex. "g200e3" gives a 200km Gaussian filter.
    trend: int,
        trend order used from calculating regional misfit if
        regional_method = 'trend'.
    fill_method: {'pygmt', 'rioxarray'},
        Choose method to fill nans, by default is 'pygmt'
    constraints: pd.DataFrame,
        Locations of constraint points to interpolate between for calculating
        regional misfit if regional_method = 'constraints'.
    tension_factor: float,
    eq_damping: float
        smoothness to impose on estimated coefficients
    block_size: float
        block reduce the data to speed up
    depth_type: str
        "relative" or "constant" for depth for eq_sources
    fname_topo: str
        set csv filename, by default is 'topo_results'
    fname_gravity: str
        set csv filename, by default is 'gravity_results'

    Returns
    -------
    list
        iter_corrections: pd.DataFrame with corrections and updated geometry of the
            inversion layer for each iteration.
        gravity: pd.DataFrame with new columns of inversion results
        layers_update: dict with updated layer geometries

    """

    if (
        kwargs.get("apply_constraints", False) is not False
        and kwargs.get("constraints_grid", None) is None
    ):
        raise ValueError(
            f"If apply_constraints = {kwargs.get('apply_constraints', False)}, ",
            "constraints_grid must be applied.",
        )

    if set(["misfit", "res", "reg"]).issubset(input_grav.columns):
        gravity = input_grav.copy()
    else:
        gravity = anomalies(
            layers=layers_dict,
            input_grav=input_grav,
            grav_spacing=grav_spacing,
            regional_method=regional_method,
            registration=registration,
            **kwargs,
        )

    layers_update = copy.deepcopy(layers_dict)

    spacing = layers_update[active_layer]["spacing"]
    delta_l2_norm = np.Inf  # positive infinity

    for ITER, _ in enumerate(range(max_iterations), start=1):
        print(f"\n{'':#<60}##################################\niteration {ITER}")
        if ITER == 1:
            pass
        else:
            gravity["res"] = gravity[f"iter_{ITER-1}_final_misfit"]

        initial_RMSE = RMSE(gravity.res)
        l2_norm = np.sqrt(initial_RMSE)
        print(f"initial misfit RMSE = {round(initial_RMSE, 2)} mGal")
        print(f"initial L2-norm : {round(l2_norm, 2)}")
        print(f"initial delta L2-norm : {round(delta_l2_norm, 2)}")

        # get prisms' coordinates from active layer and layer above
        prisms = (
            layers_update[active_layer]["prisms"]
            .to_dataframe()
            .reset_index()
            .dropna()
            .astype(float)
        )

        # load and resample upper layer surface
        above_layer = kwargs.get("above_layer", None)
        if above_layer is not None:
            tmp_grid = xr.open_zarr(above_layer)
            tmp_grid = tmp_grid[list(tmp_grid.keys())[0]].squeeze()
            # resample to match active layer
            above_layer = fetch.resample_grid(
                tmp_grid,
                spacing=layers_update[active_layer]["spacing"],
                region=buffer_region,
                registration=registration,
                verbose="q",
            )

        # sample top of upper layer
        prisms = profile.sample_grids(
            df=prisms,
            grid=above_layer,
            name="top_of_above",
            coord_names=("easting", "northing"),
        )

        # calculate jacobian
        if deriv_type == "annulus":  # major issue with grav_column_der, way too slow
            jac = jacobian_annular(
                gravity,
                layers_update[active_layer]["prisms"],
                spacing,
            )
        elif deriv_type == "prisms":
            jac = jacobian_prism(
                coordinates=gravity,
                model=layers_update[active_layer]["prisms"],
                delta=kwargs.get("delta", 1),
                field="g_z",
            )
        else:
            print("not valid derivative type")

        # sample constraints grid at gravity points
        if kwargs.get("apply_weights", False) is True:
            weights_df = profile.sample_grids(
                df=gravity,
                grid=kwargs.get("constraints_grid", None),
                name="weights",
            )
            weights = weights_df.weights

        else:
            weights = None

        # Calculate correction for each prism's surface
        # returns a 1-d array of length: number of input prisms > thickness threshold
        Surface_correction = solver(
            jacobian=jac,
            residuals=gravity.res.values,
            weights=weights,
            damping=kwargs.get("solver_damping", None),
            solver_type=solver_type,
        )

        print(
            f"Layer correction mean: {int(Surface_correction.mean())}",
            f"m, RMSE:{int(RMSE(Surface_correction))} m",
        )

        if max_layer_change_per_iter is not None:
            # for i, j in enumerate(prisms):
            for i in range(0, len(prisms)):
                if Surface_correction[i] > max_layer_change_per_iter:
                    Surface_correction[i] = max_layer_change_per_iter
                elif Surface_correction[i] < -max_layer_change_per_iter:
                    Surface_correction[i] = -max_layer_change_per_iter
            print(
                "Layer correction (after clipped) mean:",
                f"{int(Surface_correction.mean())}",
                f"m, RMSE:{int(RMSE(Surface_correction))} m",
            )

        # don't let correction bring active layer above top of above layer
        # i.e., don't let bed move above ice base/water surface
        prisms["max_allowed_change"] = prisms.top_of_above - prisms.top
        for i in range(0, len(prisms)):
            if Surface_correction[i] > prisms.max_allowed_change[i]:
                Surface_correction[i] = prisms.max_allowed_change[i]

        # add corrections to active prisms layer
        prisms["correction"] = Surface_correction

        # apply above surface corrections
        if kwargs.get("apply_constraints", False) is True:
            prisms["constraints"] = (
                kwargs.get("constraints_grid", None).to_dataframe().reset_index().z
            )
            prisms["correction"] = prisms.constraints * prisms.correction
        else:
            print("constraints not applied")

        if ITER == 1:
            iter_corrections = prisms.rename(
                columns={"easting": "x", "northing": "y"}
            ).copy()
        iter_corrections[f"iter_{ITER}_initial_top"] = prisms.top.copy()

        prisms.top += prisms.correction

        iter_corrections[f"iter_{ITER}_final_top"] = prisms.top.copy()

        # grid updated prism surfaces
        updated_top = prisms.set_index(["northing", "easting"]).to_xarray().top

        layers_update[active_layer]["prisms"].prism_layer.update_top_bottom(
            surface=updated_top, reference=layers_update[active_layer]["prisms"].bottom
        )

        # add results to df's
        gravity[f"iter_{ITER}_initial_misfit"] = gravity.res
        iter_corrections[f"iter_{ITER}_correction"] = prisms.correction.copy()
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] = layers_update[
            active_layer
        ]["prisms"].prism_layer.gravity(
            coordinates=(gravity.x, gravity.y, gravity.z), field="g_z"
        )

        # center on 0
        gravity[f"iter_{ITER}_{active_layer}_forward_grav"] -= gravity[
            f"iter_{ITER}_{active_layer}_forward_grav"
        ].mean()

        # recalculate forward gravity with dataframe column names
        gravity[f"iter_{ITER}_forward_total"] = gravity[
            f"iter_{ITER}_{active_layer}_forward_grav"
        ]

        print("updating the misfits")
        gravity[f"iter_{ITER}_final_misfit"] = anomalies(
            layers=layers_update,
            input_grav=gravity,
            grav_spacing=grav_spacing,
            regional_method=regional_method,
            input_forward_column=f"iter_{ITER}_forward_total",
            input_grav_column="Gobs",
            **kwargs,
        ).res

        # calculate updated RMSE of the misfit
        updated_RMSE = RMSE(gravity[f"iter_{ITER}_final_misfit"])
        print(f"\nupdated misfit RMSE: {round(updated_RMSE, 2)}")

        # square-root of RMSE is the l-2 norm
        updated_l2_norm = np.sqrt(updated_RMSE)

        print(
            f"updated L2-norm: {round(updated_l2_norm, 2)}, ",
            f"tolerance: {l2_norm_tolerance}",
        )

        # inversion will stop once this gets to delta_l2_norm_tolerance (0.02)
        updated_delta_l2_norm = l2_norm / updated_l2_norm

        print(
            f"updated delta L2-norm : {round(updated_delta_l2_norm, 2)}, ",
            f"tolerance: {delta_l2_norm_tolerance}",
        )

        # update the l2_norm
        l2_norm = updated_l2_norm

        # updated the delta l2_norm
        delta_l2_norm = updated_delta_l2_norm

        # save the update topo in the dictionary
        layers_update[active_layer]["inv_grid"] = (
            prisms.rename(columns={"easting": "x", "northing": "y"})
            .set_index(["y", "x"])
            .to_xarray()
            .top
        )

        if delta_l2_norm < delta_l2_norm_tolerance:
            print(
                f"\nInversion terminated after {ITER} iterations because there was",
                "no significant variation in the L2-norm",
            )
            break

        if l2_norm < l2_norm_tolerance:
            print(
                f"\nInversion terminated after {ITER} iterations with L2-norm =",
                f"{round(l2_norm, 2)} because L2-norm < {l2_norm_tolerance}",
            )
            break

        if ITER == max_iterations:
            print(
                f"\nInversion terminated after {ITER} iterations with L2-norm =",
                f"{round(l2_norm, 2)} because maximum number of iterations ",
                f"({max_iterations}) reached",
            )
            break

    if save_results is True:
        iter_corrections.to_csv(
            f"results/{kwargs.get('fname_topo', 'topo_results')}.csv", index=False
        )
        gravity.to_csv(
            f"results/{kwargs.get('fname_gravity', 'gravity_results')}.csv", index=False
        )

    return iter_corrections, gravity, layers_update


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
    registration="g",
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

    initial_RMSE = RMSE(df_grav["inv_misfit"])
    print(f"initial RMSE = {round(initial_RMSE, 2)} mGal")
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
    Density_correction = sp.sparse.linalg.lsqr(
        MAT_DENS, df_grav.inv_misfit, show=False
    )[0]

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
    # Density_correction=sp.sparse.linalg.lsqr(
    #   MAT_DENS,df_grav.inv_misfit,show=False)[0]*1000

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
        registration=registration,
        region=buffer_reg,
        spacing=grav_spacing,
        projection=buffer_proj,
    )
    dens_update = pygmt.xyz2grd(
        x=prisms.easting,
        y=prisms.northing,
        z=prisms.updated_density,
        registration=registration,
        region=buffer_reg,
        spacing=layers[density_layer]["spacing"],
        projection=buffer_proj,
    )
    initial_misfit = pygmt.xyz2grd(
        df_grav[["x", "y", "inv_misfit"]],
        region=inv_reg,
        spacing=grav_spacing,
        registration=registration,
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

    final_RMSE = RMSE(df_grav.inv_misfit)
    print(f"RMSE after inversion = {round(final_RMSE, 2)} mGal")

    final_misfit = pygmt.xyz2grd(
        df_grav[["x", "y", "inv_misfit"]],
        region=buffer_reg,
        registration=registration,
        spacing=grav_spacing,
    )

    if plot is True:
        grid = initial_misfit
        fig = maps.plot_grd(
            grid=grid,
            cmap="polar+h0",
            cbar_label=f"initial misfit (mGal) [{round(initial_RMSE, 2)}]",
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
            cbar_label=f"final misfit (mGal) [{round(final_RMSE, 2)}]",
            origin_shift="xshift",
        )

        fig.show(width=1200)
