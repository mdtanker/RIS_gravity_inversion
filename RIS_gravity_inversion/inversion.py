import copy
import functools
import time
import warnings
from typing import TYPE_CHECKING, Union

import harmonica as hm
import numba
import numpy as np
import pandas as pd
import pygmt
import scipy as sp
import verde as vd
import xarray as xr
from antarctic_plots import fetch, maps, profile, utils
import RIS_gravity_inversion.plotting as plots
import RIS_gravity_inversion.utils as inv_utils

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
    return np.sqrt(np.nanmedian(data**2).item())
    # return np.sqrt(np.nanmean(data**2).item())


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


def misfit(
    input_grav: pd.DataFrame,
    grav_spacing: int,
    registration="g",
    plot=True,
    **kwargs,
):
    input_forward_column = kwargs.get("input_forward_column", "forward")
    input_grav_column = kwargs.get("input_grav_column", "grav")

    # if inversion region not supplied, extract from dataframe
    inversion_region = kwargs.get(
        "inversion_region", vd.get_region((input_grav.x, input_grav.y))
    )

    # if misfit already calculated, drop the column
    try:
        input_grav.drop(columns=["misfit"], inplace=True)
    except KeyError:
        pass

    # get obs-forward misfit
    input_grav["misfit"] = input_grav[input_grav_column] - input_grav[input_forward_column]

    if plot is True:
        plots.misfit_plotting(
            input_grav,
            grav_spacing=grav_spacing,
            **kwargs
            )

    return input_grav

def regional_seperation(
    input_grav: pd.DataFrame,
    grav_spacing: int,
    regional_method: str,
    crs: str = "3031",
    registration="g",
    **kwargs,
):
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
    input_grav_column = kwargs.get("input_grav_column", "grav")

    # if inversion region not supplied, extract from dataframe
    inversion_region = kwargs.get(
        "inversion_region", vd.get_region((input_grav.x, input_grav.y))
    )

    # get kwargs associated with the various methods
    trend = kwargs.get("trend", None)
    filter = kwargs.get("filter", None)
    tension_factor = kwargs.get("tension_factor", 0.25)
    eq_sources = kwargs.get("eq_sources", None)

    df = kwargs.get("constraints", None)
    if df is not None:
        constraints = df.copy()

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

    # calculate misfit
    anomalies = misfit(
        input_grav = anomalies,
        input_forward_column = input_forward_column,
        input_grav_column = input_grav_column,
        grav_spacing=grav_spacing,
        plot=False,
    )

    # grid misfit
    misfit_grid = pygmt.xyz2grd(
        data=anomalies[["x", "y", "misfit"]],
        region=inversion_region,
        spacing=grav_spacing,
        registration=registration,
    )

    # anomalies["grav_corrected"] = anomalies[input_grav_column]
    # anomalies["forward"] = anomalies[input_forward_column]

    # # get obs-forward misfit
    # anomalies["misfit"] = anomalies.grav_corrected - anomalies.forward

    # # grid the misfits, used in trend, filter, and constraints, not in
    # Eq. Sources
    # misfit = pygmt.xyz2grd(
    #     data=anomalies[["x", "y", "misfit"]],
    #     region=inversion_region,
    #     spacing=grav_spacing,
    #     registration=registration,
    # )

    # Trend method
    if regional_method == "trend":
        # fill misfit nans with 1 of 2 methods
        if fill_method == "pygmt":
            # option 1) with pygmt.grdfill(), needs grav_spacing and
            # inversion_region
            misfit_filled = pygmt.grdfill(misfit_grid, mode="n")
        elif fill_method == "rioxarray":
            # option 1) with rio.interpolate(), needs crs set.
            # misfit = misfit.rio.write_crs(crs)
            misfit_filled = (
                misfit.rio.write_nodata(np.nan).rio.interpolate_na().rename("z")
            )

        df = vd.grid_to_table(misfit_filled).astype("float64")

        trend = vd.Trend(degree=trend).fit((df.x, df.y.values), df.z)
        anomalies["reg"] = trend.predict((anomalies.x, anomalies.y))

    # Filter method
    elif regional_method == "filter":
        # filter the observed-forward misfit with the provided filter in meters
        regional_misfit = pygmt.grdfilter(
            misfit_grid,
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
            grid=misfit_grid,
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
            T=tension_factor,
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
    prism_spacing: float,
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
            prism_spacing,
            prism_density / 1000,  # density
        )
    return jac


def jacobian_annular(
    coordinates: pd.DataFrame,
    model: xr.Dataset,
    prism_spacing: float,
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
    prism_spacing : float
        spacing of prisms

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
        prism_spacing,
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

    # for zref method
    # Build a small prism ontop of existing prism (thickness equal to delta)
    for col, (prism, density) in enumerate(prisms_n_density):
        # for positive densities, add prisms on top
        if density >= 0:
            bottom = prism[5]
            top = prism[5] + delta
        # for negative densities, add prism below
        elif density < 0:
            top = prism[4]
            bottom = prism[4] - delta
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

    # for absolute density discretization method
#     # Build a small prism ontop of existing prism (thickness equal to delta)
#     for col, (prism, density) in enumerate(prisms_n_density):
#         bottom = prism[5]  # - delta / 2
#         top = prism[5] + delta  # / 2
#         delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)

#         jac[:, col] = (
#             hm.prism_gravity(
#                 coordinates=(grav_x, grav_y, grav_z),
#                 prisms=delta_prism,
#                 density=density,
#                 field=field,
#                 parallel=True,
#             )
#             / delta
#         )

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
    solver_type: str = "verde least squares",
):
    """
    Calculate shift to add to prism's for each iteration of the inversion.Finds
    the least-squares solution to the Jacobian and the gravity residual

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
    solver_type : {
        'verde least squares',
        'scipy least squares',
        'scipy conjugate',
        'numpy least squares',
        'steepest descent',
        'gauss newton',
        } optional
        choose which solving method to use, by default "verde least squares"

    Returns
    -------
    np.array
        array of corrrection values to apply to each prism.
    """
    if solver_type == "verde least squares":
        """
        if damping not None, uses sklearn.linear_model.Ridge(alpha=damping)
        alpha: 0 to +inf. multiplies the L2 term, can also pass an array
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
        """
        step = vd.base.least_squares(
            jacobian,
            residuals,
            weights=weights,
            damping=damping, # float, typically 100-10,000
            copy_jacobian=False,
        )
    elif solver_type == "scipy least squares":
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
        """
        if damping is None:
            damping = 0
        step = sp.sparse.linalg.lsqr(
            jacobian,
            residuals,
            show=False,
            damp=damping, # float, typically 0-1
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
    solver_weights: np.array,
        array of weights to assign to data, typically 1/(uncertainty**2)
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


def geo_inversion(
    input_grav: pd.DataFrame,
    input_grav_column: str,
    prism_layer: xr.Dataset,
    max_iterations: int,
    l2_norm_tolerance: float = 0.2,
    delta_l2_norm_tolerance: float = 1.001,
    deriv_type: str = "prisms",
    jacobian_prism_size: float = 1,
    solver_type: str = "verde least squares",
    solver_damping: float = None,
    solver_weights: np.array = None,
    max_layer_change_per_iter: float = None,
    upper_confining_layer: xr.DataArray = None,
    lower_confining_layer: xr.DataArray = None,
    ):

    time_start = time.perf_counter()

    gravity = copy.deepcopy(input_grav)
    prisms = copy.deepcopy(prism_layer)

    assert prisms.top.values.min() == prisms.bottom.values.max()
    assert prisms.density.values.max() == -prisms.density.values.min()

    density_contrast = prisms.density.values.max()
    zref = prisms.top.values.min()

    # turn dataset into dataframe
    prisms_df = (
            prisms
            .to_dataframe()
            .reset_index()
            .dropna()
            .astype(float)
        )

    # if there is a confining surface (above or below), which the inverted layer
    # shouldn't intersect, then sample those layers into the df
    if upper_confining_layer is not None:
        prisms_df = profile.sample_grids(
                df=prisms_df,
                grid=upper_confining_layer,
                name="upper_bounds",
                coord_names=['easting', 'northing'],
            )
    if lower_confining_layer is not None:
        prisms_df = profile.sample_grids(
                df=prisms_df,
                grid=lower_confining_layer,
                name="lower_bounds",
                coord_names=['easting', 'northing'],
            )

    # set starting delta L2 norm to positive infinity
    delta_l2_norm = np.Inf

    # iteration times
    iter_times=[]

    # start the inversion loop
    for ITER, _ in enumerate(range(max_iterations), start=1):
        print(f"\n{'':#<60}##################################\niteration {ITER}")
        # start iteration timer
        iter_time_start = time.perf_counter()

        # after first iteration reset residual with previous iteration's results
        if ITER == 1:
            pass
        else:
            gravity["res"] = gravity[f"iter_{ITER-1}_final_misfit"]
            prisms_df["density"] = prisms_df[f"iter_{ITER-1}_density"]

        # add starting residual to df
        gravity[f"iter_{ITER}_initial_misfit"] = gravity.res

        # set iteration stats
        initial_RMSE = RMSE(gravity[f"iter_{ITER}_initial_misfit"])
        l2_norm = np.sqrt(initial_RMSE)

        if ITER == 1:
            starting_l2_norm = l2_norm

        # calculate jacobian sensitivity matrix (2 methods)
        if deriv_type == "annulus":  # major issue with grav_column_der, way too slow
            #get spacing of prisms
            prism_spacing = abs(
                prisms_df.northing.unique()[1] - prisms_df.northing.unique()[0])
            jac = jacobian_annular(
                gravity,
                prisms,
                prism_spacing,
            )
        elif deriv_type == "prisms":
            jac = jacobian_prism(
                coordinates=gravity,
                model=prisms,
                delta=jacobian_prism_size,
                field="g_z",
            )
        else:
            print("not valid derivative type")

        # calculate correction for each prism
        Surface_correction = solver(
            jacobian=jac,
            residuals=gravity.res.values,
            weights=solver_weights,
            damping=solver_damping,
            solver_type=solver_type,
        )

        # print correction values
        print(
            f"Layer correction median: {int(np.median(Surface_correction))}",
            f"m, RMSE:{int(RMSE(Surface_correction))} m",
        )

        # set maximum allowed change of each prism per iteration
        if max_layer_change_per_iter is not None:
            for i, j in enumerate(Surface_correction):
                if j > max_layer_change_per_iter:
                    j = max_layer_change_per_iter
                elif j < -max_layer_change_per_iter:
                    j = -max_layer_change_per_iter
            print(
                "Layer correction (after clipped) median:",
                f"{int(np.median(Surface_correction))}m,",
                f"RMSE:{int(RMSE(Surface_correction))} m",
            )

        # ensure prisms don't extend above or below confining surfaces.
        # i.e., don't let bed move above ice base/water surface
        if upper_confining_layer is not None:
            # get max change in positive direction for each prism
            prisms_df["max_allowed_change_above"] = prisms_df.upper_bounds - prisms_df.top
            for i, j in enumerate(prisms_df.max_allowed_change_above):
                if Surface_correction[i] > j:
                    Surface_correction[i] = j
        if lower_confining_layer is not None:
            # get max change in negative direction for each prism
            prisms_df["max_allowed_change_below"] = prisms_df.lower_bounds - prisms_df.bottom
            for i, j in enumerate(prisms_df.max_allowed_change_below):
                if Surface_correction[i] < j:
                    Surface_correction[i] = j

        # add corrections to prisms_df
        prisms_df = pd.concat([prisms_df, pd.DataFrame({f"iter_{ITER}_correction": Surface_correction})], axis=1)

        # constrain corrections to only within gravity region
        # prisms_df['inside'] = vd.inside(
        #     (prisms_df.easting, prisms_df.northing), region=inversion_region)
        # prisms_df.loc[prisms_df.inside == False, f"iter_{ITER}_correction"] = 0

        # for negative densities, negate the correction
        prisms_df.loc[prisms_df.density < 0, f"iter_{ITER}_correction"] *= -1

        # create surface from top and bottom
        surface_grid = xr.where(
            prisms.density > 0, prisms.top, prisms.bottom)

        # grid the corrections
        correction_grid = (
            prisms_df
            .rename(columns={f"iter_{ITER}_correction":'z'})
            .set_index(["northing", "easting"])
            .to_xarray().z
            )

        # apply correction to surface
        surface_grid += correction_grid

        # update the prism layer
        prisms.prism_layer.update_top_bottom(
            surface=surface_grid,
            reference=zref
        )

        prisms['density'] = xr.where(prisms.top > zref, density_contrast, -density_contrast)

        prisms['surface'] = surface_grid

        # turn back into dataframe
        prisms_iter = (
            prisms
            .to_dataframe()
            .reset_index()
            .dropna()
            .astype(float)
        )

        # add new cols to dict
        dict_of_cols={
            f"iter_{ITER}_top": prisms_iter.top,
            f"iter_{ITER}_bottom": prisms_iter.bottom,
            f"iter_{ITER}_density": prisms_iter.density,
            f"iter_{ITER}_layer": prisms_iter.surface,
        }

        prisms_df = pd.concat([prisms_df, pd.DataFrame(dict_of_cols)], axis=1)

        # update the forward gravity
        gravity[f"iter_{ITER}_forward_grav"] = prisms.prism_layer.gravity(
            coordinates=(gravity.x, gravity.y, gravity.z), field="g_z"
        )
        # center on 0
        gravity[f"iter_{ITER}_forward_grav"] -= gravity[f"iter_{ITER}_forward_grav"].median()

        # each iteration updates the topography of the layer to minizime the residual
        # portion of the misfit. We then want to recalculate the forward gravity of the
        # new layer, use the same original regional misfit, and re-calculate the residual
        # Gmisfit  = Gobs_corr - Gforward
        # Gres = Gmisfit - Greg
        # Gres = Gobs_corr - Gforward - Greg
        # update the residual misfit with the new forward gravity and the same regional
        gravity[f"iter_{ITER}_final_misfit"] = (
            gravity[input_grav_column] -
            gravity[f"iter_{ITER}_forward_grav"] -
            gravity.reg
        )

        # center on 0
        gravity[f"iter_{ITER}_final_misfit"] -= gravity[f"iter_{ITER}_final_misfit"].median()

        # update the misfit RMSE
        updated_RMSE = RMSE(gravity[f"iter_{ITER}_final_misfit"])
        print(f"\nupdated misfit RMSE: {round(updated_RMSE, 2)}")

        # square-root of RMSE is the l-2 norm
        updated_l2_norm = np.sqrt(updated_RMSE)

        print(
            f"updated L2-norm: {round(updated_l2_norm, 2)}, ",
            f"tolerance: {l2_norm_tolerance}",
        )

        updated_delta_l2_norm = l2_norm / updated_l2_norm

        print(
            f"updated delta L2-norm : {round(updated_delta_l2_norm, 2)}, ",
            f"tolerance: {delta_l2_norm_tolerance}",
        )
        # we want the misfit (L2-norm) to be steadily decreasing with each iteration.
        # If it increases, something is wrong, stop inversion
        # If it doesn't decrease enough, inversion has finished and can be stopped
        # delta L2 norm starts at +inf, and should decreases with each iteration.
        # if it gets close to 1, the iterations aren't making progress and can be stopped.
        # a value of 1.001 means the L2 norm has only decrease by 0.1% between iterations.
        # and RMSE has only decreased by 0.05%.

        # update the l2_norm
        l2_norm = updated_l2_norm

        # updated the delta l2_norm
        delta_l2_norm = updated_delta_l2_norm

        # end iteration timer
        iter_time_end = time.perf_counter()
        iter_times.append(iter_time_end-iter_time_start)

        if l2_norm > starting_l2_norm*1.2:
            print(
                f"\nInversion terminated after {ITER} iterations because L2 norm was",
                "more than 20% greater than starting L2 norm",
            )
            break

        if delta_l2_norm <= delta_l2_norm_tolerance: # and ITER !=1:
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

    time_end = time.perf_counter()

    elapsed_time = int(time_end-time_start)

    # collect input parameters into a dictionary
    params = {
        "density_contrast": f"{density_contrast} kg/m3",
        "max_iterations": max_iterations,
        "l2_norm_tolerance": f"{l2_norm_tolerance}",
        "delta_l2_norm_tolerance": f"{delta_l2_norm_tolerance}",

        "deriv_type": deriv_type,
        "jacobian_prism_size": f"{jacobian_prism_size} m",
        "solver_type": solver_type,
        "solver_damping": solver_damping,
        "solver_weights": 'Not enabled' if solver_weights is None else 'Enabled',

        "upper_confining_layer": 'Not enabled' if upper_confining_layer is None else 'Enabled',
        "lower_confining_layer": 'Not enabled' if lower_confining_layer is None else 'Enabled',
        "max_layer_change_per_iter": f"{max_layer_change_per_iter} m",
        "time_elapsed": f"{elapsed_time} seconds",
        "average_iteration_time": f"{round(np.mean(iter_times), 2)} seconds",
    }

    return prisms_df, gravity, params, elapsed_time

def inversion_RMSE(
    true_surface,
    constraints=None,
    plot=False,
    **kwargs
):
    """
    Calculate the RMSE of the inversion results compared with the true starting topography.
    """

    # run inversion
    with inv_utils.HiddenPrints():
        prism_results, grav_results, params, elapsed_time = geo_inversion(
            **kwargs
        )

    # grid resulting prisms
    ds = prism_results.rename(columns={'easting':'x', 'northing':'y'}).set_index(['y', 'x']).to_xarray()

    # subset the inversion region
    # ds_inner = ds.sel(
    #     x=slice(inversion_region[0], inversion_region[1]),
    #     y=slice(inversion_region[2], inversion_region[3])
    # )

    # get last iteration's layer result
    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]

    final_surface = ds[cols[-1]]

    dif = true_surface - final_surface

    rmse = RMSE(dif)

    if constraints is not None:
        df = profile.sample_grids(
            df=constraints,
            grid=final_surface,
            name="final_surface",
        )
        constraints_rmse = round(RMSE(df.z - df.final_surface), 2)
    else:
        constraints_rmse = None

    if plot:
        grids = utils.grd_compare(
            true_surface,
            final_surface,
            grid1_name='True',
            grid2_name='Final',
            plot=True,
            plot_type="xarray",
            # cmap='batlowW',
            cmap="gist_earth",
            robust=True,
            points=constraints,
            hist=True,
            inset=False,
        )

    return rmse, prism_results, grav_results, params, elapsed_time, constraints_rmse