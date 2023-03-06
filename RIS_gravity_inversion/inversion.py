import copy
import functools
import itertools
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
    input_forward_column: str = "forward",
    input_grav_column: str = "Gobs_corr",
):
    # if misfit already calculated, drop the column
    try:
        input_grav.drop(columns=["misfit"], inplace=True)
    except KeyError:
        pass

    # get obs-forward misfit
    input_grav["misfit"] = input_grav[input_grav_column] - input_grav[input_forward_column]

    return input_grav


def regional_trend(
    trend: int,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    fill_method: str = "rioxarray",
    crs: str = None,
):
    """
    seperate the regional field with a trend
    """
    # fill misfit nans with 1 of 2 methods
    if fill_method == "pygmt":
        misfit_filled = pygmt.grdfill(misfit_grid, mode="n").rename("grav")
    elif fill_method == "rioxarray":
        misfit_grid = misfit_grid.rio.write_crs(crs)
        misfit_filled = (
            misfit_grid.rio.write_nodata(np.nan).rio.interpolate_na().rename("grav")
        )
    else:
        raise ValueError("invalid string for fill_method")

    df = vd.grid_to_table(misfit_filled).astype("float64")

    trend = vd.Trend(degree=trend).fit((df.x, df.y.values), df.grav)
    anomalies["reg"] = trend.predict((anomalies.easting, anomalies.northing))
     
    return anomalies


def regional_filter(
    filter_width: str,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    registration: str = "g",
):
    """
    seperate the regional field with a low-pass filter
    """
    
    # filter the observed-forward misfit with the provided filter in meters
    regional_misfit = pygmt.grdfilter(
        misfit_grid,
        filter=filter_width,
        distance="0",
        registration=registration,
    )
    # sample the results and merge into the anomalies dataframe
    tmp_regrid = pygmt.grdtrack(
        points=anomalies[["easting", "northing"]],
        grid=regional_misfit,
        newcolname="reg",
        verbose="q",
    )
    anomalies = anomalies.merge(tmp_regrid, on=["easting", "northing"], how="left")
    
    return anomalies


def regional_constraints(
    constraint_points : pd.DataFrame,
    tension_factor : float,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    region: list,
    spacing: float,
    registration: str = "g",
    block_reduce = False,
):
    """
    seperate the regional field by sampling and regridding at the constraint points
    """
    
    constraints_df = constraint_points.copy()
    
     # sample misfit at constraint points
    constraints_df = profile.sample_grids(
        df=constraints_df,
        grid=misfit_grid,
        name="misfit",
        coord_names = ("easting", "northing"),
    )
    
    if block_reduce is True:
        # get median misfit of constraint points in each 1km cell
        constraints_df = pygmt.blockmedian(
            data=constraints_df[["easting", "northing", "misfit"]],
            spacing=grav_spacing,
            region=inversion_region,
            registration=registration,
        )
    
    # grid the entire region misfit based just on the misfit at the constraints
    regional_misfit = pygmt.surface(
        data = constraints_df[["easting", "northing", "misfit"]],
        region = region,
        spacing = spacing,
        registration = registration,
        T = tension_factor,
        verbose = "q",
    )

    # sample the resulting grid and add to anomalies dataframe
    anomalies = profile.sample_grids(
        df = anomalies,
        grid = regional_misfit,
        name = "reg",
        coord_names = ("easting", "northing"),
    )
    
    return anomalies


def regional_eq_sources(
    source_depth : float,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    eq_damping: float = None,
    block_size: float = None,
    depth_type: str = "relative",

):
    """
    seperate the regional field by estimating deep equivalent sources
    
    eq_damping : float: smoothness to impose on estimated coefficients
    block_size : float: block reduce the data to speed up
    depth_type : str: constant depths, not relative to observation heights
    """
    # create set of deep sources
    equivalent_sources = hm.EquivalentSources(
        depth = source_depth,
        damping = eq_damping, 
        block_size = block_size, 
        depth_type = depth_type,  
    )
    
    # fit the source coefficients to the data
    coordinates = (anomalies.easting, anomalies.northing, anomalies.upward)
    equivalent_sources.fit(coordinates, anomalies.misfit)
    
    # use sources to predict the regional field at the observation points
    anomalies["reg"] = equivalent_sources.predict(coordinates)
    
    return anomalies


def regional_seperation(
    input_grav: pd.DataFrame,
    grav_spacing: int,
    regional_method: str,
    crs: str = "3031",
    registration="g",
    **kwargs,
):
    """
    Seperate the regional and resiudal fields of gravity data with 1 of 4 methods.
    
    must provide kwargs with same name as method type.
    for example, if regional_method='trend', must provide a trend order via the kwarg
    'trend'=6.
    for constraints, kwarg is gridding tension factor
    for filter, kwarg is a pygmt filter string, such as "g150e3" for a 150km gaussian
    for eq_sources, kwargs is depth of sources in meters
    """
    if kwargs.get(regional_method) is None:
        raise ValueError(
            f"Must provide keyword argument '{regional_method}' if regional_method ="
            f" {regional_method}."
        )

    input_forward_column = kwargs.get("input_forward_column", "forward_total")
    input_grav_column = kwargs.get("input_grav_column", "grav")

    # if inversion region not supplied, extract from dataframe
    inversion_region = kwargs.get(
        "inversion_region", vd.get_region((input_grav.easting, input_grav.northing))
    )

    # get kwargs associated with the various methods
    trend = kwargs.get("trend", None)
    filter = kwargs.get("filter", None)
    tension_factor = kwargs.get("tension_factor", 0.25)
    eq_sources = kwargs.get("eq_sources", None)

    df = kwargs.get("constraints", None)
    if df is not None:
        constraints = df.copy()

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
    )

    # grid misfit
    misfit_grid = pygmt.xyz2grd(
        data = anomalies[["easting", "northing", "misfit"]],
        region = inversion_region,
        spacing = grav_spacing,
        registration = registration,
    )

    if regional_method == "trend":
        anomalies = regional_trend(
            trend,
            misfit_grid,
            anomalies,
            fill_method = kwargs.get("fill_method", "rioxarray"),
            crs = crs,
        )
    elif regional_method == "filter":
        anomalies = regional_filter(
            filter,
            misfit_grid,
            anomalies,
            registration = registration,
        )
    elif regional_method == "constraints":
        anomalies = regional_constraints(
            constraints,
            tension_factor,
            misfit_grid,
            anomalies,
            inversion_region,
            grav_spacing,
            registration = registration,
            block_reduce = False,
        )
    elif regional_method == "eq_sources":
        anomalies = regional_eq_sources(
            eq_sources,
            misfit_grid,
            anomalies,
            eq_damping = kwargs.get("eq_damping", None),
            block_size = kwargs.get("block_size", None),
            depth_type = kwargs.get("depth_type", "relative"),
        )
    else:
        raise ValueError("invalid string for regional_method")
            
    # calculate the residual field
    anomalies["res"] = anomalies.misfit - anomalies.reg

    return anomalies



@numba.jit(cache=True, nopython=True)
def grav_column_der(x0, y0, z0, xc, yc, z1, z2, res, rho):
    """
    Function to calculate the vertical derivate of the gravitational acceleration at an observation point caused
    by a right, rectangular prism.
    Approximated with Hammer's annulus approximation.
    x0, y0, z0: floats, coordinates of gravity observation points
    xc, yc, z1, z2: floats, coordinates of prism's y, x, top, and bottom, respectively.
    res: float, resolution of prism layer in meters,
    rho: float, density of prisms, in kg/m^3
    """
    r = np.sqrt(np.square(x0 - xc) + np.square(y0 - yc))
    r1 = r - 0.5 * res
    r2 = r + 0.5 * res
    r1[r1 < 0] = 0  # will fail if prism is under obs point
    r2[r1 < 0] = 0.5 * res
    f = np.square(res) / (np.pi * (np.square(r2) - np.square(r1)))  # eq 2.19 in McCubbine 2016 Thesis
    anomaly_grad = (
        0.0000419 * f * rho * (z1 - z0) * (
            1 / np.sqrt(np.square(r2) + np.square(z1 - z0)) - 
            1 / np.sqrt(np.square(r1) + np.square(z1 - z0))
        ))
    return anomaly_grad


@numba.njit(parallel=True)
def jacobian_annular(
    grav_easting,
    grav_northing,
    grav_upward,
    prism_easting,
    prism_northing,
    prism_top,
    prism_bottom,
    prism_density,
    prism_spacing: float,
    jac: np.ndarray,
):
    """
    Function to calculate the Jacobian matrix using the annular cylinder approximation
    The resulting Jacobian is a matrix (numpy array) with a row per gravity observation
    and a column per prism. This approximates the prisms as an annulus, and calculates
    it's vertical gravity derivative.
    
    Takes arrays from `jacobian`, feeds them into `grav_column_der`, and returns
    the jacobian.

    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
    """

    for i in numba.prange(len(grav_easting)):
        jac[i, :] = grav_column_der(
            grav_northing[i],
            grav_easting[i],
            grav_upward[i],
            prism_northing,
            prism_easting,
            prism_top,
            prism_bottom,
            prism_spacing,
            prism_density,
        )
    # for i in numba.prange(len(grav_easting)):
    #     for j in range(len(prism_northing)):
    #         jac[i, j] = grav_column_der(
    #             grav_northing[i],
    #             grav_easting[i],
    #             grav_upward[i],
    #             prism_northing[j],
    #             prism_easting[j],
    #             prism_top[j],
    #             prism_bottom[j],
    #             prism_spacing[j],
    #             prism_density[j],
    #         )
    return jac


@numba.jit(forceobj=True, parallel=True)
def jacobian_prism(
    prisms_properties,
    grav_easting,
    grav_northing,
    grav_upward,
    delta: float,
    jac: np.ndarray,
):
    """
    Function to calculate the Jacobian matrix with the vertical gravity derivative
    as a numerical approximation with small prisms

    Takes arrays from `jacobian` and calculates the jacobian.
    
    Returns
    -------
    np.ndarray
        returns a np.ndarray of shape (number of gravity points, number of prisms)
        
    """
    # Build a small prism ontop of existing prism (thickness equal to delta)
    for i in numba.prange(len(prisms_properties)):
            prism = prisms_properties[i]
            density = prism[6]
            bottom = prism[5] 
            top = prism[5] + delta  
            delta_prism = (prism[0], prism[1], prism[2], prism[3], bottom, top)

            jac[:, i] = hm.prism_gravity(
                    coordinates=(grav_easting, grav_northing, grav_upward),
                    prisms=delta_prism,
                    density=density,
                    field="g_z",
                    parallel=True,
                ) / delta

    return jac


def jacobian(
    deriv_type: str,
    coordinates: pd.DataFrame,
    empty_jac: np.ndarray = None,
    prisms_df = None,
    prisms_layer = None,
    prism_spacing = None,
    prism_size = None,
):
    """
    dispatcher for creating the jacobian matrix with 2 method options
    """
    # convert dataframes to numpy arrays
    coordinates_array = coordinates.to_numpy()

    # get various arrays based on gravity column names
    grav_easting = coordinates_array[:, coordinates.columns.get_loc("easting")]
    grav_northing = coordinates_array[:, coordinates.columns.get_loc("northing")]
    grav_upward = coordinates_array[:, coordinates.columns.get_loc("upward")]

    assert len(grav_easting) == len(grav_northing) == len(grav_upward)
    
    if empty_jac is None:
        empty_jac = np.empty(
            (len(grav_easting), len(prisms_df.easting)),
            dtype=np.float64,
        )
        print("no empty jacobian supplied")
    
    jac = empty_jac.copy()
        
    if deriv_type == "annulus":
        # convert dataframes to numpy arrays
        prisms_array = prisms_df.to_numpy()

        # get various arrays based on prisms column names
        prism_easting = prisms_array[:, prisms_df.columns.get_loc("easting")]
        prism_northing = prisms_array[:, prisms_df.columns.get_loc("northing")]
        prism_top = prisms_array[:, prisms_df.columns.get_loc("top")]
        prism_bottom = prisms_array[:, prisms_df.columns.get_loc("bottom")]
        prism_density = prisms_array[:, prisms_df.columns.get_loc("density")]
        
        jac = jacobian_annular(
            grav_easting,
            grav_northing,
            grav_upward,
            prism_easting,
            prism_northing,
            prism_top,
            prism_bottom,
            prism_density,
            prism_spacing,
            jac,
        )
    elif deriv_type == "prisms":
        # get prisms info in following format, 3 methods:
        # ((west, east, south, north, bottom, top), density)

        # with itertools.combinations
        prisms_properties = []
        for y, x, in itertools.product(range(prisms_layer.northing.size), range(prisms_layer.easting.size)):
            prisms_properties.append(
                    list(prisms_layer.prism_layer.get_prism((y, x)))+[prisms_layer.density.values[y, x]])
        prisms_properties = np.array(prisms_properties)
        # np.asarray(prisms_properties) 

        # with nested for-loops
        # prisms_properties = []
        # for y in range(prisms_layer.northing.size):
        #     for x in range(prisms_layer.easting.size):
        #         prisms_properties.append(
        #             list(prisms_layer.prism_layer.get_prism((y, x)))+[prisms_layer.density.values[y, x]])
        # np.asarray(prisms_properties) 

        # with a generator
        # slower, but doesn't allocate memory
        # prisms_properties = [
        #     list(prisms_layer.prism_layer.get_prism((y, x))) + [prisms_layer.density.values[y, x]]
        #     for y in range(prisms_layer.northing.size)
        #     for x in range(prisms_layer.easting.size)
        # ]
        
        jac = jacobian_prism(
            prisms_properties,
            grav_easting,
            grav_northing,
            grav_upward,
            prism_size,
            jac,
        )

    else:
        raise ValueError("invalid string for deriv_type")

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
    else:
        raise ValueError("invalid string for solver_type")
        
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
            coord_names=['easting', 'northing'],
        )
    if lower_confining_layer is not None:
        df = profile.sample_grids(
            df=df,
            grid=lower_confining_layer,
            name="lower_bounds",
            coord_names=['easting', 'northing'],
        )
    return df


def enforce_confining_surface(
    prisms_df: pd.DataFrame, 
    Surface_correction: np.array,
    upper_confining_layer: xr.DataArray = None, 
    lower_confining_layer: xr.DataArray = None, 
):
    """
    ensue the surface correction doesn't move the prisms above or below optional confining surfaces
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
        prisms_df["max_allowed_change_below"] = prisms_df.lower_bounds - prisms_df.bottom
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
   

def apply_surface_correction(
    prisms_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    iteration_number: int,
):
    """
    update the prisms dataframe and dataset with the surface correction
    """
    
    df = prisms_df.copy()
    ds = prisms_ds.copy()
    
    density_contrast = ds.density.values.max()
    zref = ds.top.values.min()
    
    # for negative densities, negate the correction
    df.loc[df.density < 0, f"iter_{iteration_number}_correction"] *= -1

    # create surface from top and bottom
    surface_grid = xr.where(
        ds.density > 0, ds.top, ds.bottom)

    # grid the corrections
    correction_grid = (
        df
        .rename(columns={f"iter_{iteration_number}_correction":'z'})
        .set_index(["northing", "easting"])
        .to_xarray().z
        )

    # apply correction to surface
    surface_grid += correction_grid

    # update the prism layer
    ds.prism_layer.update_top_bottom(
        surface=surface_grid,
        reference=zref
    )

    ds['density'] = xr.where(ds.top > zref, density_contrast, -density_contrast)

    ds['surface'] = surface_grid

    # turn back into dataframe
    prisms_iter = (
        ds
        .to_dataframe()
        .reset_index()
        .dropna()
        .astype(float)
    )
    
    # add new cols to dict
    dict_of_cols={
        f"iter_{iteration_number}_top": prisms_iter.top,
        f"iter_{iteration_number}_bottom": prisms_iter.bottom,
        f"iter_{iteration_number}_density": prisms_iter.density,
        f"iter_{iteration_number}_layer": prisms_iter.surface,
    }

    df = pd.concat([df, pd.DataFrame(dict_of_cols)], axis=1)

    return df, ds


def update_gravity_and_misfit(
    gravity_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    input_grav_column: str,
    iteration_number: int,
):
    gravity = gravity_df.copy()
    
    # update the forward gravity
    gravity[f"iter_{iteration_number}_forward_grav"] = prisms_ds.prism_layer.gravity(
        coordinates=(gravity.easting, gravity.northing, gravity.upward), field="g_z"
    )
    # center on 0
    gravity[f"iter_{iteration_number}_forward_grav"] -= gravity[f"iter_{iteration_number}_forward_grav"].median()

    # each iteration updates the topography of the layer to minizime the residual
    # portion of the misfit. We then want to recalculate the forward gravity of the
    # new layer, use the same original regional misfit, and re-calculate the residual
    # Gmisfit  = Gobs_corr - Gforward
    # Gres = Gmisfit - Greg
    # Gres = Gobs_corr - Gforward - Greg
    # update the residual misfit with the new forward gravity and the same regional
    gravity[f"iter_{iteration_number}_final_misfit"] = (
        gravity[input_grav_column] -
        gravity[f"iter_{iteration_number}_forward_grav"] -
        gravity.reg
    )

    # center on 0
    gravity[f"iter_{iteration_number}_final_misfit"] -= gravity[f"iter_{iteration_number}_final_misfit"].median()
        
    return gravity


def update_l2_norms(
    RMSE: float,
    l2_norm: float,
):
    """ update the l2 norm and delta l2 norm of the misfit"""
    # square-root of RMSE is the l-2 norm
    updated_l2_norm = np.sqrt(RMSE)

    updated_delta_l2_norm = l2_norm / updated_l2_norm

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

    return l2_norm, delta_l2_norm, 


def end_inversion(
    iteration_number: int,
    max_iterations: int,
    l2_norm: float,
    starting_l2_norm: float,
    l2_norm_tolerance: float,
    delta_l2_norm: float,
    delta_l2_norm_tolerance: float,
):
    if l2_norm > starting_l2_norm*1.2:
        print(
            f"\nInversion terminated after {iteration_number} iterations because L2 norm was",
            "more than 20% greater than starting L2 norm",
        )
        return True 

    if delta_l2_norm <= delta_l2_norm_tolerance: # and ITER !=1:
        print(
            f"\nInversion terminated after {iteration_number} iterations because there was",
            "no significant variation in the L2-norm",
        )
        return True 

    if l2_norm < l2_norm_tolerance:
        print(
            f"\nInversion terminated after {iteration_number} iterations with L2-norm =",
            f"{round(l2_norm, 2)} because L2-norm < {l2_norm_tolerance}",
        )
        return True 

    if iteration_number >= max_iterations:
        print(
            f"\nInversion terminated after {iteration_number} iterations with L2-norm =",
            f"{round(l2_norm, 2)} because maximum number of iterations ",
            f"({max_iterations}) reached",
        )
        return True 
        
        
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
    """
    perform a geometry inversion, where a topographic surface, represented by a layer of vertical, right-rectangular prisms, has it's topography updated based on a gravity anomaly.  
    """
    time_start = time.perf_counter()

    gravity = copy.deepcopy(input_grav)
    prisms = copy.deepcopy(prism_layer)

    assert prisms.top.values.min() == prisms.bottom.values.max()
    assert prisms.density.values.max() == -prisms.density.values.min()

    density_contrast = prisms.density.values.max()
    zref = prisms.top.values.min()

    # add starting surface to dataset
    surface_grid = xr.where(
            prisms.density > 0, prisms.top, prisms.bottom)
    prisms['surface'] = surface_grid
    
    # turn dataset into dataframe
    prisms_df = (
            prisms
            .to_dataframe()
            .reset_index()
            .dropna()
            .astype(float)
        )
        
    #get spacing of prisms
    if deriv_type == "annulus":
        prism_spacing = abs(
            prisms_df.northing.unique()[1] - prisms_df.northing.unique()[0])
    else:
        prism_spacing = None
        
    # create empty jacobian matrix
    empty_jac = np.empty(
            (len(gravity[input_grav_column]), prisms.top.size),
            dtype=np.float64,
        )
    
    # if there is a confining surface (above or below), which the inverted layer
    # shouldn't intersect, then sample those layers into the df
    prisms_df = sample_bounding_surfaces(
        prisms_df, 
        upper_confining_layer, 
        lower_confining_layer, 
    )

    # set starting delta L2 norm to positive infinity
    delta_l2_norm = np.Inf

    # iteration times
    iter_times=[]

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

        # calculate jacobian sensitivity matrix 
        jac = jacobian(
            deriv_type,
            gravity,
            empty_jac,
            prisms_df = prisms_df,
            prisms_layer = prisms,
            prism_spacing = prism_spacing,
            prism_size = jacobian_prism_size,
        )

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

        # constrain the surface correction values
        Surface_correction = constrain_surface_correction(
            prisms_df,
            Surface_correction,
            upper_confining_layer, 
            lower_confining_layer, 
            max_layer_change_per_iter,
        )
    
        # add corrections to prisms_df
        prisms_df = pd.concat([prisms_df, pd.DataFrame({f"iter_{ITER}_correction": Surface_correction})], axis=1)
        
        # constrain corrections to only within gravity region
        # prisms_df['inside'] = vd.inside(
        #     (prisms_df.easting, prisms_df.northing), region=inversion_region)
        # prisms_df.loc[prisms_df.inside == False, f"iter_{ITER}_correction"] = 0
        
        # apply the surface correction to the prisms dataframe and dataset
        prisms_df, prisms = apply_surface_correction(
            prisms_df,
            prisms,
            ITER,
        )

        # update the forward gravity and the misfit
        gravity = update_gravity_and_misfit(
            gravity,
            prisms,
            input_grav_column,
            ITER,
        )
    
        # update the misfit RMSE
        updated_RMSE = RMSE(gravity[f"iter_{ITER}_final_misfit"])
        print(f"\nupdated misfit RMSE: {round(updated_RMSE, 2)}")

        # update the l2 and delta l2 norms
        l2_norm, delta_l2_norm = update_l2_norms(updated_RMSE, l2_norm)
        
        print(
            f"updated L2-norm: {round(l2_norm, 2)}, ",
            f"tolerance: {l2_norm_tolerance}")
        print(
            f"updated delta L2-norm : {round(delta_l2_norm, 2)}, ",
            f"tolerance: {delta_l2_norm_tolerance}")
        
        # end iteration timer
        iter_time_end = time.perf_counter()
        iter_times.append(iter_time_end-iter_time_start)
        
        # decide if to end the inversion
        end = end_inversion(
            ITER,
            max_iterations,
            l2_norm,
            starting_l2_norm,
            l2_norm_tolerance,
            delta_l2_norm,
            delta_l2_norm_tolerance,
        )
        if end is True:
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
    ds = prism_results.set_index(['northing', 'easting']).to_xarray()

    # get last iteration's layer result
    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]
    final_surface = ds[cols[-1]]
    
    dif = utils.grd_compare(
            true_surface,
            final_surface,
            plot=False,
        )[0]
    
    # dif = true_surface - final_surface

    rmse = RMSE(dif)
    
    if constraints is not None:
        df = profile.sample_grids(
            df=constraints,
            grid=final_surface,
            name="final_surface",
            coord_names = ("easting", "northing"),
        )
        constraints_rmse = round(RMSE(df.upward - df.final_surface), 2)
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
