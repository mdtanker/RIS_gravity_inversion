import copy
import itertools
import time
import warnings

import harmonica as hm
import numba
import numpy as np
import pandas as pd
import scipy as sp
import verde as vd
import xarray as xr
from antarctic_plots import profile, utils
from constrained_linear_regression import ConstrainedLinearRegression

import RIS_gravity_inversion.utils as inv_utils

warnings.filterwarnings("ignore", message="pandas.Int64Index")
warnings.filterwarnings("ignore", message="pandas.Float64Index")


def misfit(
    input_grav: pd.DataFrame,
    input_forward_column: str = "forward",
    input_grav_column: str = "Gobs_corr",
):
    """
    If the observed gravity column doesn't end in "_shift", apply a DC shift to the
    observed gravity to match its median to the median of the forward gravity.

    Additionally, calculate a 'misfit' defined as observed_shift - forward.

    Parameters
    ----------
    input_grav : pd.DataFrame
        Dataframe containing observed and forward gravity data.
    input_forward_column : str, optional
        Column name for forward gravity, by default "forward"
    input_grav_column : str, optional
        Column name for observed gravity, by default "Gobs_corr"

    Returns
    -------
    pd.DataFrame
        Returns a dataframe with a "misfit" column, and if input_grav_column didn't end
        "_shift", a new observed gravity column with "_shift" appended to the name.
    """

    # if misfit already calculated, drop the column
    try:
        input_grav.drop(columns=["misfit"], inplace=True)
    except KeyError:
        pass

    # set observed gravity misfit to be centered on median value of forward gravity
    # if any("_shift" in s for s in list(input_grav.columns)):
    if "_shift" in input_grav_column:
        # get obs-forward misfit
        input_grav["misfit"] = (
            input_grav[input_grav_column] - input_grav[input_forward_column]
        )
    else:
        # if DC shift hasn't been applied yet, apply it.
        offset = (
            input_grav[input_grav_column].median()
            - input_grav[input_forward_column].median()
        )
        input_grav[f"{input_grav_column}_shift"] = input_grav[input_grav_column].copy()
        input_grav[f"{input_grav_column}_shift"] -= offset
        print(
            f"DC shifted observed gravity by {round(offset,0)}mGal to match forward"
            " gravity."
        )
        # get obs-forward misfit
        input_grav["misfit"] = (
            input_grav[f"{input_grav_column}_shift"] - input_grav[input_forward_column]
        )

    return input_grav


# @numba.jit(cache=True, nopython=True)
# def grav_column_der_old(x0, y0, z0, xc, yc, z1, z2, res, rho):
#     """
#     Function to calculate the vertical derivate of the gravitational acceleration at an
#     observation point caused
#     by a right, rectangular prism.
#     Approximated with Hammer's annulus approximation.
#     x0, y0, z0: floats, coordinates of gravity observation points
#     xc, yc, z1, z2: floats, coordinates of prism's y, x, top, and bottom, respectively.
#     res: float, resolution of prism layer in meters,
#     rho: float, density of prisms, in kg/m^3
#     """
#     r = np.sqrt(np.square(x0 - xc) + np.square(y0 - yc))
#     r1 = r - 0.5 * res
#     r2 = r + 0.5 * res
#     r1[r1 < 0] = 0  # will fail if prism is under obs point
#     r2[r1 < 0] = 0.5 * res
#     f = np.square(res) / (
#         np.pi * (np.square(r2) - np.square(r1))
#     )  # eq 2.19 in McCubbine 2016 Thesis
#     anomaly_grad = (
#         0.0000419
#         * f
#         * rho
#         * (z1 - z0)
#         * (
#             1 / np.sqrt(np.square(r2) + np.square(z1 - z0))
#             - 1 / np.sqrt(np.square(r1) + np.square(z1 - z0))
#         )
#     )
#     return anomaly_grad

@numba.jit(cache=True, nopython=True)
def grav_column_der(
    grav_easting: np.array,
    grav_northing: np.array,
    grav_upward: np.array,
    prism_easting: np.array,
    prism_northing: np.array,
    prism_top: np.array,
    prism_spacing: float,
    prism_density: np.array,
):
    """
    Function to calculate the vertical derivate of the gravitational acceleration at an
    observation point caused
    by a right, rectangular prism.
    Approximated with Hammer's annulus approximation.
    grav_easting, grav_northing, grav_upward: floats, coordinates of gravity observation points
    prism_northing, easting, top: floats, coordinates of prism's center in northing,
    easting, and upward directions, respectively.
    prism_resolution: float, resolution of prism layer in meters,
    prism_density: float, density of prisms, in kg/m^3
    """
    r = np.sqrt(np.square(grav_northing - prism_northing) + np.square(grav_easting - prism_easting))
    r1 = r - 0.5 * prism_spacing
    r2 = r + 0.5 * prism_spacing

    # gravity observation point can't be within prism
    # if it is, instead calculate gravity on prism edge
    r1[r1 < 0] = 0
    r2[r2 < prism_spacing] = prism_spacing

    f = np.square(prism_spacing) / (
        np.pi * (np.square(r2) - np.square(r1))
    )  # eq 2.19 in McCubbine 2016 Thesis
    anomaly_grad = (
        0.0000419
        * f
        * prism_density
        * (prism_top - grav_upward)
        * (
            1 / np.sqrt(np.square(r2) + np.square(prism_top - grav_upward))
            - 1 / np.sqrt(np.square(r1) + np.square(prism_top - grav_upward))
        )
    )
    return anomaly_grad

@numba.njit(parallel=True)
def jacobian_annular(
    grav_easting,
    grav_northing,
    grav_upward,
    prism_easting,
    prism_northing,
    prism_top,
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
            grav_easting[i],
            grav_northing[i],
            grav_upward[i],
            prism_easting,
            prism_northing,
            prism_top,
            prism_spacing,
            prism_density,
        )

    return jac


def prism_properties(
    prisms_layer,
    method="itertools",
):
    """ extract prism properties """

    if method == "itertools":
        prisms_properties = []
        for (
            y,
            x,
        ) in itertools.product(
            range(prisms_layer.northing.size), range(prisms_layer.easting.size)
        ):
            prisms_properties.append(
                list(prisms_layer.prism_layer.get_prism((y, x)))
                + [prisms_layer.density.values[y, x]]
            )
        prisms_properties = np.array(prisms_properties)
    elif method == "forloops":
        prisms_properties = []
        for y in range(prisms_layer.northing.size):
            for x in range(prisms_layer.easting.size):
                prisms_properties.append(
                    list(prisms_layer.prism_layer.get_prism((y, x)))+\
                [prisms_layer.density.values[y, x]])
        np.asarray(prisms_properties)
    elif method == "generator":
        # slower, but doesn't allocate memory
        prisms_properties = [
            list(prisms_layer.prism_layer.get_prism((y, x))) + [
                    prisms_layer.density.values[y, x]]
            for y in range(prisms_layer.northing.size)
            for x in range(prisms_layer.easting.size)
            ]
    else:
        raise ValueError("method must be one of 'itertools', 'forloops', or 'generator'")

    return prisms_properties


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

        jac[:, i] = (
            hm.prism_gravity(
                coordinates=(grav_easting, grav_northing, grav_upward),
                prisms=delta_prism,
                density=density,
                field="g_z",
                parallel=True,
            )
            / delta
        )

    return jac


def jacobian(
    deriv_type: str,
    coordinates: pd.DataFrame,
    empty_jac: np.ndarray = None,
    prisms_layer=None,
    prism_spacing=None,
    prism_size=None,
    apply_weights=False,
    prisms_properties_method="itertools",
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
            (len(grav_easting), prisms_layer.top.size),
            dtype=np.float64,
        )
        print("no empty jacobian supplied")

    jac = empty_jac.copy()

    if deriv_type == "annulus":
        # convert dataframe to arrays
        # arrays = {
        #   k:prisms_layer[k].to_numpy().ravel() for k in list(prisms_layer.variables)}
        df = prisms_layer.to_dataframe().reset_index().dropna().astype(float)
        prism_easting = df._easting.to_numpy()
        prism_northing = df.northing.to_numpy()
        prism_top = df.top.to_numpy()
        prism_density = df.density.to_numpy()

        jac = jacobian_annular(
            grav_easting,
            grav_northing,
            grav_upward,
            prism_easting,
            prism_northing,
            prism_top,
            prism_density,
            prism_spacing,
            jac,
        )

    elif deriv_type == "prisms":
        # get prisms info in following format, 3 methods:
        # ((west, east, south, north, bottom, top), density)

        prisms_properties = prism_properties(
            prisms_layer,
            method=prisms_properties_method,
        )

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

    # scale jacobian by weights grid, based on distance to nearest constraint

    if apply_weights is True:
        prisms_weights = prisms_layer.weights.to_numpy().ravel()
        shape = np.shape(jac)
        jac = np.multiply(jac, prisms_weights)
        assert shape == np.shape(jac)

    return jac


def solver(
    jacobian: np.array,
    residuals: np.array,
    weights: np.array = None,
    damping: float = None,
    solver_type: str = "scipy least squares",
    bounds=None,
    surface=None,
):
    """
    Calculate shift to add to prism's for each iteration of the inversion. Finds
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
        choose which solving method to use, by default "scipy least squares"

    Returns
    -------
    np.array
        array of corrrection values to apply to each prism.
    """
    if solver_type == "verde least squares":
        """
        if damping not None, uses sklearn.linear_model.Ridge(alpha=damping)
        alpha: 0 to +inf. multiplies the L2 term, can also pass an array
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html # noqa E501
        """
        step = vd.base.least_squares(
            jacobian=jacobian,
            data=residuals,
            weights=weights,
            damping=damping,  # float, typically 100-10,000
            copy_jacobian=False,
        )

    elif solver_type == "scipy least squares":
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html # noqa E501
        """
        if damping is None:
            damping = 0
        step = sp.sparse.linalg.lsqr(
            A=jacobian,
            b=residuals,
            show=False,
            damp=damping,  # float, typically 0-1
        )[0]

    elif solver_type == "scipy constrained":
        """
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear # noqa E501
        """
        if bounds is None:
            step = sp.optimize.lsq_linear(
                A=jacobian,
                b=residuals,
                method="trf",
                max_iter=5,
            )["x"]
        else:
            step = sp.optimize.lsq_linear(
                A=jacobian,
                b=residuals,
                bounds=bounds,
                method="trf",
                max_iter=5,
            )["x"]
    # elif solver_type == "scipy nonlinear lsqr":
    #     """
    #     https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html#scipy.optimize.least_squares # noqa E501
    #     """
    #     if bounds is None:
    #         bounds = [-np.inf, np.inf]


    elif solver_type == "CLR":
        """
        https://github.com/avidale/constrained-linear-regression
        """
        model = ConstrainedLinearRegression(
            # max_iter=2,
            ridge=damping,
            # fit_intercept=False,
        )
        if bounds is None:
            step = model.fit(
                X=jacobian,
                y=residuals,
            ).coef_
        else:
            step = model.fit(
                X=jacobian,
                y=residuals,
                min_coef=bounds[0],
                max_coef=bounds[1],
            ).coef_

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
        """ Jacobian transppose algorithm """
        residuals = residuals
        step = jacobian.T @ residuals

    elif solver_type == "gauss newton":
        """
        Gauss Newton w/ 1st order Tikhonov regularization
        from https://nbviewer.org/github/compgeolab/2020-aachen-inverse-problems/blob/main/gravity-inversion.ipynb # noqa E501
        """
        if damping in [None, 0]:
            hessian = jacobian.T @ jacobian
            gradient = jacobian.T @ residuals
        else:
            fdmatrix = finite_difference_matrix(jacobian[1].size)
            hessian = jacobian.T @ jacobian + damping * fdmatrix.T @ fdmatrix
            gradient = (
                jacobian.T @ residuals - damping * fdmatrix.T @ fdmatrix @ surface
            )

        # scipy solver appears to be slightly faster
        # step = np.linalg.solve(hessian, gradient)
        step = sp.linalg.solve(hessian, gradient)

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


def apply_surface_correction(
    prisms_df: pd.DataFrame,
    prisms_ds: xr.Dataset,
    iteration_number: int,
    zref: float,
):
    """
    update the prisms dataframe and dataset with the surface correction
    """

    df = prisms_df.copy()
    ds = prisms_ds.copy()

    density_contrast = ds.density.values.max()
    # zref = ds.top.values.min()

    # for negative densities, negate the correction
    df.loc[df.density < 0, f"iter_{iteration_number}_correction"] *= -1

    # create surface from top and bottom
    surface_grid = xr.where(ds.density > 0, ds.top, ds.bottom)

    # grid the corrections
    correction_grid = (
        df.rename(columns={f"iter_{iteration_number}_correction": "z"})
        .set_index(["northing", "easting"])
        .to_xarray()
        .z
    )

    # apply correction to surface
    surface_grid += correction_grid

    # update the prism layer
    ds.prism_layer.update_top_bottom(surface=surface_grid, reference=zref)

    ds["density"] = xr.where(ds.top > zref, density_contrast, -density_contrast)

    ds["surface"] = surface_grid

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
        coordinates=(gravity.easting, gravity.northing, gravity.upward),
        field="g_z",
    )

    # center on 0
    # Could be introducing a DC shift into the misfit by removing the median.
    # might be better to sample the misfit at the constraints,
    # gravity[f"iter_{iteration_number}_forward_grav"] -= gravity[
    #   f"iter_{iteration_number}_forward_grav"].median()

    # each iteration updates the topography of the layer to minizime the residual
    # portion of the misfit. We then want to recalculate the forward gravity of the
    # new layer, use the same original regional misfit, and re-calculate the residual
    # Gmisfit  = Gobs_corr - Gforward
    # Gres = Gmisfit - Greg
    # Gres = Gobs_corr_shift - Gforward - Greg
    # update the residual misfit with the new forward gravity and the same regional
    gravity[f"iter_{iteration_number}_final_misfit"] = (
        gravity[input_grav_column]
        - gravity[f"iter_{iteration_number}_forward_grav"]
        - gravity.reg
    )

    # center on 0
    # gravity[f"iter_{iteration_number}_final_misfit"] -= gravity[
    #   f"iter_{iteration_number}_final_misfit"].median()

    return gravity


def update_l2_norms(
    RMSE: float,
    l2_norm: float,
):
    """update the l2 norm and delta l2 norm of the misfit"""
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

    return (
        l2_norm,
        delta_l2_norm,
    )


def end_inversion(
    iteration_number: int,
    max_iterations: int,
    l2_norm: float,
    starting_l2_norm: float,
    l2_norm_tolerance: float,
    delta_l2_norm: float,
    delta_l2_norm_tolerance: float,
):
    if l2_norm > starting_l2_norm * 1.2:
        print(
            f"\nInversion terminated after {iteration_number} iterations because L2 ",
            "norm was more than 20% greater than starting L2 norm",
        )
        return True

    if delta_l2_norm <= delta_l2_norm_tolerance:  # and ITER !=1:
        print(
            f"\nInversion terminated after {iteration_number} iterations because there",
            " was no significant variation in the L2-norm",
        )
        return True

    if l2_norm < l2_norm_tolerance:
        print(
            f"\nInversion terminated after {iteration_number} iterations with L2-norm ",
            f"={round(l2_norm, 2)} because L2-norm < {l2_norm_tolerance}",
        )
        return True

    if iteration_number >= max_iterations:
        print(
            f"\nInversion terminated after {iteration_number} iterations with L2-norm ",
            f"={round(l2_norm, 2)} because maximum number of iterations ",
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
    apply_weights: bool = False,
    bounds: list = None,
):
    """
    perform a geometry inversion, where a topographic surface, represented by a layer of
    vertical, right-rectangular prisms, has it's topography updated based on a gravity
    anomaly.
    """
    time_start = time.perf_counter()

    gravity = copy.deepcopy(input_grav)
    prisms = copy.deepcopy(prism_layer)

    assert prisms.top.values.min() == prisms.bottom.values.max()
    assert prisms.density.values.max() == -prisms.density.values.min()

    density_contrast = prisms.density.values.max()
    zref = prisms.top.values.min()

    # add starting surface to dataset
    surface_grid = xr.where(prisms.density > 0, prisms.top, prisms.bottom)
    prisms["surface"] = surface_grid

    # turn dataset into dataframe
    prisms_df = prisms.to_dataframe().reset_index().dropna().astype(float)

    # get spacing of prisms
    if deriv_type == "annulus":
        prism_spacing = abs(
            prisms_df.northing.unique()[1] - prisms_df.northing.unique()[0]
        )
    else:
        prism_spacing = None

    # create empty jacobian matrix
    empty_jac = np.empty(
        (len(gravity[input_grav_column]), prisms.top.size),
        dtype=np.float64,
    )

    # if there is a confining surface (above or below), which the inverted layer
    # shouldn't intersect, then sample those layers into the df
    prisms_df = inv_utils.sample_bounding_surfaces(
        prisms_df,
        upper_confining_layer,
        lower_confining_layer,
    )

    # set starting delta L2 norm to positive infinity
    delta_l2_norm = np.Inf

    # iteration times
    iter_times = []

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
        initial_RMSE = inv_utils.RMSE(gravity[f"iter_{ITER}_initial_misfit"])
        l2_norm = np.sqrt(initial_RMSE)

        if ITER == 1:
            starting_l2_norm = l2_norm

        # calculate jacobian sensitivity matrix
        jac = jacobian(
            deriv_type,
            gravity,
            empty_jac,
            prisms_layer=prisms,
            prism_spacing=prism_spacing,
            prism_size=jacobian_prism_size,
            apply_weights=apply_weights,
        )

        # calculate correction for each prism
        Surface_correction = solver(
            jacobian=jac,
            residuals=gravity.res.values,
            weights=solver_weights,
            damping=solver_damping,
            solver_type=solver_type,
            bounds=bounds,
        )

        # print correction values
        print(
            f"Layer correction median: {int(np.median(Surface_correction))}",
            f"m, RMSE:{int(inv_utils.RMSE(Surface_correction))} m",
        )

        # constrain the surface correction values
        Surface_correction = inv_utils.constrain_surface_correction(
            prisms_df,
            Surface_correction,
            upper_confining_layer,
            lower_confining_layer,
            max_layer_change_per_iter,
        )

        # add corrections to prisms_df
        prisms_df = pd.concat(
            [prisms_df, pd.DataFrame({f"iter_{ITER}_correction": Surface_correction})],
            axis=1,
        )

        # constrain corrections to only within gravity region
        # prisms_df['inside'] = vd.inside(
        #     (prisms_df.easting, prisms_df.northing), region=inversion_region)
        # prisms_df.loc[prisms_df.inside == False, f"iter_{ITER}_correction"] = 0

        # apply the surface correction to the prisms dataframe and dataset
        prisms_df, prisms = apply_surface_correction(
            prisms_df,
            prisms,
            ITER,
            zref,
        )

        # update the forward gravity and the misfit
        gravity = update_gravity_and_misfit(
            gravity,
            prisms,
            input_grav_column,
            ITER,
        )

        # update the misfit RMSE
        updated_RMSE = inv_utils.RMSE(gravity[f"iter_{ITER}_final_misfit"])
        print(f"\nupdated misfit RMSE: {round(updated_RMSE, 2)}")

        # update the l2 and delta l2 norms
        l2_norm, delta_l2_norm = update_l2_norms(updated_RMSE, l2_norm)

        print(
            f"updated L2-norm: {round(l2_norm, 2)}, ", f"tolerance: {l2_norm_tolerance}"
        )
        print(
            f"updated delta L2-norm : {round(delta_l2_norm, 2)}, ",
            f"tolerance: {delta_l2_norm_tolerance}",
        )

        # end iteration timer
        iter_time_end = time.perf_counter()
        iter_times.append(iter_time_end - iter_time_start)

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

    elapsed_time = int(time_end - time_start)

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
        "solver_weights": "Not enabled" if solver_weights is None else "Enabled",
        "upper_confining_layer": "Not enabled"
        if upper_confining_layer is None
        else "Enabled",
        "lower_confining_layer": "Not enabled"
        if lower_confining_layer is None
        else "Enabled",
        "max_layer_change_per_iter": f"{max_layer_change_per_iter} m",
        "time_elapsed": f"{elapsed_time} seconds",
        "average_iteration_time": f"{round(np.mean(iter_times), 2)} seconds",
    }

    return prisms_df, gravity, params, elapsed_time


def inversion_RMSE(true_surface, constraints=None, plot=False, **kwargs):
    """
    Calculate the RMSE of the inversion results compared with the true starting
    topography.
    """

    # run inversion
    # with inv_utils.HiddenPrints():
    prism_results, grav_results, params, elapsed_time = geo_inversion(**kwargs)

    # grid resulting prisms
    ds = prism_results.set_index(["northing", "easting"]).to_xarray()

    # get last iteration's layer result
    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]
    final_surface = ds[cols[-1]]

    dif = true_surface - final_surface

    rmse = inv_utils.RMSE(dif)

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})
        df = profile.sample_grids(
            df=constraints,
            grid=final_surface,
            name="final_surface",
        )
        constraints_rmse = round(inv_utils.RMSE(df.upward - df.final_surface), 2)
    else:
        constraints_rmse = None

    if plot:
        _ = utils.grd_compare(
            true_surface,
            final_surface,
            grid1_name="True",
            grid2_name="Final",
            plot=True,
            points=constraints,
            # plot_type="xarray",
            # cmap="gist_earth",
            # robust=True,
            plot_type="pygmt",
            cmap="rain",
            reverse_cpt=True,
            hist=True,
            inset=False,
            points_style="c.15c",
            subplot_labels=True,
        )

    return rmse, prism_results, grav_results, params, elapsed_time, constraints_rmse
