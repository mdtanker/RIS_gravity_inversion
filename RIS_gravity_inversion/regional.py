import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from antarctic_plots import profile

import RIS_gravity_inversion.inversion as inv
import RIS_gravity_inversion.utils as inv_utils


def regional_trend(
    trend: int,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    fill_method: str = "verde",
):
    """
    seperate the regional field with a trend
    """
    # get coordinate names
    original_dims = misfit_grid.dims

    misfit_filled = inv_utils.nearest_grid_fill(misfit_grid, method=fill_method)

    df = vd.grid_to_table(misfit_filled).astype("float64")
    trend = vd.Trend(degree=trend).fit(
        (df[original_dims[1]], df[original_dims[0]].values),
        df[misfit_filled.name],
    )
    anomalies["reg"] = trend.predict(
        (anomalies[original_dims[1]], anomalies[original_dims[0]])
    )

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
    # get coordinate names
    original_dims = misfit_grid.dims

    # filter the observed-forward misfit with the provided filter in meters
    regional_misfit = inv_utils.filter_grid(
        misfit_grid,
        float(filter_width[1:]),
        filt_type="lowpass",
    )

    # sample the results and merge into the anomalies dataframe
    tmp_regrid = pygmt.grdtrack(
        points=anomalies[[original_dims[1], original_dims[0]]],
        grid=regional_misfit,
        newcolname="reg",
        verbose="q",
    )
    anomalies = anomalies.merge(
        tmp_regrid, on=[original_dims[1], original_dims[0]], how="left"
    )

    return anomalies


def regional_constraints(
    constraint_points: pd.DataFrame,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    region: list,
    spacing: float,
    tension_factor: float = 1,
    registration: str = "g",
    constraint_block_size=None,
    grid_method="pygmt",
    dampings=None,
    delayed=False,
    constraint_weights_col=None,
):
    """
    seperate the regional field by sampling and regridding at the constraint points
    """
    # get coordinate names
    original_dims = misfit_grid.dims

    constraints_df = constraint_points.copy()

    # sample misfit at constraint points
    constraints_df = profile.sample_grids(
        df=constraints_df,
        grid=misfit_grid,
        name="misfit",
        coord_names=(original_dims[1], original_dims[0]),
        no_skip=True,
        verbose="q",
    )

    constraints_df = constraints_df[constraints_df.misfit.notna()]

    if constraint_block_size is not None:
        # get median misfit of constraint points in each cell
        constraints_df = pygmt.blockmedian(
            data=constraints_df[[original_dims[1], original_dims[0], "misfit"]],
            spacing=constraint_block_size,
            region=region,
            registration=registration,
        )

    # grid the entire region misfit based just on the misfit at the constraints
    if grid_method == "pygmt":
        regional_misfit = pygmt.surface(
            data=constraints_df[[original_dims[1], original_dims[0], "misfit"]],
            region=region,
            spacing=spacing,
            registration=registration,
            T=tension_factor,
            verbose="q",
        )
    elif grid_method == "verde":
        if dampings is None:
            dampings = list(np.logspace(-10, -2, num=9))
            dampings.append(None)

        spline = inv_utils.best_SplineCV(
            coordinates=(
                constraints_df[original_dims[1]],
                constraints_df[original_dims[0]],
            ),
            data=constraints_df.misfit,
            weights=constraints_df[constraint_weights_col],
            dampings=dampings,
            delayed=delayed,
        )

        regional_misfit = spline.grid(region=region, spacing=spacing).scalars

    elif grid_method == "eq_sources":
        pass
        # coords = (
        #     constraints_df[original_dims[1]],
        #     constraints_df[original_dims[0]],
        #     constraints_df.upward,
        # )
        #         dampings = [.001, .01, .05, .1]
        #         depths = [10e3, 30e3, 60e3, 90e3]
        #         parameter_sets = [
        #             dict(damping=combo[0], depth=combo[1])
        #             for combo in itertools.product(dampings, depths)
        #         ]
        #         eqs_best, regional_misfit, reg_df = inv_utils.eq_sources_best(
        #             parameter_sets = parameter_sets,
        #             coordinates = coords,
        #             data = constraints_df.misfit,
        #             region = region,
        #             spacing = spacing,
        #         )
        # study_df, eqs = inv_utils.optimize_eq_source_params(
        #     coords,
        #     constraints_df.misfit,
        #     n_trials=20,
        #     damping_limits=[0.0005, 0.01],
        #     depth_limits=[60e3, 150e3],
        #     plot=False,
        #     parallel=True,
        # )
        # # Define grid coordinates
        # grid_coords = vd.grid_coordinates(
        #     region=region,
        #     spacing=spacing,
        #     extra_coords=coords[2].max(),
        # )
        # # predict sources onto grid to get regional
        # regional_misfit = eqs.grid(grid_coords, data_names="pred").pred
    else:
        raise ValueError("invalid string for grid_method")

    # sample the resulting grid and add to anomalies dataframe
    anomalies = profile.sample_grids(
        df=anomalies,
        grid=regional_misfit,
        name="reg",
        coord_names=(original_dims[1], original_dims[0]),
        verbose="q",
    )

    return anomalies


def regional_eq_sources(
    source_depth: float,
    anomalies: pd.DataFrame,
    eq_damping: float = None,
    block_size: float = None,
    depth_type: str = "relative",
    input_misfit_name: str = "misfit",
    input_coord_names: list = ["easting", "northing"],
):
    """
    seperate the regional field by estimating deep equivalent sources

    eq_damping : float: smoothness to impose on estimated coefficients
    block_size : float: block reduce the data to speed up
    depth_type : str: constant depths, not relative to observation heights
    """

    df = anomalies[anomalies[input_misfit_name].notna()]
    # create set of deep sources
    equivalent_sources = hm.EquivalentSources(
        depth=source_depth,
        damping=eq_damping,
        block_size=block_size,
        depth_type=depth_type,
    )

    # fit the source coefficients to the data
    coordinates = (df[input_coord_names[0]], df[input_coord_names[1]], df.upward)
    equivalent_sources.fit(coordinates, df[input_misfit_name])

    # use sources to predict the regional field at the observation points
    df["reg"] = equivalent_sources.predict(coordinates)

    return df


def regional_seperation(
    input_grav: pd.DataFrame,
    grav_spacing: int,
    regional_method: str,
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
    tension_factor = kwargs.get("tension_factor", None)
    eq_sources = kwargs.get("eq_sources", None)

    df = kwargs.get("constraints", None)
    if df is not None:
        constraints = df.copy()
    else:
        constraints = None

    anomalies = input_grav[input_grav[input_grav_column].notna()]

    # if anomalies already calculated, drop the columns
    try:
        anomalies.drop(columns=["misfit", "reg", "res"], inplace=True)
    except KeyError:
        pass

    # calculate misfit
    anomalies = inv.misfit(
        input_grav=anomalies,
        input_forward_column=input_forward_column,
        input_grav_column=input_grav_column,
        constraints=constraints,
    )

    # grid misfit
    misfit_grid = anomalies.set_index(["northing", "easting"]).to_xarray().misfit
    # misfit_grid = pygmt.xyz2grd(
    #     data=anomalies[["easting", "northing", "misfit"]],
    #     region=inversion_region,
    #     spacing=grav_spacing,
    #     registration=registration,
    # )

    if regional_method == "trend":
        anomalies = regional_trend(
            trend,
            misfit_grid,
            anomalies,
            fill_method=kwargs.get("fill_method", "pygmt"),
        )
    elif regional_method == "filter":
        anomalies = regional_filter(
            filter,
            misfit_grid,
            anomalies,
            registration=registration,
        )
    elif regional_method == "constraints":
        anomalies = regional_constraints(
            constraint_points=constraints,
            misfit_grid=misfit_grid,
            anomalies=anomalies,
            region=inversion_region,
            spacing=grav_spacing,
            tension_factor=tension_factor,
            registration=registration,
            constraint_block_size=kwargs.get("constraint_block_size"),
            grid_method=kwargs.get("grid_method", "pygmt"),
            dampings=kwargs.get("dampings", None),
            delayed=kwargs.get("delayed", False),
            constraint_weights_col=kwargs.get("constraint_weights_col", None),
        )
    elif regional_method == "eq_sources":
        anomalies = regional_eq_sources(
            source_depth=eq_sources,
            anomalies=anomalies,
            eq_damping=kwargs.get("eq_damping", None),
            block_size=kwargs.get("block_size", None),
            depth_type=kwargs.get("depth_type", "relative"),
            input_misfit_name="misfit",
        )
    else:
        raise ValueError("invalid string for regional_method")

    # calculate the residual field
    anomalies["res"] = anomalies.misfit - anomalies.reg

    # mask regional based on residual
    anomalies["reg"] = anomalies.reg.mask(anomalies.res.isnull())

    return anomalies
