import harmonica as hm
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from antarctic_plots import profile

import RIS_gravity_inversion.inversion as inv


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
    constraint_points: pd.DataFrame,
    misfit_grid: xr.DataArray,
    anomalies: pd.DataFrame,
    region: list,
    spacing: float,
    tension_factor: float = 1,
    registration: str = "g",
    block_reduce=False,
    grid_method="pygmt",
    dampings = np.logspace(-6, 3, num=10),
    mindists = np.linspace(1e3, 100e3, 10),
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
        coord_names=("easting", "northing"),
    )

    if block_reduce is True:
        # get median misfit of constraint points in each 1km cell
        constraints_df = pygmt.blockmedian(
            data=constraints_df[["easting", "northing", "misfit"]],
            spacing=spacing,
            region=region,
            registration=registration,
        )

    # grid the entire region misfit based just on the misfit at the constraints
    if grid_method == "pygmt":
        regional_misfit = pygmt.surface(
            data=constraints_df[["easting", "northing", "misfit"]],
            region=region,
            spacing=spacing,
            registration=registration,
            T=tension_factor,
            verbose="q",
        )
    elif grid_method == "verde":
        spline = vd.SplineCV(
            dampings = dampings,
            mindists = mindists,
            delayed=True,
        )
        spline.fit(
            (constraints_df.easting, constraints_df.northing),
            constraints_df.misfit,
        )
        regional_misfit = spline.grid(region=region, spacing=spacing).scalars
    else:
        raise ValueError("invalid string for grid_method")

    # sample the resulting grid and add to anomalies dataframe
    anomalies = profile.sample_grids(
        df=anomalies,
        grid=regional_misfit,
        name="reg",
        coord_names=("easting", "northing"),
    )

    return anomalies


def regional_eq_sources(
    source_depth: float,
    anomalies: pd.DataFrame,
    eq_damping: float = None,
    block_size: float = None,
    depth_type: str = "relative",
    input_misfit_name: str = "misfit",
):
    """
    seperate the regional field by estimating deep equivalent sources

    eq_damping : float: smoothness to impose on estimated coefficients
    block_size : float: block reduce the data to speed up
    depth_type : str: constant depths, not relative to observation heights
    """
    # create set of deep sources
    equivalent_sources = hm.EquivalentSources(
        depth=source_depth,
        damping=eq_damping,
        block_size=block_size,
        depth_type=depth_type,
    )

    # fit the source coefficients to the data
    coordinates = (anomalies.easting, anomalies.northing, anomalies.upward)
    equivalent_sources.fit(coordinates, anomalies[input_misfit_name])

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
    anomalies = inv.misfit(
        input_grav=anomalies,
        input_forward_column=input_forward_column,
        input_grav_column=input_grav_column,
    )

    # grid misfit
    misfit_grid = pygmt.xyz2grd(
        data=anomalies[["easting", "northing", "misfit"]],
        region=inversion_region,
        spacing=grav_spacing,
        registration=registration,
    )

    if regional_method == "trend":
        anomalies = regional_trend(
            trend,
            misfit_grid,
            anomalies,
            fill_method=kwargs.get("fill_method", "rioxarray"),
            crs=crs,
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
            constraints,
            tension_factor,
            misfit_grid,
            anomalies,
            inversion_region,
            grav_spacing,
            registration=registration,
            block_reduce=False,
        )
    elif regional_method == "eq_sources":
        anomalies = regional_eq_sources(
            eq_sources,
            misfit_grid,
            anomalies,
            eq_damping=kwargs.get("eq_damping", None),
            block_size=kwargs.get("block_size", None),
            depth_type=kwargs.get("depth_type", "relative"),
            input_misfit_name = "misfit",
        )
    else:
        raise ValueError("invalid string for regional_method")

    # calculate the residual field
    anomalies["res"] = anomalies.misfit - anomalies.reg

    return anomalies
