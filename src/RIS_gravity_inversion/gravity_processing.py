from __future__ import annotations

import itertools
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pygmt
import scipy
import seaborn as sns
import shapely
import verde as vd
from polartoolkit import utils
from shapely.geometry import LineString, MultiPoint, Point
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from RIS_gravity_inversion import utils as inv_utils


def plot_levelling_convergence(
    results,
    mistie_prefix="mistie_trend",
    logy=False,
    title="Levelling convergence",
):
    sns.set_theme()

    # get mistie columns
    cols = [s for s in results.columns.to_list() if s.startswith(mistie_prefix)]

    iters = len(cols)
    mistie_rmses = [utils.rmse(results[i]) for i in cols]

    _fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title(title)
    ax1.plot(range(iters), mistie_rmses, "bo-")
    ax1.set_xlabel("Iteration")
    if logy:
        ax1.set_yscale("log")
    ax1.set_ylabel("Cross-over RMS (mGal)", color="k")
    ax1.tick_params(axis="y", colors="k", which="both")

    plt.tight_layout()


def distance_along_line(data, line_col_name="line", time_col_name="unixtime"):
    """
    return distances along each flight line in meters, assuming the lowest time value is
    the start of each lines.
    data: gpd.GeoDataFrame containing columns with names set by line_col_name and
    time_col_name.
    line_col_name: str, name of column containing line names.
    time_col_name: str, name of column containing time.
    """
    gdf = data.copy()

    distances = []
    for i in gdf[line_col_name].unique():
        gdf2 = gdf[gdf[line_col_name] == i]
        dist = gdf2.distance(gdf2.sort_values(by=time_col_name).geometry.iloc[0]).values
        distances.extend(dist)

    return distances


def create_intersection_table(
    data,
    line_col_name="line",
    exclude_ints=None,
    cutoff_dist=None,
    plot=True,
):
    """
    create a dataframe which contains the intersections between lines. Intersections are
    only included if the distance between the closest points on each line and the
    intersection point is within "cutoff_dist". The intersections are calculated by
    representing the point data as lines, and finding the hypothetical crossover.
    By default crossovers will only be between the first and last point of a line. If
    there is an expected crossover just beyond the end of a line which should be
    included, use the `buffer_dist` arg to extend the line representation of the data.

    data: gpd.GeoDataFrame, containing a column specifying the line names
    (line_col_name), and a "geometry" column
    line_col_name: str, name of the column containing the line names
    cutoff_dist: float, distance in meters that is the maximum between a data point and
    a intersection for the intersection to be included
    plot: bool, choose to plot the resulting intersection points.
    robust: bool, use a robust color range for the plot.
    """

    gdf = data.copy()

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in gdf.columns:
        rows_to_drop = gdf[gdf.is_intersection]
        gdf = gdf.drop(index=rows_to_drop.index)
    gdf = gdf.drop(columns="is_intersection", errors="ignore")

    # group by lines
    grouped = gdf.groupby([line_col_name], as_index=False)["geometry"]

    # from points to lines
    grouped = grouped.apply(lambda x: LineString(x.tolist()))

    # get intersection points
    inters = get_line_intersections(grouped.geometry)
    inters = gpd.GeoDataFrame(geometry=inters)

    # get nearest 2 lines to each intersection point
    # and nearest data point on each line to the intersection point
    line1_names = []
    line2_names = []
    line1_dists = []
    line2_dists = []
    for p in inters.geometry:
        # look into shapely.interpolate() to get points based on distance along line
        # look into shapely.project() to get distance along line1 which is closest point
        # to line2
        # shapely.crosses or shapely.intersects for if lines cross or not
        # shapely.nearest_points()

        # find nearest 2 lines to intersection point using LineString's
        grouped["dist"] = grouped.geometry.distance(p)
        nearest_lines = grouped.sort_values(by="dist")[[line_col_name]].iloc[0:2]
        nearest_lines = nearest_lines.sort_values(by=[line_col_name])

        # get line names
        line1 = nearest_lines.iloc[0][line_col_name]
        line2 = nearest_lines.iloc[1][line_col_name]

        # append names to nearest 2 lines to lists
        line1_names.append(line1)
        line2_names.append(line2)

        # get actually datapoints for each line (not LineString representation)
        line1_points = gdf[gdf[line_col_name] == line1]
        line2_points = gdf[gdf[line_col_name] == line2]

        # get nearest data point on each line to intersection point
        nearest_datapoint_line1 = (
            line1_points.geometry.distance(p).sort_values().iloc[0]
        )
        nearest_datapoint_line2 = (
            line2_points.geometry.distance(p).sort_values().iloc[0]
        )

        # add distance to nearest data point on each line to lists
        line1_dists.append(nearest_datapoint_line1)
        line2_dists.append(nearest_datapoint_line2)

    # add names and distances as columns
    inters["line1"] = line1_names
    inters["line2"] = line2_names
    inters["line1_dist"] = line1_dists
    inters["line2_dist"] = line2_dists

    inters["max_dist"] = inters[["line1_dist", "line2_dist"]].max(axis=1)

    # if intersection is not within cutoff_dist, remove rows
    if cutoff_dist is not None:
        prior_len = len(inters)
        inters = inters[inters.max_dist < cutoff_dist]
        print(
            f"removed {prior_len - len(inters)} intersections points which were",
            f"further than {int(cutoff_dist/1000)}km from nearest data point",
        )

    # get coords from geometry column
    inters["easting"] = inters.geometry.x
    inters["northing"] = inters.geometry.y

    if exclude_ints is not None:
        exclude_inds = []
        for i in exclude_ints:
            # if pair of lines numbers given, get those indices
            if len(i) == 2:
                ind = inters[
                    (inters.line1 == i[0]) & (inters.line2 == i[1])
                ].index.values
            # if single line number, get all intersections of that line
            elif len(i) == 1:
                ind = inters[
                    (inters.line1 == i[0]) | (inters.line2 == i[0])
                ].index.values
            exclude_inds.extend(ind)
        inters = inters.drop(index=exclude_inds)

    a = len(inters)
    # keep only the closest of duplicated intersections
    inters = (
        inters.sort_values(
            "max_dist",
            ascending=False,
        )
        .drop_duplicates(
            subset=["line1", "line2"],
            keep="last",
        )
        .sort_index()
    )
    b = len(inters)
    if a != b:
        print(f"Dropped {a-b} duplicate intersections")

    if plot is True:
        plotly_points(
            inters,
            color_col="max_dist",
            hover_cols=["line1", "line2", "max_dist", "line1_dist", "line2_dist"],
            robust=True,
            point_size=6,
            theme=None,
            cmap="greys",
        )

    return inters.drop(columns=["line1_dist", "line2_dist"])


def add_intersections(
    df,
    intersections,
    line_col_name="line",
):
    gdf = df.copy()
    inters = intersections.copy()

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in gdf.columns:
        rows_to_drop = gdf[gdf.is_intersection]
        gdf = gdf.drop(index=rows_to_drop.index)
    gdf = gdf.drop(columns="is_intersection", errors="ignore")

    prior_length = len(gdf)

    # add boolean column for whether point is an intersection
    gdf["is_intersection"] = False
    gdf["intersecting_line"] = ""

    # collect intersections to be added
    dfs = []
    for _, row in inters.iterrows():
        for i in list(gdf[line_col_name].unique()):
            if i in (row.line1, row.line2):
                df = pd.DataFrame(
                    {
                        line_col_name: [i],
                        "easting": row.geometry.x,
                        "northing": row.geometry.y,
                        "is_intersection": True,
                    }
                )
                if i == row.line1:
                    df["intersecting_line"] = row.line2
                else:
                    df["intersecting_line"] = row.line1
                df["geometry"] = gpd.points_from_xy(df.easting, df.northing)
                dfs.append(df)

    # add intersections
    gdf = pd.concat([gdf, *dfs])

    # check correct number of intersections were added
    assert len(gdf) == prior_length + (2 * len(inters))

    # sort by lines
    gdf = gdf.sort_values(by=line_col_name)

    # get distance along each line
    gdf["dist_along_line"] = distance_along_line(gdf, line_col_name=line_col_name)

    # sort by distance and reset index
    gdf = gdf.sort_values(by=[line_col_name, "dist_along_line"])
    gdf = gdf.reset_index(drop=True)

    # add dist along line to intersections dataframe
    # iterate through intersections
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line1_value = gdf[
            (gdf[line_col_name] == row.line1) & (gdf.intersecting_line == row.line2)
        ].dist_along_line.values[0]
        line2_value = gdf[
            (gdf[line_col_name] == row.line2) & (gdf.intersecting_line == row.line1)
        ].dist_along_line.values[0]

        inters.loc[ind, "dist_along_line1"] = line1_value
        inters.loc[ind, "dist_along_line2"] = line2_value

    # inters["dist_along_lines"]=inters.line1_value-inters.line2_value

    return gdf, inters


def extend_lines(
    gdf,
    max_interp_dist,
):
    """
    WIP attempt to extend lines to intersect nearby lines
    """
    grouped = gdf[(gdf.line == 1040) | (gdf.line == 20)].groupby(
        "line", as_index=False
    )["geometry"]
    gdf2 = gdf[(gdf.line == 1040) | (gdf.line == 20)]
    # grouped = grouped.apply(lambda x: LineString(x.tolist()))
    # lines = grouped.iloc[0:2].geometry.copy()

    # for name1, name2 in itertools.combinations(list(grouped.groups.keys()), 2):
    for name1, name2 in itertools.combinations(gdf2.line.unique(), 2):
        line1 = LineString(grouped.get_group(name1).tolist())
        line2 = LineString(grouped.get_group(name2).tolist())

        # get line endpoints
        # line1_endpoints = MultiPoint(
        #   [Point(list(line1.coords)[0]), Point(list(line1.coords)[-1])])
        # line2_endpoints = MultiPoint(
        #   [Point(list(line2.coords)[0]), Point(list(line2.coords)[-1])])
        line1_endpoints = MultiPoint([list(line1.coords)[0], list(line1.coords)[-1]])  # noqa: RUF015
        line2_endpoints = MultiPoint([list(line2.coords)[0], list(line2.coords)[-1]])  # noqa: RUF015

        # print(line1_endpoints)
        # print(line2_endpoints)

        # get nearest points on each line to the closest of the other lines endpoints
        nearest_line1_point_to_line2_endpoints = shapely.nearest_points(
            line1, line2_endpoints
        )[0]
        nearest_line2_point_to_line1_endpoints = shapely.nearest_points(
            line2, line1_endpoints
        )[0]

        # print(nearest_line2_point_to_line1_endpoints)
        # print(nearest_line1_point_to_line2_endpoints)

        # get distances between nearest points on line with closest endpoint of
        # other line
        distance_line2_endpoint_to_line1 = np.min(
            [
                x.distance(nearest_line1_point_to_line2_endpoints)
                for x in line2_endpoints
            ]
        )
        distance_line1_endpoint_to_line2 = np.min(
            [
                x.distance(nearest_line2_point_to_line1_endpoints)
                for x in line1_endpoints
            ]
        )

        # print(distance_line2_endpoint_to_line1)
        # print(distance_line1_endpoint_to_line2)

        # if distance is lower than cutoff, add intersection points to extend lines
        if distance_line1_endpoint_to_line2 <= max_interp_dist:
            line2_new = LineString(
                list(line2.coords) + list(nearest_line1_point_to_line2_endpoints.coords)
            )
            assert len(list(line2.coords)) + 1 == len(list(line2_new.coords))
            print(f"extended line: {name1}")
        else:
            line2_new = line2

        # repeat for line2
        if distance_line2_endpoint_to_line1 <= max_interp_dist:
            line1_new = LineString(
                list(line1.coords) + list(nearest_line2_point_to_line1_endpoints.coords)
            )
            assert len(list(line1.coords)) + 1 == len(list(line1_new.coords))
            print(f"extended line: {name2}")
        else:
            line1_new = line1

        # print(len(list(line1.coords)))
        # print(len(list(line2.coords)))
        # print(len(list(line1_new.coords)))
        # print(len(list(line2_new.coords)))


def get_line_intersections(
    lines,
):
    """
    adapted from https://gis.stackexchange.com/questions/137909/intersecting-lines-to-get-crossings-using-python-with-qgis
    """

    inters = []
    for line1, line2 in itertools.combinations(lines, 2):
        if line1.intersects(line2):
            inter = line1.intersection(line2)

            if inter.type == "Point":
                inters.append(inter)
            elif inter.type == "MultiPoint":
                inters.extend(list(inter.geoms))
            elif inter.type == "MultiLineString":
                multi_line = list(inter.geoms)
                first_coords = multi_line[0].coords[0]
                last_coords = multi_line[len(multi_line) - 1].coords[1]
                inters.append(Point(first_coords[0], first_coords[1]))
                inters.append(Point(last_coords[0], last_coords[1]))
            elif inter.type == "GeometryCollection":
                for geom in inter:
                    if geom.type == "Point":
                        inters.append(geom)
                    elif geom.type == "MultiPoint":
                        inters.extend(list(geom))
                    elif geom.type == "multi_lineString":
                        multi_line = list(geom)
                        first_coords = multi_line[0].coords[0]
                        last_coords = multi_line[len(multi_line) - 1].coords[1]
                        inters.append(Point(first_coords[0], first_coords[1]))
                        inters.append(Point(last_coords[0], last_coords[1]))
    return inters


def scipy_interp1d(
    df,
    to_interp=None,
    interp_on=None,
    method=None,
):
    """
    interpolate NaN's in "to_interp" column, based on values from "interp_on" column
    method:
        'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
        'previous', 'next'
    use kwargs to pass other arguments to scipy.interpolate.interp1d()
    """
    df1 = df.copy()

    # drop NaN's
    df_no_nans = df1.dropna(subset=[to_interp, interp_on], how="any")

    # define interpolation function
    f = scipy.interpolate.interp1d(
        df_no_nans[interp_on],
        df_no_nans[to_interp],
        kind=method,
    )

    # get interpolated values at points with NaN's
    values = f(df1[df1[to_interp].isnull()][interp_on])

    # fill NaN's  with values
    df1.loc[df1[to_interp].isnull(), to_interp] = values

    return df1


def verde_interp1d(
    df,
    to_interp=None,
    interp_on=None,
    method=None,
):
    """
    interpolate NaN's in "to_interp" column, based on coordinates from "interp_on"
    columns,
    method: vd.Spline(), vd.SplineCV(), vd.KNeighbors(), vd.Linear(), vd.Cubic()
    """
    df1 = df.copy()

    # drop NaN's
    df_no_nans = df1.dropna(subset=[to_interp, *interp_on], how="any")

    # fit interpolator to data
    method.fit(
        (df_no_nans[interp_on[0]], df_no_nans[interp_on[1]]), df_no_nans[to_interp]
    )

    # predict at NaN's
    values = method.predict(
        (
            df1[df1[to_interp].isnull()][interp_on[0]],
            df1[df1[to_interp].isnull()][interp_on[1]],
        ),
    )

    # fill NaN's  with values
    df1.loc[df1[to_interp].isnull(), to_interp] = values

    return df1


def interp1d_single_col(
    df,
    to_interp=None,
    interp_on=None,
    engine=None,
    method=None,
    plot=False,
    line_col="line",
    dist_col="dist_along_line",
):
    """
    interpolate NaN's in "to_interp" column, based on value(s) from "interp_on"
    column(s).
    engine: "verde" or "scipy"
    method:
        for "verde": vd.Spline(), vd.SplineCV(), vd.KNeighbors(), vd.Linear(),
        vd.Cubic()
        for "scipy": 'linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic',
        'cubic', 'previous', 'next'
    """
    args = {
        "df": df,
        "to_interp": to_interp,
        "interp_on": interp_on,
        "method": method,
    }

    if engine == "verde":
        filled = verde_interp1d(**args)
    elif engine == "scipy":
        filled = scipy_interp1d(**args)
    else:
        msg = "invalid string for engine type"
        raise ValueError(msg)

    if plot is True:
        plot_line_and_crosses(
            filled,
            line=filled[line_col].iloc[0],
            x=dist_col,
            y=[to_interp],
            y_axes=[i + 1 for i in range(len([to_interp]))],
        )

    return filled


def interp1d_windows_single_col(
    df,
    window_width=None,
    dist_col="dist_along_line",
    line_col="line",
    to_interp=None,
    plot_windows=False,
    plot_line=False,
    **kwargs,
):
    """
    Create a window of data either side of NaN's based on "dist_along_line" column and
    interpolate the value. Useful when NaN's are sparse, or lines are long. All kwargs
    are based to function "interp1d"
    """
    df1 = df.copy()

    # iterate through NaNs
    values = []
    for i in df1[df1[to_interp].isnull()].index:
        # get distance along line of NaN
        dist_at_nan = df[dist_col].loc[i]

        # try interpolation with set window width, if there's not enough data,
        # double the width
        win = window_width
        while True:
            try:
                # get data inside window
                llim, ulim = dist_at_nan - win, dist_at_nan + win
                df_inside = df1[df1[dist_col].between(llim, ulim)]

                # run interpolation with bounds_error=False
                # may be multiple NaN's within window (some outside of bounds)
                # but we only extract the fill value for loc[i]
                filled = interp1d(
                    df_inside, to_interp=[to_interp], **kwargs, bounds_error=True
                )
                # extract just the filled value
                value = filled[to_interp].loc[i]
                # save value to a list
                values.append(value)
            except ValueError as e:
                # error messages for too few points in window
                scipy_error = "cannot reshape array of"
                verde_error = "Found array with"
                # error message for bounds error
                above_error = "in x_new is above the interpolation range"
                below_error = "in x_new is below the interpolation range"
                if any(item in str(e) for item in [scipy_error, verde_error]):
                    win += win
                    print(
                        "too few points in window for intersection of lines",
                        f"{df.intersecting_line.loc[i]} & {df[line_col].loc[i]},"
                        f" doubling window size to {win/1000}km",
                    )
                elif any(item in str(e) for item in [above_error, below_error]):
                    win += win
                    print(
                        "bounds error for interpolation of intersection of lines",
                        f"{df.intersecting_line.loc[i]} & {df[line_col].loc[i]},"
                        f" doubling window size to {win/1000}km",
                    )
                else:  # raise other errors
                    raise e

                continue
            break

        if plot_windows is True:
            plot_line_and_crosses(
                filled,
                line=filled[line_col].iloc[0],
                x=dist_col,
                y=[to_interp],
                y_axes=[i + 1 for i in range(len([to_interp]))],
            )

    # add values into dataframe
    df1.loc[df1[to_interp].isnull(), to_interp] = values

    if plot_line is True:
        plot_line_and_crosses(
            df1,
            line=df1[line_col].iloc[0],
            x=dist_col,
            y=[to_interp],
            y_axes=[i + 1 for i in range(len([to_interp]))],
        )

    return df1


def interp1d_windows(
    df,
    to_interp=None,
    plot_line=False,
    line_col="line",
    dist_col="dist_along_line",
    **kwargs,
):
    if line_col is not None:
        assert (
            len(df[line_col].unique()) <= 1
        ), "Warning: provided more than 1 flight line"

    if isinstance(to_interp, str):
        to_interp = [to_interp]

    df1 = df.copy()

    # iterate through columns
    for col in to_interp:
        filled = interp1d_windows_single_col(
            df1,
            to_interp=col,
            line_col=line_col,
            **kwargs,
        )
        df1[col] = filled[col]

    if plot_line is True:
        plot_line_and_crosses(
            df1,
            line=df1[line_col].iloc[0],
            x=dist_col,
            y=to_interp,
            y_axes=[i + 1 for i in range(len(to_interp))],
        )

    return df1


def interp1d(
    df,
    to_interp=None,
    interp_on=None,
    engine=None,
    method=None,
    plot_line=False,
    line_col="line",
    dist_col="dist_along_line",
):
    """ """
    if line_col is not None:
        assert (
            len(df[line_col].unique()) <= 1
        ), "Warning: provided more than 1 flight line"

    if isinstance(to_interp, str):
        to_interp = [to_interp]

    df1 = df.copy()

    # iterate through columns
    for col in to_interp:
        filled = interp1d_single_col(
            df1,
            to_interp=col,
            interp_on=interp_on,
            engine=engine,
            method=method,
        )
        df1[col] = filled[col]

    if plot_line is True:
        plot_line_and_crosses(
            df1,
            line=df1[line_col].iloc[0],
            x=dist_col,
            y=to_interp,
            y_axes=[i + 1 for i in range(len(to_interp))],
        )

    return df1


def interp1d_all_lines(
    df,
    line_col="line",
    to_interp=None,
    interp_on=None,
    window_width=None,
    method=None,
    engine=None,
    plot=False,
    dist_col="dist_along_line",
    wait_for_input=False,
):
    df1 = df.copy()

    lines = df1.groupby(line_col)
    filled_lines = []
    for line, line_df in lines:
        if window_width is None:
            filled = interp1d(
                line_df,
                to_interp=to_interp,
                interp_on=interp_on,
                engine=engine,
                method=method,
            )
        else:
            filled = interp1d_windows(
                line_df,
                to_interp=to_interp,
                window_width=window_width,
                interp_on=interp_on,
                engine=engine,
                method=method,
            )
        filled_lines.append(filled)

        if plot is True:
            plot_line_and_crosses(
                filled,
                line=line,
                x=dist_col,
                y=to_interp,
                y_axes=[i + 1 for i in range(len(to_interp))],
                # plot_inters = [True, True],
                # marker_sizes=[2]
            )
        if wait_for_input is True:
            input("Press key to continue...")

    return pd.concat(filled_lines)


def calculate_misties(
    intersections,
    data,
    data_col=None,
    line_col="line",
    mistie_name="mistie",
    plot=False,
    robust=True,
):
    """
    Calculate mistie values for all intersections. Add value to intersections dataframe,
    and to rows of data dataframe which are intersection points.
    """
    inters = intersections.copy()
    df = data.copy()

    df[mistie_name] = np.nan
    # iterate through intersections
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line1_value = df[
            (df[line_col] == row.line1) & (df.intersecting_line == row.line2)
        ][data_col].values[0]
        line2_value = df[
            (df[line_col] == row.line2) & (df.intersecting_line == row.line1)
        ][data_col].values[0]

        assert line1_value != np.nan
        assert line2_value != np.nan

        inters.loc[ind, "line1_value"] = line1_value
        inters.loc[ind, "line2_value"] = line2_value

        # add misties to rows of data df which are intersection points
        conditions = (df[line_col] == row.line1) & (df.intersecting_line == row.line2)
        df.loc[conditions, mistie_name] = line1_value - line2_value

        conditions = (df[line_col] == row.line2) & (df.intersecting_line == row.line1)
        df.loc[conditions, mistie_name] = line1_value - line2_value

    # misties are defined as line1 - line2
    misties = inters.line1_value - inters.line2_value
    inters[mistie_name] = misties

    # df.loc[df.is_intersection==True, mistie_name] = misties

    # print(f"mistie RMSE: {utils.rmse(inters[mistie_name])}")
    if plot is True:
        plotly_points(
            inters,
            color_col=mistie_name,
            hover_cols=["line1", "line2", "line1_value", "line2_value"],
            robust=robust,
            point_size=5,
        )

    return inters, df


def verde_predict_trend(
    data_to_fit: pd.DataFrame,
    cols_to_fit: list,
    data_to_predict: pd.DataFrame,
    cols_to_predict: list,
    degree: int,
):
    """
    data_to_fit: pd.DataFrame with at least 3 columns: x, y, and data
    cols_to_fit: column names representing x, y, and data
    data_to_predict: pd.DataFrame with at least 2 columns: x, y
    cols_to_predict: column names representing x and y, and 3rd value for new column
        with predicted data
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # fit a polynomial trend through the lines mistie values
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Under-determined problem")
        # fit a trend to the data values
        fit_coords = (fit_df[cols_to_fit[0]], fit_df[cols_to_fit[1]])
        trend = vd.Trend(degree=degree).fit(fit_coords, fit_df[cols_to_fit[2]])

        # predict the trend on the new values
        predict_coords = (
            predict_df[cols_to_predict[0]],
            predict_df[cols_to_predict[1]],
        )
        predicted = trend.predict(predict_coords)
        predict_df[cols_to_predict[2]] = predicted

    return predict_df


def skl_predict_trend(
    data_to_fit: pd.DataFrame,
    cols_to_fit: list,
    data_to_predict: pd.DataFrame,
    cols_to_predict: list,
    degree: int,
):
    """
    data_to_fit: pd.DataFrame with at least 2 columns: distance, and data
    cols_to_fit: column names representing distance and data
    data_to_predict: pd.DataFrame with at least 1 columns: distance
    cols_to_predict: column names representing distance and new column
        with predicted data
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # # fit a polynomial trend through the lines mistie values
    polynomial_features = PolynomialFeatures(
        degree=degree,
        include_bias=True,
    )
    linear_regression = LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )

    pipeline.fit(
        fit_df[cols_to_fit[0]].to_numpy()[:, np.newaxis],
        fit_df[cols_to_fit[1]].to_numpy(),
    )

    predicted = pipeline.predict(
        predict_df[cols_to_predict[0]].to_numpy()[:, np.newaxis]
    )

    predict_df[cols_to_predict[1]] = predicted

    return predict_df


def level_lines(
    inters,
    data,
    lines_to_level=None,
    cols_to_fit=None,
    cols_to_predict=None,
    degree=None,
    levelled_col="levelled",
    data_col=None,
    mistie_col=None,
    new_mistie_col=None,
    line_col="line",
    plot=False,
):
    df = data.copy()

    # convert columns to fit on into a list if its a string
    if isinstance(cols_to_fit, str):
        cols_to_fit = [cols_to_fit]
    if isinstance(cols_to_predict, str):
        cols_to_predict = [cols_to_predict]

    df[levelled_col] = np.nan
    df["levelling_correction"] = np.nan

    # iterate through the chosen lines
    for line in lines_to_level:
        line_df = df[df[line_col] == line].copy()

        # get intersections of line of interest
        ints = inters[(inters.line1 == line) | (inters.line2 == line)]

        # fit a polynomial trend through the lines mistie values
        # if predicting on 2 variables (easting and northing) use verde
        if len(cols_to_fit) > 1:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Under-determined problem")
                try:
                    line_df = verde_predict_trend(
                        data_to_fit=ints,
                        cols_to_fit=cols_to_fit + [mistie_col],  # noqa: RUF005
                        data_to_predict=line_df,
                        cols_to_predict=cols_to_predict + ["levelling_correction"],  # noqa: RUF005
                        degree=degree,
                    )
                except ValueError as e:
                    if "zero-size array to reduction operation minimum which" in str(e):
                        print(f"Issue with line {line}, skipping")
                        # if issues, correction is 0
                        line_df["levelling_correction"] = 0
                    else:
                        raise e
        # if predicting on 1 variable (distance along line) use scikitlearn
        elif len(cols_to_fit) == 1:
            try:
                line_df = skl_predict_trend(
                    data_to_fit=ints,  # df with mistie values
                    cols_to_fit=cols_to_fit  # noqa: RUF005
                    + [mistie_col],  # column names for distance/mistie
                    data_to_predict=line_df,  # df with line data
                    cols_to_predict=cols_to_predict  # noqa: RUF005
                    + [
                        "levelling_correction"
                    ],  # column names for distance/ levelling correction
                    degree=degree,  # degree order for fitting line to misties
                )
            except ValueError as e:
                if "Found array with " in str(e):
                    print(f"Issue with line {line}, skipping")
                    # if issues, correction is 0
                    line_df["levelling_correction"] = 0
                else:
                    raise e

        # if levelling tie lines, negate the correction
        if cols_to_fit[0][-1] == "2":
            line_df["levelling_correction"] *= -1
        else:
            pass

        # remove the trend from the gravity
        values = line_df[data_col] - line_df.levelling_correction

        # update main df
        df.loc[df[line_col] == line, levelled_col] = values
        df.loc[df[line_col] == line, "levelling_correction"] = (
            line_df.levelling_correction
        )

    # add unchanged values for lines not included
    for line in df[line_col].unique():
        if line not in lines_to_level:
            df.loc[df[line_col] == line, levelled_col] = df.loc[
                df[line_col] == line, data_col
            ]

    # update mistie with levelled data
    # print(f"previous mistie RMSE: {utils.rmse(inters[mistie_col])}")
    inters_new, df = calculate_misties(
        inters,
        df,
        data_col=levelled_col,
        mistie_name=new_mistie_col,
        plot=False,
    )

    if plot is True:
        # plot old and new misties
        ints = inters_new[
            inters_new.line1.isin(lines_to_level)
            | inters_new.line2.isin(lines_to_level)
        ]
        plotly_points(
            ints,
            color_col=mistie_col,
            point_size=4,
            hover_cols=[
                "line1",
                "line2",
                "line1_value",
                "line2_value",
                mistie_col,
                new_mistie_col,
            ],
        )
        plotly_points(
            ints,
            color_col=new_mistie_col,
            point_size=4,
            hover_cols=[
                "line1",
                "line2",
                "line1_value",
                "line2_value",
                mistie_col,
                new_mistie_col,
            ],
        )

        plotly_points(
            df[df[line_col].isin(lines_to_level)],
            color_col="levelling_correction",
            point_size=2,
            hover_cols=[line_col, data_col, levelled_col],
        )

    return df, inters_new


def iterative_line_levelling(
    inters,
    data,
    flight_line_names,
    degree,
    starting_mistie_col,
    starting_data_col,
    iterations,
    cols_to_fit,
    mistie_prefix=None,
    levelled_data_prefix=None,
    plot_iterations=False,
    plot_results=False,
    plot_convergence=False,
    **kwargs,
):
    df = data.copy()
    ints = inters.copy()

    if mistie_prefix is None:
        mistie_prefix = f"mistie_trend{degree}"
    if levelled_data_prefix is None:
        levelled_data_prefix = f"levelled_data_trend{degree}"

    rmse = utils.rmse(ints[starting_mistie_col])
    print(f"Starting RMS mistie: {rmse} mGal")

    for i in range(1, iterations + 1):
        if i == 1:
            data_col = starting_data_col
            mistie_col = starting_mistie_col
        else:
            data_col = f"{levelled_data_prefix}_{i-1}"
            mistie_col = f"{mistie_prefix}_{i-1}"

        with inv_utils.HiddenPrints():
            df, ints = level_lines(
                ints,
                df,
                lines_to_level=flight_line_names,
                cols_to_fit=cols_to_fit,
                cols_to_predict="dist_along_line",
                degree=degree,
                data_col=data_col,
                levelled_col=f"{levelled_data_prefix}_{i}",
                mistie_col=mistie_col,
                new_mistie_col=f"{mistie_prefix}_{i}",
                line_col="line",
            )
        rmse = utils.rmse(ints[f"{mistie_prefix}_{i}"])
        print(f"RMS mistie after iteration {i}: {rmse} mGal")
        rmse_corr = utils.rmse(
            df[(df.line.isin(flight_line_names))].levelling_correction
        )
        print(f"RMS correction to lines: {rmse_corr} mGal")

        if plot_iterations is True:
            # plot flight lines
            plotly_points(
                df[df.line.isin(flight_line_names)],
                color_col="levelling_correction",
                point_size=2,
                hover_cols=[
                    "line",
                    f"{levelled_data_prefix}_{i}",
                ],
            )

    levelled_col = list(df.columns)[-2]

    df["levelling_correction"] = df[starting_data_col] - df[levelled_col]
    if plot_convergence is True:
        plot_levelling_convergence(
            df,
            mistie_prefix=mistie_prefix,
            logy=kwargs.get("logy", False),
        )
    if plot_results is True:
        # plot flight lines
        plotly_points(
            df[df.line.isin(flight_line_names)],
            color_col="levelling_correction",
            point_size=4,
            hover_cols=[
                "line",
                f"{levelled_data_prefix}_{i}",
            ],
        )

    return (
        df,
        ints,
    )


def iterative_levelling_alternate(
    inters,
    data,
    tie_line_names,
    flight_line_names,
    degree,
    starting_mistie_col,
    starting_data_col,
    iterations,
    mistie_prefix=None,
    levelled_data_prefix=None,
    plot_iterations=False,
    plot_results=False,
    plot_convergence=False,
    **kwargs,
):
    df = data.copy()
    ints = inters.copy()

    if mistie_prefix is None:
        mistie_prefix = f"mistie_trend{degree}"
    if levelled_data_prefix is None:
        levelled_data_prefix = f"levelled_data_trend{degree}"

    rmse = utils.rmse(ints[starting_mistie_col])
    print(f"Starting RMSE mistie: {rmse} mGal")

    for i in range(1, iterations + 1):
        if i == 1:
            data_col = starting_data_col
            mistie_col = starting_mistie_col
        else:
            data_col = f"{levelled_data_prefix}_{i-1}t"
            mistie_col = f"{mistie_prefix}_{i-1}t"

        # level lines to ties
        with inv_utils.HiddenPrints():
            df, ints = level_lines(
                ints,
                df,
                lines_to_level=flight_line_names,
                cols_to_fit="dist_along_line1",
                cols_to_predict="dist_along_line",
                degree=degree,
                data_col=data_col,
                levelled_col=f"{levelled_data_prefix}_{i}l",
                mistie_col=mistie_col,
                new_mistie_col=f"{mistie_prefix}_{i}l",
                line_col="line",
            )
        rmse = utils.rmse(ints[f"{mistie_prefix}_{i}l"])
        print(f"RMSE mistie after iteration {i}: L -> T: {rmse} mGal")
        rmse_corr = utils.rmse(
            df[(df.line.isin(flight_line_names))].levelling_correction
        )
        print(f"RMS correction to lines: {rmse_corr} mGal")

        # level ties to lines
        with inv_utils.HiddenPrints():
            df, ints = level_lines(
                ints,
                df,
                lines_to_level=tie_line_names,
                cols_to_fit="dist_along_line2",
                cols_to_predict="dist_along_line",
                degree=degree,
                data_col=f"{levelled_data_prefix}_{i}l",
                levelled_col=f"{levelled_data_prefix}_{i}t",
                mistie_col=f"{mistie_prefix}_{i}l",
                new_mistie_col=f"{mistie_prefix}_{i}t",
                line_col="line",
            )
        rmse = utils.rmse(ints[f"{mistie_prefix}_{i}t"])
        print(f"RMSE mistie after iteration {i}: T -> L: {rmse} mGal")
        rmse_corr = utils.rmse(df[(df.line.isin(tie_line_names))].levelling_correction)
        print(f"RMS correction to ties: {rmse_corr} mGal")

        if plot_iterations is True:
            # plot flight lines
            plotly_points(
                df[df.line.isin(flight_line_names)],
                color_col="levelling_correction",
                point_size=2,
                hover_cols=[
                    "line",
                    f"{levelled_data_prefix}_{i}l",
                ],
            )
            # plot tie lines
            plotly_points(
                df[df.line.isin(tie_line_names)],
                color_col="levelling_correction",
                point_size=5,
                hover_cols=[
                    "line",
                    f"{levelled_data_prefix}_{i}t",
                ],
            )

    levelled_col = list(df.columns)[-2]
    df["levelling_correction"] = df[starting_data_col] - df[levelled_col]

    if plot_convergence is True:
        plot_levelling_convergence(
            df,
            mistie_prefix=mistie_prefix,
            logy=kwargs.get("logy", False),
        )

    if plot_results is True:
        # plot flight lines
        plotly_points(
            df[df.line.isin(flight_line_names)],
            color_col="levelling_correction",
            point_size=2,
            hover_cols=[
                "line",
                f"{levelled_data_prefix}_{i}l",
            ],
        )
        # plot tie lines
        plotly_points(
            df[df.line.isin(tie_line_names)],
            color_col="levelling_correction",
            point_size=4,
            hover_cols=[
                "line",
                f"{levelled_data_prefix}_{i}t",
            ],
        )

    return df, ints


def plotly_points(
    df,
    coord_names=None,
    color_col=None,
    hover_cols=None,
    point_size=4,
    cmap=None,
    cmap_middle=None,
    robust=True,
    theme="plotly_dark",
):
    """
    Create a scatterplot of spatial data. By default, coordinates are extracted from
    geopandas geometry column, or from user specified columns given by 'coord_names'.
    """
    data = df[df[color_col].notna()].copy()

    if coord_names is None:
        try:
            x = data.geometry.x
            y = data.geometry.y
        except AttributeError:
            try:
                x = data["easting"]
                y = data["northing"]
            except AttributeError:
                try:
                    x = data["x"]
                    y = data["y"]
                except AttributeError:
                    pass
        coord_names = (x, y)

    vmin, vmax = utils.get_min_max(data[color_col], robust=robust)
    lims = (vmin, vmax)

    if cmap is None:
        if (lims[0] < 0) and (lims[1] > 0):
            cmap = "RdBu_r"
            cmap_middle = 0
        else:
            cmap = None
            cmap_middle = None
    else:
        cmap_middle = None

    if cmap_middle == 0:
        max_abs = vd.maxabs((vmin, vmax))
        lims = (-max_abs, max_abs)

    fig = px.scatter(
        data,
        x=coord_names[0],
        y=coord_names[1],
        color=data[color_col],
        color_continuous_scale=cmap,
        color_continuous_midpoint=cmap_middle,
        range_color=lims,
        hover_data=hover_cols,
        template=theme,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )

    fig.update_traces(marker={"size": point_size})

    fig.show()


def plotly_profiles(
    data,
    x="dist_along_line",
    y=("FAG_levelled"),
    y_axes=None,
    xlims=None,
    ylims=None,
    **kwargs,
):
    """
    plot data profiles with plotly
    currently only allows 3 separate y axes, set with "y_axes", starting with 1
    """
    df = data.copy()

    # turn y column name into list
    if isinstance(y, str):
        y = [y]

    # list of y axes to use, if none, all will be same
    y_axes = ["" for _ in y] if y_axes is None else [str(x) for x in y_axes]
    assert "0" not in y_axes, "No '0' or 0 allowed, axes start with 1"
    # convert y axes to plotly expected format: "y", "y2", "y3" ...
    y_axes = [s.replace("1", "") for s in y_axes]
    y_axes = [f"y{x}" for x in y_axes]

    # lim x and y ranges
    if xlims is not None:
        df = df[df[x].between(*xlims)]
    if ylims is not None:
        df = df[df[y].between(*ylims)]

    # set plotting mode
    modes = kwargs.get("modes")
    if modes is None:
        modes = ["markers" for _ in y]

    # set marker properties
    marker_sizes = kwargs.get("marker_sizes")
    marker_symbols = kwargs.get("marker_symbols")
    if marker_sizes is None:
        marker_sizes = [2 for _ in y]
    if marker_symbols is None:
        marker_symbols = ["circle" for _ in y]

    fig = go.Figure()

    # iterate through data columns
    for i, col in enumerate(y):
        fig.add_trace(
            go.Scatter(
                mode=modes[i],
                x=df[x],
                y=df[col],
                name=col,
                marker_size=marker_sizes[i],
                marker_symbol=marker_symbols[i],
                yaxis=y_axes[i],
            )
        )

    unique_axes = len(pd.Series(y_axes).unique())

    if unique_axes >= 1:
        y_axes_args = dict(yaxis=dict(title=y[y_axes.index("y")]))
        x_domain = [0, 1]
    if unique_axes >= 2:
        y_axes_args["yaxis2"] = dict(
            title=y[y_axes.index("y2")], overlaying="y", side="right"
        )
        x_domain = [0, 1]
    if unique_axes >= 3:
        y_axes_args["yaxis3"] = dict(
            title=y[y_axes.index("y3")],
            anchor="free",
            overlaying="y",
        )
        x_domain = [0.15, 1]
    else:
        pass

    fig.update_layout(
        title_text=kwargs.get("title"),
        xaxis=dict(
            title=x,
            domain=x_domain,
        ),
        **y_axes_args,
    )

    return fig


def plot_line_and_crosses(
    df,
    line=None,
    line_col_name="line",
    x="dist_along_line",
    y=("FAG_levelled"),
    plot_inters=None,
    y_axes=None,
    xlims=None,
    ylims=None,
    plot_type="plotly",
    **kwargs,
):
    """
    plot lines and crosses
    """

    # turn y column name into list
    if isinstance(y, str):
        y = [y]

    # list of y axes to use, if none, all will be same
    if y_axes is None:
        y_axes = ["1" for _ in y]

    if len(df[line_col_name].unique()) <= 1:
        line = df[line_col_name].iloc[0]

    df = df[df[line_col_name] == line]

    if xlims is not None:
        df = df[df[x].between(*xlims)]
    if ylims is not None:
        df = df[df[x].between(*ylims)]

    # list of which dataset to plot intersections for

    if plot_inters is None:
        plot_inters = []
        for i in y:
            if len(df[i].dropna()) == len(df[df.is_intersection]):
                plot_inters.append(False)
            else:
                plot_inters.append(True)
    else:
        pass

    if plot_type == "plotly":
        fig = plotly_profiles(
            df,
            x=x,
            y=y,
            y_axes=y_axes,
            title=f"Line: {df.line.iloc[0]}",
            **kwargs,
        )
        # convert numbers to strings
        y_axes = [str(x) for x in y_axes]
        assert "0" not in y_axes, "No '0' or 0 allowed, axes start with 1"
        # convert y axes to plotly expected format: "y", "y2", "y3" ...
        y_axes = [s.replace("1", "") for s in y_axes]
        y_axes = [f"y{x}" for x in y_axes]

        if plot_inters is not False:
            for i, z in enumerate(y):
                j = 0
                if plot_inters[i] is True:
                    text = df[df.is_intersection].intersecting_line if j == 0 else None
                    j += 1
                    fig.add_trace(
                        go.Scatter(
                            mode="markers+text",
                            x=df[df.is_intersection][x],
                            y=df[df.is_intersection][z],
                            yaxis=y_axes[i],
                            marker_size=5,
                            marker_symbol="diamond",
                            name="intersections",
                            text=text,
                            textposition="top center",
                        ),
                    )
                else:
                    pass

        fig.show()

    elif plot_type == "mpl":
        # plt.figure(dpi=200)
        fig, ax1 = plt.subplots(figsize=(9, 6))
        plt.grid()

        for a, j in enumerate(y):
            if a > 0:
                ax2 = ax1.twinx()
                axis = ax2
                color = kwargs.get("point_color", "orangered")
            else:
                axis = ax1
                color = kwargs.get("point_color", "mediumslateblue")

            axis.plot(
                df[x],
                df[j],
                linewidth=0.5,
                color=color,
                marker=".",
                markersize=kwargs.get("point_size", 0.1),
                label=j,
            )

            axis.scatter(
                x=df[df.is_intersection][x],
                y=df[df.is_intersection][j],
                s=kwargs.get("point_size", 20),
                c=kwargs.get("point_color", "r"),
                marker="x",
                zorder=2,
            )
            for i, txt in enumerate(df[df.is_intersection].intersecting_line):
                axis.text(
                    df[df.is_intersection][x].values[i],
                    df[df.is_intersection][j].values[i],
                    s=str(txt),
                    fontsize="x-small",
                )
            axis.set_ylabel(j)

        if len(y) > 1:
            h1, l1 = ax1.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1.legend(h1 + h2, l1 + l2)
        else:
            plt.legend()
        ax1.set_xlabel(x)

        plt.title(f"Line number: {line}")
        plt.show()


def plot_flightlines(
    fig: pygmt.Figure,
    df: pd.DataFrame,
    direction: str = "EW",
    plot_labels: bool = True,
    plot_lines: bool = True,
    **kwargs,
):
    # group lines by their line number
    lines = [v for _, v in df.groupby("line")]

    # plot lines
    if plot_lines is True:
        for i in list(range(len(lines))):
            fig.plot(
                x=lines[i].easting,
                y=lines[i].northing,
                pen=kwargs.get("pen", "0.3p,white"),
            )

    # plot labels
    if plot_labels is True:
        for i in list(range(len(lines))):
            # switch label locations for every other line
            if (i % 2) == 0:
                if direction == "EW":
                    offset = "0.25c/0c"
                    # plot label at max value of x-coord
                    x_or_y = "easting"
                    # angle of label
                    angle = 0
                elif direction == "NS":
                    offset = "0c/0.25c"
                    # plot label at max value of y-coord
                    x_or_y = "northing"
                    # angle of label
                    angle = 90
                else:
                    msg = "invalid direction string"
                    raise ValueError(msg)
                # plot label
                fig.text(
                    x=lines[i].easting.loc[lines[i][x_or_y].idxmax()],
                    y=lines[i].northing.loc[lines[i][x_or_y].idxmax()],
                    text=str(int(lines[i].line.iloc[0])),
                    justify="CM",
                    font=kwargs.get("font", "5p,black"),
                    fill="white",
                    offset=offset,
                    angle=angle,
                )
            else:
                if direction == "EW":
                    offset = "-0.25c/0c"
                    # plot label at max value of x-coord
                    x_or_y = "easting"
                    # angle of label
                    angle = 0
                elif direction == "NS":
                    offset = "0c/-0.25c"
                    # plot label at max value of y-coord
                    x_or_y = "northing"
                    # angle of label
                    angle = 90
                else:
                    msg = "invalid direction string"
                    raise ValueError(msg)
                # plot label
                fig.text(
                    x=lines[i].easting.loc[lines[i][x_or_y].idxmin()],
                    y=lines[i].northing.loc[lines[i][x_or_y].idxmin()],
                    text=str(int(lines[i].line.iloc[0])),
                    justify="CM",
                    font=kwargs.get("font", "5p,black"),
                    fill="white",
                    offset=offset,
                    angle=angle,
                )
