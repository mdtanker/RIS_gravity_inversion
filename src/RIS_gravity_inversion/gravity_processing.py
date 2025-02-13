from __future__ import annotations

import itertools
import logging
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
from invert4geom import utils as invert4geom_utils
from polartoolkit import utils
from shapely.geometry import LineString, MultiPoint, Point
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm.autonotebook import tqdm

log = logging.getLogger(__name__)

log.addHandler(logging.NullHandler())


def plot_levelling_convergence(
    results,
    logy=False,
    title="Levelling convergence",
):
    sns.set_theme()

    # get mistie columns
    cols = [s for s in results.columns.to_list() if s.startswith("mistie")]

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


def distance_along_line(
    data: gpd.GeoDataFrame,
    line_col_name: str = "line",
    time_col_name: str = "unixtime",
) -> pd.Series:
    """
    Calculate the distances along each flight line in meters, assuming the lowest time
    value is the start of each lines. If you don't have time information, you can pass
    the index of the dataframe as the time column.

    Parameters
    ----------
    data : gpd.GeoDataFrame
        Dataframe containing the data points to calculate the distance along each line,
        must have a set geometry column.
    line_col_name : str, optional
        Column name specifying the line number, by default "line"
    time_col_name : str, optional
        Column name containing time in seconds for each datapoint, by default "unixtime"

    Returns
    -------
    pd.Series
        The distance along each line in meters
    """

    gdf = data.copy()

    gdf["dist_along_line"] = np.nan
    for i in gdf[line_col_name].unique():
        line = gdf[gdf[line_col_name] == i]
        dist = line.distance(line.sort_values(by=time_col_name).geometry.iloc[0]).values
        gdf.loc[gdf[line_col_name] == i, "dist_along_line"] = dist

    return gdf.dist_along_line


def create_intersection_table(
    flight_lines: gpd.GeoDataFrame,
    tie_lines: gpd.GeoDataFrame,
    line_col_name: str = "line",
    exclude_ints: list[list[int]] | None = None,
    cutoff_dist: float | None = None,
    plot: bool = True,
) -> gpd.GeoDataFrame:
    """
    TODO: add buffer dist option to extend end of lines for expected intersections

    create a dataframe which contains the intersections between provide flight and tie
    lines. For each intersection point, find the distance to the closest data point of
    each line. If the further of these two distances is greater than "cutoff_dist", the
    intersection is excluded. The intersections are calculated by
    representing the point data as lines, and finding the hypothetical crossover.
    By default crossovers will only be between the first and last point of a line. If
    there is an expected crossover just beyond the end of a line which should be
    included, use the `buffer_dist` arg to extend the line representation of the data.

    Parameters
    ----------
    flight_lines : gpd.GeoDataFrame
        Flight line data which must be a geodataframe with a registered geometry column
        and a column set by `line_col_name` specifying the line number.
    tie_lines : gpd.GeoDataFrame
        Tie line data which must be a geodataframe with a registered geometry column
        and a column set by `line_col_name` specifying the line number.
    line_col_name : str, optional
        Column name specifying the line numbers, by default "line"
    exclude_ints : list[list[int]] | None, optional
        List of lists where each sublist is either a single line number to exclude from
        all intersections, or a pair of line numbers specifying specific intersections
        to exclude, by default None
    cutoff_dist : float, optional
        The maximum allowed distance from a theoretical intersection to the further of
        nearest data point of each intersecting line, by default None
    plot : bool, optional
        Plot a map of the resulting intersection points colored by distance to the
        further of the two nearest data points, by default True

    Returns
    -------
    gpd.GeoDataFrame
        An intersection table containing the locations of the theoretical intersections,
        the line and tie numbers, and the distance to the further of the two nearest
        datapoints of each line, and a geometry column.
    """

    lines_df = flight_lines.copy()
    ties_df = tie_lines.copy()

    # if is_intersection column exists, delete it and rows where it's true
    if "is_intersection" in lines_df.columns:
        rows_to_drop = lines_df[lines_df.is_intersection]
        lines_df = lines_df.drop(index=rows_to_drop.index)
    lines_df = lines_df.drop(columns="is_intersection", errors="ignore")
    if "is_intersection" in ties_df.columns:
        rows_to_drop = ties_df[ties_df.is_intersection]
        ties_df = ties_df.drop(index=rows_to_drop.index)
    ties_df = ties_df.drop(columns="is_intersection", errors="ignore")

    # group by lines/ties
    grouped_lines = lines_df.groupby([line_col_name], as_index=False)["geometry"]
    grouped_ties = ties_df.groupby([line_col_name], as_index=False)["geometry"]

    # from points to lines
    grouped_lines = grouped_lines.apply(lambda x: LineString(x.tolist()))
    grouped_ties = grouped_ties.apply(lambda x: LineString(x.tolist()))

    # get intersection points
    inters = get_line_tie_intersections(
        lines=grouped_lines.geometry,
        ties=grouped_ties.geometry,
    )
    inters = gpd.GeoDataFrame(geometry=inters)

    # get nearest 2 lines to each intersection point
    # and nearest data point on each line to the intersection point
    line_names = []
    tie_names = []
    line_dists = []
    tie_dists = []
    log.info("total number of intersections found: %s", len(inters))
    for _i, p in enumerate(inters.geometry):
        # look into shapely.interpolate() to get points based on distance along line
        # look into shapely.project() to get distance along line which is closest point
        # to tie
        # shapely.crosses or shapely.intersects for if lines cross or not
        # shapely.nearest_points()

        # find nearest line/tie to intersection point using LineString's
        grouped_lines["dist"] = grouped_lines.geometry.distance(p)
        grouped_ties["dist"] = grouped_ties.geometry.distance(p)
        nearest_line = grouped_lines.sort_values(by="dist")[[line_col_name]].iloc[0]
        nearest_tie = grouped_ties.sort_values(by="dist")[[line_col_name]].iloc[0]

        # get line/tie names
        line = nearest_line[line_col_name]
        tie = nearest_tie[line_col_name]

        # append names to lists
        line_names.append(line)
        tie_names.append(tie)

        # get actual datapoints for each line (not LineString representation)
        line_points = lines_df[lines_df[line_col_name] == line]
        tie_points = ties_df[ties_df[line_col_name] == tie]

        # get nearest data point on each line/tie to intersection point
        nearest_datapoint_line = line_points.geometry.distance(p).sort_values().iloc[0]
        nearest_datapoint_tie = tie_points.geometry.distance(p).sort_values().iloc[0]

        # add distance to nearest data point on each line to lists
        line_dists.append(nearest_datapoint_line)
        tie_dists.append(nearest_datapoint_tie)

    # add names and distances as columns
    inters["line"] = line_names
    inters["tie"] = tie_names
    inters["line_dist"] = line_dists
    inters["tie_dist"] = tie_dists

    # get the largest of the two distance to each lines' nearest data point to the
    # theoretical intersection
    inters["max_dist"] = inters[["line_dist", "tie_dist"]].max(axis=1)

    # if intersection is not within cutoff_dist, remove rows
    if cutoff_dist is not None:
        prior_len = len(inters)
        inters = inters[inters.max_dist < cutoff_dist]
        msg = ()
        log.info(
            "removed %s intersections points which were further than %s km from "
            "nearest data point",
            prior_len - len(inters),
            int(cutoff_dist / 1000),
        )

    # get coords from geometry column
    inters["easting"] = inters.geometry.x
    inters["northing"] = inters.geometry.y

    if exclude_ints is not None:
        exclude_inds = []
        for i in exclude_ints:
            if isinstance(i, int | float):
                msg = (
                    "exclude_ints must be a list of lists of individual or pairs of "
                    "line numbers"
                )
                raise ValueError(msg)
            # if pair of lines numbers given, get those indices
            if len(i) == 2:
                ind = inters[(inters.line == i[0]) & (inters.tie == i[1])].index.values
            # if single line number, get all intersections of that line
            elif len(i) == 1:
                ind = inters[(inters.line == i[0]) | (inters.tie == i[0])].index.values
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
            subset=["line", "tie"],
            keep="last",
        )
        .sort_index()
    )
    b = len(inters)
    if a != b:
        log.info("Dropped %s duplicate intersections", a - b)

    if plot is True:
        plotly_points(
            inters,
            color_col="max_dist",
            hover_cols=["line", "tie", "max_dist", "line_dist", "tie_dist"],
            robust=True,
            point_size=6,
            theme=None,
            cmap="greys",
            title="Distance from intersection to nearest data point",
        )

    return inters.drop(columns=["line_dist", "tie_dist"])


def add_intersections(
    df: gpd.GeoDataFrame,
    intersections: gpd.GeoDataFrame,
    line_col_name: str = "line",
    time_col_name: str = "unixtime",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Add new rows to the dataframe for each intersection point and columns
    `is_intersection` and `intersection_line` to identify these intersections. All of
    the data column for these rows will have NaNs and should be filled with the
    function `interp1d_all_lines()`. Add columns to the intersections table for the
    distance along each line (flight and tie) to the intersection point. During
    levelling, levelling corrections are calculated using mistie values at intersections
    and interpolated along the entire lines based on these distances. Distances are
    calculate using the geometry column, and the time column informs which end of the
    line is the start.

    Parameters
    ----------
    df : gpd.GeoDataFrame
        Flight survey dataframe containing the data points to add intersections to.
        Must contain a geometry column and columns set by `line_col_name` and
        `time_col_name`
    intersections : gpd.GeoDataFrame
        Intersections table created by `create_intersection_table()`
    line_col_name : str, optional
        Column name specifying the line and tie names, by default "line"
    time_col_name : str, optional
        Column name specifying the time of each datapoints collection, by default
        "unixtime"

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        The updated flight survey dataframe and intersections table.
    """
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
    gdf["intersecting_line"] = np.nan

    # collect intersections to be added
    dfs = []
    for _, row in inters.iterrows():
        for i in list(gdf[line_col_name].unique()):
            if i in (row.line, row.tie):
                df = pd.DataFrame(
                    {
                        line_col_name: [i],
                        "easting": row.geometry.x,
                        "northing": row.geometry.y,
                        "is_intersection": True,
                    }
                )
                if i == row.line:
                    df["intersecting_line"] = row.tie
                else:
                    df["intersecting_line"] = row.line
                df["geometry"] = gpd.points_from_xy(df.easting, df.northing)
                dfs.append(df)

    # add intersections
    gdf = pd.concat([gdf, *dfs])

    # check correct number of intersections were added
    assert len(gdf) == prior_length + (2 * len(inters))

    # sort by lines
    gdf = gdf.sort_values(by=line_col_name)

    # get distance along each line
    gdf["dist_along_line"] = distance_along_line(
        gdf,
        line_col_name=line_col_name,
        time_col_name=time_col_name,
    )

    # sort by distance and reset index
    gdf = gdf.sort_values(by=[line_col_name, "dist_along_line"])
    gdf = gdf.reset_index(drop=True)

    # add dist along line to intersections dataframe
    # iterate through intersections
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_value = gdf[
            (gdf[line_col_name] == row.line) & (gdf.intersecting_line == row.tie)
        ].dist_along_line.values[0]
        tie_value = gdf[
            (gdf[line_col_name] == row.tie) & (gdf.intersecting_line == row.line)
        ].dist_along_line.values[0]

        inters.loc[ind, "dist_along_flight_line"] = line_value
        inters.loc[ind, "dist_along_flight_tie"] = tie_value

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
        line = LineString(grouped.get_group(name1).tolist())
        tie = LineString(grouped.get_group(name2).tolist())

        # get line endpoints
        # line_endpoints = MultiPoint(
        #   [Point(list(line.coords)[0]), Point(list(line.coords)[-1])])
        # tie_endpoints = MultiPoint(
        #   [Point(list(tie.coords)[0]), Point(list(tie.coords)[-1])])
        line_endpoints = MultiPoint([list(line.coords)[0], list(line.coords)[-1]])  # noqa: RUF015
        tie_endpoints = MultiPoint([list(tie.coords)[0], list(tie.coords)[-1]])  # noqa: RUF015

        # log.info(line_endpoints)
        # log.info(tie_endpoints)

        # get nearest points on each line to the closest of the other lines endpoints
        nearest_line_point_to_tie_endpoints = shapely.nearest_points(
            line, tie_endpoints
        )[0]
        nearest_tie_point_to_line_endpoints = shapely.nearest_points(
            tie, line_endpoints
        )[0]

        # log.info(nearest_tie_point_to_line_endpoints)
        # log.info(nearest_line_point_to_tie_endpoints)

        # get distances between nearest points on line with closest endpoint of
        # other line
        distance_tie_endpoint_to_line = np.min(
            [x.distance(nearest_line_point_to_tie_endpoints) for x in tie_endpoints]
        )
        distance_line_endpoint_to_tie = np.min(
            [x.distance(nearest_tie_point_to_line_endpoints) for x in line_endpoints]
        )

        # log.info(distance_tie_endpoint_to_line)
        # log.info(distance_line_endpoint_to_tie)

        # if distance is lower than cutoff, add intersection points to extend lines
        if distance_line_endpoint_to_tie <= max_interp_dist:
            tie_new = LineString(
                list(tie.coords) + list(nearest_line_point_to_tie_endpoints.coords)
            )
            assert len(list(tie.coords)) + 1 == len(list(tie_new.coords))
            log.info("extended line: %s", name1)
        else:
            tie_new = tie

        # repeat for tie
        if distance_tie_endpoint_to_line <= max_interp_dist:
            line_new = LineString(
                list(line.coords) + list(nearest_tie_point_to_line_endpoints.coords)
            )
            assert len(list(line.coords)) + 1 == len(list(line_new.coords))
            log.info("extended line: %s", name2)
        else:
            line_new = line

        # log.info(len(list(line.coords)))
        # log.info(len(list(tie.coords)))
        # log.info(len(list(line_new.coords)))
        # log.info(len(list(tie_new.coords)))


def get_line_tie_intersections(
    lines: gpd.GeoSeries,
    ties: gpd.GeoSeries,
):
    """
    adapted from https://gis.stackexchange.com/questions/137909/intersecting-lines-to-get-crossings-using-python-with-qgis
    """

    inters = []

    pbar = tqdm(
        itertools.product(lines, ties),
        desc="Line/Tie combinations:",
        total=len(list(itertools.product(lines, ties))),
    )

    # for i, (l, t) in enumerate(itertools.product(lines, ties)):
    for line, t in pbar:
        if line.intersects(t):
            inter = line.intersection(t)

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
    to_interp,
    interp_on,
    engine="scipy",
    method="cubic",
    plot=False,
    line_col="line",
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
        # try:
        filled = scipy_interp1d(**args)
        # except ValueError as e:
        #     # log.error(args)
        #     filled = np.nan
    else:
        msg = "invalid string for engine type"
        raise ValueError(msg)

    if plot is True:
        plot_line_and_crosses(
            filled,
            line=filled[line_col].iloc[0],
            x=interp_on,
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
    df = df.copy()

    # iterate through NaNs
    values = []
    for i in df[df[to_interp].isnull()].index:
        log.debug(f"Interpolating index: {i}")
        # get distance along line of NaN
        dist_at_nan = df[dist_col].loc[i]

        # try interpolation with set window width, if there's not enough data,
        # double the width
        win = window_width
        while True:
            try:
                # get data inside window
                llim, ulim = dist_at_nan - win, dist_at_nan + win
                df_inside = df[df[dist_col].between(llim, ulim)]

                # may be multiple NaN's within window (some outside of bounds)
                # but we only extract the fill value for loc[i]
                filled = interp1d(
                    df_inside,
                    to_interp=[to_interp],
                    **kwargs,
                )
                # extract just the filled value
                value = filled[to_interp].loc[i]
                # save value to a list
                values.append(value)
            except ValueError as e:
                # error messages for too few points in window
                few_points_errors = [
                    "cannot reshape array of",
                    "Found array with",
                    "The number of derivatives at boundaries does not match:",
                ]
                # error message for bounds error
                bounds_errors = [
                    "in x_new is above the interpolation range",
                    "in x_new is below the interpolation range",
                ]
                if any(item in str(e) for item in few_points_errors):
                    win += win
                    log.warning(
                        "too few points in window for intersection of lines %s & %s "
                        "doubling window size",
                        df.intersecting_line.loc[i],
                        df[line_col].loc[i],
                    )
                elif any(item in str(e) for item in bounds_errors):
                    win += win
                    log.warning(
                        "bounds error for interpolation of intersection of lines %s "
                        "and %s, doubling window size",
                        df.intersecting_line.loc[i],
                        df[line_col].loc[i],
                    )
                else:  # raise other errors
                    win += win
                    log.error(e)
                    log.warning(
                        "Error for interpolation of intersection of lines %s and %s, "
                        "doubling window size",
                        df.intersecting_line.loc[i],
                        df[line_col].loc[i],
                    )
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
    # print(values)
    df.loc[df[to_interp].isnull(), to_interp] = values

    if plot_line is True:
        plot_line_and_crosses(
            df,
            line=df[line_col].iloc[0],
            x=dist_col,
            y=[to_interp],
            y_axes=[i + 1 for i in range(len([to_interp]))],
        )

    return df


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
    with invert4geom_utils.DuplicateFilter(log):
        for col in to_interp:
            log.debug(f"Interpolating column: {col}")
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
    to_interp,
    interp_on,
    engine="scipy",
    method="cubic",
    plot_line=False,
    line_col="line",
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
    with invert4geom_utils.DuplicateFilter(log):
        for col in to_interp:
            filled = interp1d_single_col(
                df1,
                to_interp=col,
                interp_on=interp_on,
                engine=engine,
                method=method,
            )
            # try:
            df1[col] = filled[col]
            # except:
            #     log.error(f"Error with filling nans in column: {col}")

    if plot_line is True:
        plot_line_and_crosses(
            df1,
            line=df1[line_col].iloc[0],
            x=interp_on,
            y=to_interp,
            y_axes=[i + 1 for i in range(len(to_interp))],
        )

    return df1


def interp1d_all_lines(
    df: pd.DataFrame | gpd.GeoDataFrame,
    to_interp: list[str] | None = None,
    interp_on: str = "dist_along_line",
    method="cubic",
    engine="scipy",
    line_col="line",
    window_width: float | None = None,
    plot: bool = False,
    wait_for_input: bool = False,
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    _summary_

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        Dataframe containing the data to interpolate
    to_interp : list[str] | None, optional
        specify which column to interpolate NaNs for, by default is all columns except
        "is_intersection" and "intersecting_line"
    interp_on : str, optional
        Decide which column interpolation is based on, by default "dist_along_line"
    method : str, optional
        Decide between interpolation methods of 'linear', 'nearest', 'nearest-up',
        'zero', 'slinear', 'quadratic','cubic', 'previous', or 'next' if engine is
        "scipy", or vd.Spline(), vd.SplineCV(), vd.KNeighbors(), vd.Linear(), or
        vd.Cubic() if engine is "verde", by default "cubic"
    engine : str, optional
        Decide between "scipy" and "verde" for performing the interpolation, by default
        "scipy"
    line_col : str, optional
        Column name specifying the line numbers, by default "line"
    window_width : float, optional
        window width around each NaN to use for interpolation fitting, by default None
    plot : bool, optional
        plot the lines and interpolated points at intersections, by default False
    wait_for_input : bool, optional
        if true, will pause after each plot to allow inspection, by default False

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        the survey dataframe with NaN's filled in the specified columns
    """
    df = df.copy()

    if to_interp is None:
        to_interp = df.columns.drop(["is_intersection", "intersecting_line"])

    lines = df.groupby(line_col)
    filled_lines = []
    pbar = tqdm(lines, desc="Lines")
    for line, line_df in pbar:
        pbar.set_description(f"Line {line}")

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
            if "geometry" in to_interp:
                to_interp = to_interp.drop("geometry")

            plot_line_and_crosses(
                filled,
                line=line,
                x=interp_on,
                y=to_interp,
                y_axes=[i + 1 for i in range(len(to_interp))],
            )

            if wait_for_input is True:
                input("Press key to continue...")

    return pd.concat(filled_lines)


def calculate_misties(
    intersections: gpd.GeoDataFrame,
    data: gpd.GeoDataFrame,
    data_col: str,
    line_col: str = "line",
    plot: bool = False,
    robust: bool = True,
) -> gpd.GeoDataFrame:
    """
    Calculate mistie values for all intersections. For each intersection, find the data
    values for the line and tie from the survey dataframe and add those values to the
    intersection table as `line_value` and `tie_value`. If they exist, overwrite them.
    Calculate the mistie value as line_value - tie_value, and save this to a column
    `mistie_0`. If `mistie_0` exists, make a new column `mistie_1`, etc. If the new
    mistie values exactly match previous, don't make a new column. This allow to run
    the function multiple times without changing anything and not building up a large
    number of mistie columns.

    Parameters
    ----------
    intersections : gpd.GeoDataFrame
        Intersections table created by `create_intersection_table()`, then supplied to
        `add_intersections()`.
    data : gpd.GeoDataFrame
        Survey dataframe with intersection rows added by `add_intersections()` and
        interpolated with `interp1d_all_lines()`.
    data_col : str
        Column name for data values to calculate misties for
    line_col : str, optional
        Column name specifying the line numbers, by default "line"
    plot : bool, optional
        Plot the resulting mistie points on a map, by default False
    robust : bool, optional
        Use robust color limits for the map, by default True

    Returns
    -------
    gpd.GeoDataFrame
        An intersections table with new columns `line_value`, `tie_value` and `mistie_x`
        where x is incremented each time a new mistie is calculated.
    """

    inters = intersections.copy()
    df = data.copy()

    # save previous mistie columns by adding an integer to the end
    # if "mistie" in inters.columns:
    #     version = 0
    #     col_name = f"mistie_{0}"
    #     while col_name in inters.columns:
    #         version += 1
    #         col_name = f"mistie_{version}"
    #     inters = inters.rename(columns={"mistie": col_name})
    # get list of columns starting with "mistie"

    cols = [col for col in inters.columns if "mistie_" in col]
    if len(cols) == 0:
        mistie_col = "mistie_0"
        past_mistie_col = None
    else:
        mistie_col = f"mistie_{len(cols)}"
        past_mistie_col = f"mistie_{len(cols)-1}"
        log.info("Previous mistie column: %s", past_mistie_col)

    log.info("New mistie column: %s", mistie_col)
    # get the latest mistie column
    # mistie_col = [int(col.split("_")[-1]) for col in inters.columns if "mistie" in col] # noqa: E501
    # mistie_col = f"mistie_{max(mistie_col)}"

    # iterate through intersections
    for ind, row in inters.iterrows():
        # search data for values at intersecting lines
        line_value = df[(df[line_col] == row.line) & (df.intersecting_line == row.tie)][
            data_col
        ].values[0]
        tie_value = df[(df[line_col] == row.tie) & (df.intersecting_line == row.line)][
            data_col
        ].values[0]

        assert line_value != np.nan
        assert tie_value != np.nan

        inters.loc[ind, "line_value"] = line_value
        inters.loc[ind, "tie_value"] = tie_value

        # add misties to rows of data df which are intersection points
        # conditions = (df[line_col] == row.line) & (df.intersecting_line == row.tie)
        # df.loc[conditions, "mistie"] = line_value - tie_value

        # conditions = (df[line_col] == row.tie) & (df.intersecting_line == row.line)
        # df.loc[conditions, "mistie"] = line_value - tie_value

    # misties are defined as line - tie
    misties = inters.line_value - inters.tie_value

    log.info(f"mistie RMSE: {utils.rmse(misties)}")

    if past_mistie_col is not None:
        try:
            pd.testing.assert_series_equal(
                inters[past_mistie_col],
                misties,
                check_names=False,
            )
            log.error("Mistie values are equal, not create a new column")
            mistie_col = past_mistie_col
        except AssertionError:
            inters[mistie_col] = misties

    if plot is True:
        plotly_points(
            inters,
            color_col=mistie_col,
            hover_cols=["line", "tie", "line_value", "tie_value"],
            robust=robust,
            point_size=5,
        )

    return inters  # , df


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
) -> pd.DataFrame:
    """
    data_to_fit: pd.DataFrame with at least 2 columns: distance, and data
    cols_to_fit: column names representing distance and data
    data_to_predict: pd.DataFrame with at least 1 columns: distance
    cols_to_predict: column names representing distance and new column
        with predicted data
    """
    fit_df = data_to_fit.copy()
    predict_df = data_to_predict.copy()

    # fit a polynomial trend through the lines mistie values
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


def level_survey_lines_to_grid(
    df: pd.DataFrame | gpd.GeoDataFrame,
    grid_column_name: str,
    degree: int,
    data_column_name: str,
    line_column_name: str = "line",
    distance_column_name: str = "dist_along_line",
    levelled_column_name: str = "levelled",
) -> pd.DataFrame | gpd.GeoDataFrame:
    """
    With grid values sampled along survey flight lines (grid_column_name), fit a trend
    or specified order to the misfit values (data_column_name - grid_column_name) and
    apply the correction to the data. The levelled data is saved in a new column
    specified by levelled_column_name.

    Parameters
    ----------
    df : pd.DataFrame | gpd.GeoDataFrame
        _description_
    grid_column_name : str
        _description_
    degree : int
        _description_
    data_column_name : str
        _description_
    line_column_name : str, optional
        _description_, by default "line"
    distance_column_name : str, optional
        _description_, by default "dist_along_line"
    levelled_column_name : str, optional
        _description_, by default "levelled"

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        _description_

    Raises
    ------
    ValueError
        _description_
    """
    df = df.copy()

    # calculate misfit survey data and sampled grid values
    if "misfit" in df.columns:
        msg = "Column 'misfit' already exists in dataframe, drop or rename it first"
        raise ValueError(msg)

    df["misfit"] = df[data_column_name] - df[grid_column_name]

    # fit a trend to the misfits on line-by-line basis
    for line in df[line_column_name].unique():
        # subset a line
        line_df = df[df[line_column_name] == line]

        # calculate correction by fitting trend to misfit values
        correction = skl_predict_trend(
            data_to_fit=line_df,
            cols_to_fit=[distance_column_name, "misfit"],
            data_to_predict=line_df,
            cols_to_predict=[distance_column_name, "correction"],
            degree=degree,
        ).correction

        # add correction values to the main dataframe
        df.loc[df[line_column_name] == line, "levelling_correction"] = correction

        # apply correction to the data
        # df.loc[df[line_column_name]==line, levelled_column_name] = line_df[data_column_name] - correction # noqa: E501

    # apply correction to the data
    df[levelled_column_name] = df[data_column_name] - df.levelling_correction

    # # add correction to existing correction column if it exists
    # if correction_column_name in df.columns:
    #     df[correction_column_name] += df[f"trend_{degree}_correction"]
    # else:
    #     df[correction_column_name] = df[f"trend_{degree}_correction"]

    return df.drop(columns=["misfit", "levelling_correction"])


def level_lines(
    inters: gpd.GeoDataFrame | pd.DataFrame,
    data: gpd.GeoDataFrame | pd.DataFrame,
    lines_to_level: list[float],
    data_col: str,
    cols_to_fit: str | None = None,
    cols_to_predict: str = "dist_along_line",
    degree: int | None = None,
    line_col: str = "line",
    plot=False,
):
    """
    Level lines based on intersection misties values. Fit a trend of specified order to
    intersection misties, and apply the correction to the `data_col` column.
    """
    df = data.copy()

    if cols_to_fit is None:
        # if levelling to ties, fit to dist_along_flight_tie
        if lines_to_level[0] in inters.tie.unique():
            cols_to_fit = "dist_along_flight_tie"
        elif lines_to_level[0] in inters.line.unique():
            cols_to_fit = "dist_along_flight_line"

    # convert columns to fit on into a list if its a string
    if isinstance(cols_to_fit, str):
        cols_to_fit = [cols_to_fit]
    if isinstance(cols_to_predict, str):
        cols_to_predict = [cols_to_predict]

    levelled_col = f"{data_col}_levelled"
    df[levelled_col] = np.nan
    df["levelling_correction"] = np.nan

    # get the latest mistie column
    mistie_col = [int(col.split("_")[-1]) for col in inters.columns if "mistie" in col]
    mistie_col = f"mistie_{max(mistie_col)}"

    # iterate through the chosen lines
    for line in lines_to_level:
        line_df = df[df[line_col] == line].copy()

        # get intersections of line of interest
        ints = inters[(inters.line == line) | (inters.tie == line)]

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
                        log.error("Issue with line %s, skipping", line)
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
                    log.error("Issue with line %s, skipping", line)
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
    log.info(f"previous mistie RMSE: {utils.rmse(inters[mistie_col])}")
    inters_new = calculate_misties(
        inters,
        df,
        data_col=levelled_col,
        plot=False,
    )

    if plot is True:
        # plot old and new misties
        ints = inters_new[
            inters_new.line.isin(lines_to_level) | inters_new.tie.isin(lines_to_level)
        ]
        plotly_points(
            ints,
            color_col=mistie_col,
            point_size=4,
            hover_cols=[
                "line",
                "tie",
                "line_value",
                "tie_value",
                mistie_col,
            ],
        )
        plotly_points(
            ints,
            color_col=mistie_col,
            point_size=4,
            hover_cols=[
                "line",
                "tie",
                "line_value",
                "tie_value",
                mistie_col,
            ],
        )

        plotly_points(
            df[df[line_col].isin(lines_to_level)],
            color_col="levelling_correction",
            point_size=2,
            hover_cols=[line_col, data_col, levelled_col],
        )

    return df.drop(columns=["levelling_correction"]), inters_new


def iterative_line_levelling(
    inters,
    data,
    lines_to_level,
    degree,
    starting_mistie_col,
    starting_data_col,
    iterations,
    cols_to_fit,
    cols_to_predict,
    mistie_prefix=None,
    levelled_data_prefix=None,
    plot_iterations=False,
    plot_results=False,
    plot_convergence=False,
    line_col="line",
    **kwargs,
):
    df = data.copy()
    ints = inters.copy()

    if mistie_prefix is None:
        mistie_prefix = f"mistie_trend{degree}"
    if levelled_data_prefix is None:
        levelled_data_prefix = f"levelled_data_trend{degree}"

    rmse = utils.rmse(ints[starting_mistie_col])
    log.info("Starting RMS mistie: %s mGal", rmse)

    for i in range(1, iterations + 1):
        if i == 1:
            data_col = starting_data_col
            mistie_col = starting_mistie_col
        else:
            data_col = f"{levelled_data_prefix}_{i-1}"
            mistie_col = f"{mistie_prefix}_{i-1}"

        # with inv_utils.HiddenPrints():
        df, ints = level_lines(
            ints,
            df,
            lines_to_level=lines_to_level,
            cols_to_fit=cols_to_fit,
            cols_to_predict=cols_to_predict,
            degree=degree,
            data_col=data_col,
            levelled_col=f"{levelled_data_prefix}_{i}",
            mistie_col=mistie_col,
            new_mistie_col=f"{mistie_prefix}_{i}",
            line_col=line_col,
        )
        rmse = utils.rmse(ints[f"{mistie_prefix}_{i}"])
        log.info("RMS mistie after iteration %s: %s mGal", i, rmse)
        rmse_corr = utils.rmse(df[(df.line.isin(lines_to_level))].levelling_correction)
        log.info("RMS correction to lines: %s mGal", rmse_corr)

        if plot_iterations is True:
            # plot flight lines
            plotly_points(
                df[df.line.isin(lines_to_level)],
                color_col="levelling_correction",
                point_size=2,
                hover_cols=[
                    line_col,
                    f"{levelled_data_prefix}_{i}",
                ],
            )

    levelled_col = list(df.columns)[-1]

    df["levelling_correction"] = df[starting_data_col] - df[levelled_col]
    if plot_convergence is True:
        plot_levelling_convergence(
            ints,
            mistie_prefix=mistie_prefix,
            logy=kwargs.get("logy", False),
        )
    if plot_results is True:
        # plot flight lines
        plotly_points(
            df[df.line.isin(lines_to_level)],
            color_col="levelling_correction",
            point_size=4,
            hover_cols=[
                line_col,
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
    log.info("Starting RMSE mistie: %s mGal", rmse)

    for i in range(1, iterations + 1):
        if i == 1:
            data_col = starting_data_col
            mistie_col = starting_mistie_col
        else:
            data_col = f"{levelled_data_prefix}_{i-1}t"
            mistie_col = f"{mistie_prefix}_{i-1}t"

        # level lines to ties
        df, ints = level_lines(
            ints,
            df,
            lines_to_level=flight_line_names,
            cols_to_fit="dist_along_flight_line",
            cols_to_predict="dist_along_line",
            degree=degree,
            data_col=data_col,
            levelled_col=f"{levelled_data_prefix}_{i}l",
            mistie_col=mistie_col,
            new_mistie_col=f"{mistie_prefix}_{i}l",
            line_col="line",
        )
        rmse = utils.rmse(ints[f"{mistie_prefix}_{i}l"])
        log.info(f"RMSE mistie after iteration {i}: L -> T: {rmse} mGal")
        rmse_corr = utils.rmse(
            df[(df.line.isin(flight_line_names))].levelling_correction
        )
        log.info(f"RMS correction to lines: {rmse_corr} mGal")

        # level ties to lines
        df, ints = level_lines(
            ints,
            df,
            lines_to_level=tie_line_names,
            cols_to_fit="dist_along_flight_tie",
            cols_to_predict="dist_along_line",
            degree=degree,
            data_col=f"{levelled_data_prefix}_{i}l",
            levelled_col=f"{levelled_data_prefix}_{i}t",
            mistie_col=f"{mistie_prefix}_{i}l",
            new_mistie_col=f"{mistie_prefix}_{i}t",
            line_col="line",
        )
        rmse = utils.rmse(ints[f"{mistie_prefix}_{i}t"])
        log.info("RMSE mistie after iteration %s: T -> L: %s mGal", i, rmse)
        rmse_corr = utils.rmse(df[(df.line.isin(tie_line_names))].levelling_correction)
        log.info("RMS correction to ties: %s mGal", rmse_corr)

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
    cmap_lims=None,
    robust=True,
    theme="plotly_dark",
    title=None,
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

    if cmap_lims is None:
        vmin, vmax = utils.get_min_max(data[color_col], robust=robust)
        cmap_lims = (vmin, vmax)
    else:
        vmin, vmax = cmap_lims

    if cmap is None:
        if (cmap_lims[0] < 0) and (cmap_lims[1] > 0):
            cmap = "RdBu_r"
            cmap_middle = 0
        else:
            cmap = None
            cmap_middle = None
    else:
        cmap_middle = None

    if cmap_middle == 0:
        max_abs = vd.maxabs((vmin, vmax))
        cmap_lims = (-max_abs, max_abs)

    fig = px.scatter(
        data,
        x=coord_names[0],
        y=coord_names[1],
        color=data[color_col],
        color_continuous_scale=cmap,
        color_continuous_midpoint=cmap_middle,
        range_color=cmap_lims,
        hover_data=hover_cols,
        template=theme,
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        title_text=title,
        autosize=False,
        width=800,
        height=800,
    )

    fig.update_traces(marker={"size": point_size})

    fig.show()


def plotly_profiles(
    data,
    y: tuple[str],
    x="dist_along_line",
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
    y: tuple[str],
    intersections=None,
    line=None,
    line_col_name="line",
    x="dist_along_line",
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
                    # if intersections supplied, plot them
                    if intersections is not None:
                        # is supplied line is a line (not a tie)
                        if line in intersections.line.unique():
                            # if crossing line value exists, use that for y axis
                            if "tie_value" in intersections.columns:
                                y = intersections[intersections.line == line].tie_value
                            # if crossing line value doesn't exist, just plot
                            # intersection at lines value
                            else:
                                y = df[df.is_intersection][z]
                            text = intersections[intersections.line == line].tie
                        # if supplied line is a tie (not a line)
                        elif line in intersections.tie.unique():
                            # if crossing line value exists, use that for y axis
                            if "line_value" in intersections.columns:
                                y = intersections[intersections.tie == line].line_value
                            # if crossing line value doesn't exist, just plot
                            # intersection at lines value
                            else:
                                y = df[df.is_intersection][z]
                            text = intersections[intersections.tie == line].line
                        fig.add_trace(
                            go.Scatter(
                                mode="markers+text",
                                x=df[df.is_intersection][x],
                                y=y,
                                yaxis=y_axes[i],
                                marker_size=5,
                                marker_symbol="diamond",
                                name="intersections",
                                text=text,
                                textposition="top center",
                            ),
                        )
                        text = line if j == 0 else None
                    else:
                        text = (
                            df[df.is_intersection].intersecting_line if j == 0 else None
                        )
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
