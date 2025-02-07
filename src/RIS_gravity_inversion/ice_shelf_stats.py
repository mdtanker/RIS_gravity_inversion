from __future__ import annotations

import itertools
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import patheffects
from adjustText import adjust_text
import numpy as np
import pandas as pd
import scipy as sp
import xarray as xr
import plotly.express as px
import plotly.graph_objects as go
import pygmt
import scipy
import seaborn as sns
import shapely
import verde as vd
from polartoolkit import utils, fetch, maps, profiles
from shapely.geometry import LineString, MultiPoint, Point
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from tqdm.autonotebook import tqdm
from invert4geom import utils as invert4geom_utils
from invert4geom import regional

from RIS_gravity_inversion import log

inverted_shelves = [
    "Ross", # tinto 2019
    "Amery", # yang 2021
    "Borchgrevink", # eisermann 2021
    "Baudouin", # eisermann 2021
    "George_VI", # constantino 2020
    "Thwaites", # tinto and bell 2011, jordan 2020, millan 2017
    "Cook", # constantino and tinto 2023
    "Ninnis", # constantino and tinto 2023
    "Ekstrom", # eisermann 2020
    "Atka", # eisermann 2020
    "Jelbart", # eisermann 2020
    "Fimbul", # eisermann 2020
    "Vigrid", # eisermann 2020
    "Crosson", # jordan 2020, millan 2017
    "Dotson", # jordan 2020, millan 2017
    "Getz", # wei 2020, millan 2020, cochran 2020
    "Totten", # greenbaum 2015, vankova 2023
    "Brunt_Stancomb", # hodgson 2019
    "Pine_Island", # millan 2017, muto 2013, muto 2016
    "Abbot", # cochran 2014
    "LarsenC", # cochran and bell 2012
    "Nivl", # eisermann 2024
    "Lazarev", # eisermann 2024
    "Venable", # Locke et al. 2025
]

def combine_ibcso_offshore_bedmap_onshore(
    region: tuple[float, float, float, float],
    bedmap_version:str = "all",
    grounded_as_points: bool = False,
    spacing: float = 1e3,
):
    # fetch points for region
    ibcso_points_gdf, ibcso_polygons_gdf = fetch.ibcso_coverage(region=region)

    # drop points with TID of 45 meaning it's from gravity inversion
    ibcso_points_gdf = ibcso_points_gdf[ibcso_points_gdf.dataset_tid!=45]

    # convert polygons (mostly swath multibeam) to grid of points which are within the
    # polygon with a grid spacing specified
    polygon_points = None
    if len(ibcso_polygons_gdf) > 0:
        try:
            polygon_points = polygons_to_points(
                ibcso_polygons_gdf,
                spacing=spacing,
            )
        except ValueError as e:
            log.error(e)
            log.error("Failed to convert polygons to points")

    # combine ibcso points and polygon points
    total_ibcso_points = pd.concat([ibcso_points_gdf, polygon_points])

    # drop geometry column
    total_ibcso_points = total_ibcso_points.drop(columns=["geometry"])

    # used bedmap points just for grounded region
    if grounded_as_points is False:
        bedmap_points_gdf = fetch.bedmap_points(region=region, version=bedmap_version)

        # drop points without bed elevations
        bedmap_points_gdf = bedmap_points_gdf.dropna(subset=["bedrock_altitude (m)"])

        # get grid of rock outcrops
        rock_mask = fetch.bedmap2(
            layer="rockmask",
            region=region,
            spacing=1e3,
            verbose="q",
        ).rename("bedrock_altitude (m)")

        # convert to dataframe
        rock_df = vd.grid_to_table(rock_mask).dropna()

        # merge with bedmap points
        bedmap_points_gdf = pd.concat([rock_df, bedmap_points_gdf])

        # convert to a geodataframe
        geometry = [shapely.geometry.Point(xy) for xy in zip(bedmap_points_gdf.x, bedmap_points_gdf.y)]
        bedmap_points_gdf = gpd.GeoDataFrame(bedmap_points_gdf, crs="EPSG:3031", geometry=geometry)

        # get grounding line shapefile
        grounding_line = gpd.read_file(fetch.groundingline(version="measures-v2"), engine="pyogrio")

        # add column indicating if points are grounded or not
        bedmap_points_gdf = gpd.tools.sjoin(
            bedmap_points_gdf,
            grounding_line,
            predicate="within",
            how="left",
        )

        # discard points offshore
        bedmap_points_grounded = bedmap_points_gdf[bedmap_points_gdf.NAME=="Grounded"]

        # rename bedmap columns to match ibcso and drop unused columns
        bedmap_points_grounded = bedmap_points_grounded[["x", "y", "project"]].copy()
        bedmap_points_grounded = bedmap_points_grounded.rename(
            columns={
                "x": "easting",
                "y": "northing",
                "project":"dataset_name",
            })

    # treat every gridcell of grounded ice as a constraint point
    elif grounded_as_points is True:
        # get grid with 0 for grounded and 1 for floating ice
        grounded_da = fetch.bedmap2(
            layer="icemask_grounded_and_shelves",
            region=region,
            spacing=1e3,
            verbose="q",
        )
        # turn into dataframe
        bedmap_points_grounded = vd.grid_to_table(grounded_da).dropna()

        # keep only grounded points
        bedmap_points_grounded = bedmap_points_grounded[bedmap_points_grounded.z==0]

        # add column for project
        bedmap_points_grounded["dataset_name"] = "bedmap2_grid_points"

        # rename columns
        bedmap_points_grounded = bedmap_points_grounded.rename(
            columns={
                "x": "easting",
                "y": "northing",
            }
        )

        # drop some columns
        bedmap_points_grounded = bedmap_points_grounded.drop(columns=["z"])

    # add column indicating if points are onshore or offshore
    bedmap_points_grounded["onshore"] = True

    total_ibcso_points["onshore"] = False

    bed_points = pd.concat(
        [
            total_ibcso_points.drop(columns=["weight", "dataset_tid"]),
            bedmap_points_grounded,
        ],
        ignore_index=True,
        sort=False,
    )

    return bed_points


def get_ice_shelves():

    # read into a geodataframe
    ice_shelves = gpd.read_file(fetch.antarctic_boundaries(version="IceShelf"))

    # get list of unique ice shelf names
    names = ice_shelves.NAME.unique()

    log.info(f"Number of unmerged ice shelves: {len(names)}")

    # log.info(names)

    # get Series of first words in ice shelf names and number of shelves with that name
    first_words = pd.Series(n.split("_")[0] for n in names).value_counts()

    # get list of lists of ice shelves which should be combined
    shelves_to_combine = {
        "Ronne_Filchner": ["Ronne", "Filchner"],
    }
    dont_combine = ["Hamilton"]
    for i, j in first_words.items():
        if (j > 1) & (i not in dont_combine):
            shelves_to_combine[i] = [n for n in names if n.startswith(i)]

    shelves_to_merge = []
    num_sub_shelves = 0
    for (k, v) in shelves_to_combine.items():
        log.info("Combining %s %s sub-shelves", len(v), k)

        num_sub_shelves += len(v)
        # merge into new shelf
        shelf = ice_shelves[ice_shelves.NAME.isin(v)].dissolve()
        shelf.NAME = k

        # remove old shelves
        ice_shelves = ice_shelves[~ice_shelves.NAME.isin(v)]

        shelves_to_merge.append(shelf)

    log.info(f"Number of removed sub-shelves: {num_sub_shelves}")

    # remove old shelves
    ice_shelves = ice_shelves[~ice_shelves.NAME.isin(v)]

    # add to dataframe
    ice_shelves = pd.concat([ice_shelves]+shelves_to_merge)

    # there are 2 Fox ice shelves, one in East Antarctica and one in West Antarctica
    # append the region to the name to differentiate
    ice_shelves.loc[(ice_shelves.NAME=="Fox") & (ice_shelves.Regions=="West"), "NAME"] = "Fox_West"
    ice_shelves.loc[(ice_shelves.NAME=="Fox") & (ice_shelves.Regions=="East"), "NAME"] = "Fox_East"

    # calculate area in sq km of each ice shelf
    ice_shelves["area_km"] = round((ice_shelves.area / (1000**2)))

    # sort by area
    ice_shelves = ice_shelves.sort_values(by="area_km", ascending=False).reset_index(drop=True)

    log.info(f"Number of merged ice shelves: {len(ice_shelves)}")
    # get coordinate columns
    # ice_shelves["easting"] = ice_shelves.get_coordinates().x
    # ice_shelves["northing"] = ice_shelves.get_coordinates().y

    return ice_shelves


def plot_ice_shelf_names(
    fig: pygmt.Figure,
    ice_shelves: gpd.GeoDataFrame,
    font="8p,Helvetica",
    offset=".4c/.4c+v.6p",
):
    # plot names with white shadow
    fig.text(
        x=ice_shelves.geometry.centroid.x,
        y=ice_shelves.geometry.centroid.y,
        text=[x.replace("_", " ") for x in ice_shelves.NAME],
        font=f"{font},black,-=2p,white",
        justify="BL",
        offset=offset,
        no_clip=True,
    )
    fig.text(
        x=ice_shelves.geometry.centroid.x,
        y=ice_shelves.geometry.centroid.y,
        text=[x.replace("_", " ") for x in ice_shelves.NAME],
        font=f"{font},black",
        justify="BL",
        offset=offset,
        no_clip=True,
    )
    # for i, (_, row) in enumerate(ice_shelves.iterrows()):
    #     if i % 2 == 0:
    #         offset=.4
    #         justify="BL"
    #     else:
    #         offset=-.4
    #         justify="TR"
    #     fig.text(
    #         x=row.geometry.centroid.x,
    #         y=row.geometry.centroid.y,
    #         text=row.NAME.replace("_", " "),
    #         font="8p,Helvetica,black,-=3p,white",
    #         justify=justify,
    #         offset=f"{offset}c/{offset}c+v.6p",
    #         no_clip=True,
    #     )
    #     fig.text(
    #         x=row.geometry.centroid.x,
    #         y=row.geometry.centroid.y,
    #         text=row.NAME.replace("_", " "),
    #         font="8,Helvetica,black",
    #         justify=justify,
    #         offset=f"{offset}c/{offset}c+v.6p",
    #         no_clip=True,
    #     )


def constraints_and_min_distances_single(
    ice_shelf: gpd.GeoSeries,
    buffer: float = 10e3,
    spacing: float = 1e3,
    bedmap_version: str = "all",
    grounded_as_points: bool = False,
    fname: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:

    # convert to geodataframe
    gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

    # define region around each ice shelf with 10km buffer
    reg = utils.region_to_bounding_box(gdf.iloc[0].geometry.bounds)
    reg = vd.pad_region(reg, buffer)

    # get ibcso points offshore and bedmap points onshore for the region
    bed_points = combine_ibcso_offshore_bedmap_onshore(
        bedmap_version=bedmap_version,
        region=reg,
        grounded_as_points=grounded_as_points,
        spacing=spacing,
    )

    # add column indicating which ice shelf each point is within
    # bed_points = gpd.tools.sjoin(
    #     bed_points,
    #     gdf,
    #     predicate="within",
    #     how="left",
    # )
    # use only points within specific ice shelf (ungrounded)
    # bed_points = bed_points[bed_points.NAME==gdf.iloc[0].NAME]

    coords = vd.grid_coordinates(
        region=reg,
        spacing=spacing,
    )
    grid = vd.make_xarray_grid(coords, np.ones_like(coords[0]), data_names="z").z

    # calculate minimum distance between each grid cell and the nearest point
    min_dist = invert4geom_utils.normalized_mindist(
        bed_points, #.rename(columns={"easting": "x", "northing": "y"}),
        grid = grid,
    )

    # mask to the ice shelf outline
    min_dist = utils.mask_from_shp(
        shapefile=gdf,
        xr_grid=min_dist,
        invert=False,
        masked=True,
    )

    if fname is not None:
        min_dist.to_netcdf(f"{fname}_min_dist.nc")
        bed_points.to_csv(
            f"{fname}_constraints.csv.gz",
            sep=",",
            na_rep="",
            header=True,
            index=False,
            encoding="utf-8",
            compression="gzip",
        )

    if plot:
        if save_plot:
            fname = fname
        else:
            fname = None
        plot_constraints_and_min_distances(
            gdf.iloc[0],
            min_dist,
            bed_points,
            fname,
        )


def constraints_and_min_distances(
    ice_shelves: gpd.GeoDataFrame,
    buffer: float = 10e3,
    spacing: float = 1e3,
    bedmap_version: str = "all",
    grounded_as_points: bool = False,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:
    """
    Compute the median of the minimum distances between each grid cell and the
    nearest constraint point (bedmap onshore, ibcso offshore) for each supplied ice
    shelf shapefile. Save the constraints to a csv and the minimum distance grid to a
    netcdf. Optionally, plot the results.

    Parameters
    ----------
    ice_shelves : gpd.GeoDataFrame
        ice shelf shapefiles, output from `get_ice_shelves`
    buffer : float, optional
        buffer zone around ice shelf bounding box to include points in, by default 10e3
    spacing : float, optional
        grid spacing to use for computing minimum distances, by default 1e3
    bedmap_version : str, optional
        which bedmap points to include, either "bedmap1", "bedmap2", "bedmap3", or
        "all", by default "all"
    grounded_as_points : bool, optional
        use bedmap grid cells as points rather than actual points, by default False
    file_path : str | None, optional
        path to save the constraints and minimum distance files, by default None
    plot : bool, optional
        show a plot for each ice shelf, by default False

    Returns
    -------
    gpd.GeoDataFrame
        Ice shelf geodataframe with additional column "median_min_dist" containing the
        median minimum distance to the nearest point
    """
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for i, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")
        constraints_and_min_distances_single(
            row,
            buffer=buffer,
            spacing=spacing,
            bedmap_version=bedmap_version,
            grounded_as_points=grounded_as_points,
            fname=file_path+row.NAME,
            plot=plot,
            save_plot=save_plot,
        )


def load_constraints_and_min_distances(
    ice_shelves: gpd.GeoDataFrame,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")
        constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
        min_dist = xr.open_dataarray(f"{file_path}{row.NAME}_min_dist.nc")

        gdf.loc[gdf.index, f"median_constraint_distance"] = min_dist.median().values

        if plot:
            if save_plot:
                fname = file_path+row.NAME
            else:
                fname = None

            plot_constraints_and_min_distances(
                row,
                min_dist=min_dist,
                constraints_df=constraints_df,
                fname=fname,
            )

    return gdf

def plot_constraints_and_min_distances(
    ice_shelf: gpd.GeoSeries,
    min_dist: xr.DataArray,
    constraints_df: pd.DataFrame,
    fname: str | None = None,
):
    name = ice_shelf.NAME

    fig = maps.plot_grd(
        min_dist,
        title=f"{name.replace("_", " ")} Ice Shelf",
        cbar_label=f"Constraint median distance; {int(min_dist.median())} (m)",
        cmap="dense",
        hist=True,
        inset=True,
        robust=True,
        scalebar=True,
        coast=True,
        points=constraints_df,
        points_style="c1p",
    )
    fig.show()

    if fname is not None:
        fig.savefig(f"{fname}_constraints.png")


def gravity_anomalies_single(
    ice_shelf: gpd.GeoSeries,
    regional_grav_kwargs: dict,
    constraints_df: pd.DataFrame,
    buffer: float = 10e3,
    spacing: float = 10e3,
    progressbar: bool = False,
    fname: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None :

    # convert to geodataframe
    gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

    # define region around each ice shelf with 10km buffer
    reg = utils.region_to_bounding_box(gdf.iloc[0].geometry.bounds)
    reg = vd.pad_region(reg, buffer)

    log.debug("calculated region %s", reg)
    log.debug("fetching, subsetting, and resampling datasets")

    # download gravity data
    eigen_gravity = fetch.gravity(
        version="eigen",
        spacing=spacing,
        region=reg,
        registration="g",
        verbose="q",
    )
    antgg_disturbance = fetch.gravity(
        version="antgg-2021",
        anomaly_type="DG",
        spacing=spacing,
        region=reg,
        registration="g",
        verbose="q",
    )
    antgg_uncertainty = fetch.gravity(
        version="antgg-2021",
        anomaly_type="Err",
        spacing=spacing,
        region=reg,
        registration="g",
        verbose="q",
    )

    # download topography
    surface = fetch.bedmap2(
        layer="surface",
        spacing=spacing,
        region=reg,
        fill_nans=True,
        reference="ellipsoid",  # converts to be referenced to the ellipsoid
        verbose="q",
    ).rename({"x":"easting", "y":"northing"})
    icebase = fetch.bedmap2(
        layer="icebase",
        spacing=spacing,
        region=reg,
        fill_nans=True,  # fills nans over ocean with 0's (while still reference to the geoid)
        reference="ellipsoid",  # converts to be referenced to the ellipsoid
        verbose="q",
    ).rename({"x":"easting", "y":"northing"})
    bed = fetch.bedmap2(
        layer="bed",
        spacing=spacing,
        region=reg,
        fill_nans=True,  # fills nans over ocean with 0's (while still reference to the geoid)
        reference="ellipsoid",  # converts to be referenced to the ellipsoid
        verbose="q",
    ).rename({"x":"easting", "y":"northing"})

    # make gravity dataframe
    grav_df = xr.merge(
        [
            eigen_gravity.gravity,
            antgg_disturbance.gravity_disturbance,
            antgg_disturbance.ellipsoidal_height,
            antgg_uncertainty.error,
        ]
    ).to_dataframe().reset_index().rename(columns={"x":"easting", "y":"northing"})

    # check layers to cross
    # anywhere bed is above icebase, set equal to icebase
    bed = xr.where(bed > icebase, icebase, bed)

    # anywhere bed is above surface, set equal to surface
    bed = xr.where(bed > surface, surface, bed)

    # anywhere icebase is above surface, set equal to surface
    icebase = xr.where(icebase > surface, surface, icebase)

    # check it worked
    assert np.all(surface - icebase) >= 0
    assert np.all(surface - bed) >= 0
    assert np.all(icebase - bed) >= 0

    # set densities
    air_density = 1
    ice_density = 917
    water_density = 1025
    rock_density = 2670

    log.debug("calculating terrain mass effect components")
    # make prism layers for terrain mass effect components
    ice_surface_prisms = invert4geom_utils.grids_to_prisms(
        surface=surface,
        reference=0,
        density=xr.where(
            surface >= 0,
            ice_density - air_density,
            air_density - ice_density
        ),
    )

    water_surface_prisms = invert4geom_utils.grids_to_prisms(
        surface=icebase,
        reference=0,
        density=xr.where(
            icebase >= 0,
            water_density - ice_density,
            ice_density - water_density
        ),
    )

    rock_surface_prisms = invert4geom_utils.grids_to_prisms(
        surface=bed,
        reference=0,
        density=xr.where(
            bed >= 0,
            rock_density - water_density,
            water_density - rock_density,
        ),
    )

    # calculate terrain mass effect components
    coords = (grav_df.easting, grav_df.northing, grav_df.ellipsoidal_height)
    grav_df["ice_surface_grav"] = ice_surface_prisms.prism_layer.gravity(
        coordinates = coords,
        field="g_z",
        progressbar=progressbar,
    )
    grav_df["water_surface_grav"] = water_surface_prisms.prism_layer.gravity(
        coordinates = coords,
        field="g_z",
        progressbar=progressbar,
    )
    grav_df["rock_surface_grav"] = rock_surface_prisms.prism_layer.gravity(
        coordinates = coords,
        field="g_z",
        progressbar=progressbar,
    )

    # calculate anomalies
    grav_df["terrain_mass_effect"] = grav_df.ice_surface_grav + grav_df.water_surface_grav + grav_df.rock_surface_grav
    grav_df["partial_topo_free_disturbance"] = grav_df.gravity_disturbance - grav_df.ice_surface_grav - grav_df.water_surface_grav
    grav_df["topo_free_disturbance"] = grav_df.partial_topo_free_disturbance - grav_df.rock_surface_grav

    # calculate gravity misfit, regional and residual
    # requires columns "starting_gravity" and "gravity_anomaly"
    grav_df["starting_gravity"] = grav_df.rock_surface_grav
    grav_df["gravity_anomaly"] = grav_df.partial_topo_free_disturbance

    log.debug("calculating gravity misfit and regional/residual components")

    try:
        grav_df = regional.regional_separation(
            grav_df=grav_df,
            constraints_df=constraints_df,
            **regional_grav_kwargs,
        )
    except Exception as e:
        log.error(e)
        log.error("Error in regional separation")

        grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity
        grav_df["reg"] = np.nan
        grav_df["res"] = np.nan

    log.debug("calculating stats for each anomaly type")
    # create mask for ice shelf
    grav_grid = grav_df.set_index(["northing","easting"]).to_xarray()
    grav_grid["ice_shelf_mask"] = utils.mask_from_shp(
        shapefile=gdf,
        xr_grid=grav_grid.partial_topo_free_disturbance,
        invert=False,
    ).rename("ice_shelf_mask")

    # subset data to ice shelf
    grav_df = grav_grid.to_dataframe().reset_index()

    if fname is not None:
        grav_df.to_csv(
            f"{fname}_grav_anomalies.csv.gz",
            sep=",",
            na_rep="",
            header=True,
            index=False,
            encoding="utf-8",
            compression="gzip",
        )
    if plot:
        if save_plot:
            fname = fname
        else:
            fname = None
        plot_grav_anomalies(
            gdf,
            grav_df,
            constraints_df,
            fname,
        )


def gravity_anomalies(
    ice_shelves: gpd.GeoDataFrame,
    regional_grav_kwargs: dict,
    buffer: float = 10e3,
    spacing: float = 10e3,
    progressbar: bool = False,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
) -> None:
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )
    for i, row in pbar:
        pbar.set_description(f"Processing {row.NAME}")
        gravity_anomalies_single(
            row,
            regional_grav_kwargs=regional_grav_kwargs,
            constraints_df=pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz"),
            buffer=buffer,
            spacing=spacing,
            progressbar=progressbar,
            fname=file_path+row.NAME,
            plot=plot,
            save_plot=save_plot,
        )


def load_grav_anomalies(
    ice_shelves: gpd.GeoDataFrame,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )

    for i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")

        try:
            constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
        except FileNotFoundError as e:
            log.error(e)
            log.error(f"Failed to load constraints for {row.NAME}")
            continue

        try:
            grav_df = pd.read_csv(f"{file_path}{row.NAME}_grav_anomalies.csv.gz")
        except FileNotFoundError as e:
            log.error(e)
            log.error(f"Failed to load gravity anomalies for {row.NAME}")
            continue

        # subset data to ice shelf
        grav_df_subset=grav_df[grav_df.ice_shelf_mask==True]

        # calculate stats on following columns
        anoms = [
            "gravity_disturbance",
            "partial_topo_free_disturbance",
            "topo_free_disturbance",
            "starting_gravity",
            # "gravity_anomaly",
            # "misfit",
            "reg",
            "res",
            "error",
        ]
        stats_df = pd.DataFrame(
            {
                "root_mean_square": [utils.rmse(grav_df_subset[a]) for a in anoms],
                "mean_absolute": [np.abs(grav_df_subset[a]).mean() for a in anoms],
                "stdev": [grav_df_subset[a].std() for a in anoms],
            },
            index=anoms,
        )

        for row_name, _ in stats_df.iterrows():
            for col_name in stats_df.columns:
                gdf.loc[i, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]


        if plot:
            if save_plot:
                fname = file_path+row.NAME
            else:
                fname = None
            plot_grav_anomalies(
                gdf,
                grav_df=grav_df,
                constraints_df=constraints_df,
                fname=fname,
            )

    return gdf


def plot_grav_anomalies(
    ice_shelf: gpd.GeoSeries,
    grav_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    fname: str | None = None,
):
    name = ice_shelf.iloc[0].NAME

    grav_grid = grav_df.set_index(["northing","easting"]).to_xarray()
    grav_grid = grav_grid.where(grav_grid.ice_shelf_mask==True, np.nan)

    anoms = [
        "gravity_disturbance",
        "topo_free_disturbance",
        "starting_gravity",
        "reg",
        "res",
    ]
    anom_titles = [
        "Gravity disturbance",
        "Topo-free disturbance",
        "Starting gravity",
        "Regional misfit",
        "Residual misfit",
    ]
    grids = [grav_grid[a] for a in anoms]

    grids.insert(0, grav_grid.error)

    cmaps = (
        ["thermal"]+
        ["viridis"]*(len(anoms)-1)+
        ["balance+h0"]
    )
    reverse_cpts = [False]*(len(grids)-1)+[True]
    insets = [True]+[False]*(len(grids)-1)
    cbar_labels = (
        [f"uncertainty; RMSE: {int(utils.rmse(grav_grid.error))} mGal"]+
        [f"stdev: {round(grav_df[grav_df.ice_shelf_mask==True][a].std(),0)} mGal" for a in anoms]
    )
    offshore_points = constraints_df[constraints_df.onshore==False]
    if offshore_points.empty:
        offshore_points = None
    point_sets = (
        [constraints_df]+
        [offshore_points]*(len(grids)-1)
    )
    scalebars = [False]+[True]+[False]*(len(grids)-2)
    titles = (
        ["AntGG gravity uncertainty"]+
        anom_titles
    )

    try:
        fig = maps.subplots(
            grids,
            fig_title=f"{name.replace("_", " ")} Ice Shelf",
            cmaps=cmaps,
            reverse_cpts=reverse_cpts,
            insets=insets,
            robust=True,
            coast=True,
            hist=True,
            cbar_labels=cbar_labels,
            titles=titles,
            point_sets=point_sets,
            cbar_font="15p,Helvetica,black",
            points_style="c1p",
            scalebars=scalebars,
        )
        fig.show()
        if fname is not None:
            fig.savefig(f"{fname}_grav_anomalies.png")

    except Exception as e:
        log.error(e)
        log.error(f"Failed to plot {name}")


def polygons_to_points(
    polygons: gpd.GeoDataFrame,
    spacing: float = 100,
) -> pd.DataFrame:
    """
    Convert a geodataframe of polygons to a grid of points with a specified spacing.

    Parameters
    ----------
    polygons : gpd.GeoDataFrame
        Geodataframe of polygons to convert to points
    spacing : float, optional
        Spacing between points, by default 100

    Returns
    -------
    pd.DataFrame
        Dataframe of points with columns "easting", "northing", and "geometry"
    """
    points_list = []
    for _, row in polygons.iterrows():
        points_list.append(polygon_to_points(row, spacing=spacing))

    return pd.concat(points_list)


def polygon_to_points(
    polygon: gpd.GeoSeries,
    spacing: float = 100,
) -> pd.DataFrame:
    """
    Convert a polygon shapefile to a grid of points with a specified spacing. Also
    include the vertices of the polygon.

    Parameters
    ----------
    polygon : gpd.GeoSeries
        Polygon shapefile to convert to points
    spacing : float, optional
        Spacing between points, by default 100

    Returns
    -------
    pd.DataFrame
        Dataframe of points with columns "easting", "northing", and "geometry"
    """

    # get bounds of polygon
    bounds = polygon.geometry.bounds

    # create grid of points
    x_coords = np.arange(bounds[0], bounds[2], spacing)
    y_coords = np.arange(bounds[1], bounds[3], spacing)
    points = [shapely.geometry.Point(x, y) for x in x_coords for y in y_coords]

    # Filter points within the polygon
    points_in_polygon = [point for point in points if polygon.geometry.contains(point)]

    # add vertices of polygon
    points_in_polygon.extend([shapely.geometry.Point(x, y) for (x,y) in polygon.geometry.exterior.coords])

    # add points along edges
    points_in_polygon.extend([shapely.geometry.Point(x, y) for (x,y) in polygon.geometry.segmentize(spacing).exterior.coords])

    # convert to geodataframe
    points_gdf = gpd.GeoDataFrame(geometry=points_in_polygon)

    # add easting and northing columns
    points_gdf["easting"] = [p.x for p in points_gdf.geometry]
    points_gdf["northing"] = [p.y for p in points_gdf.geometry]

    return points_gdf


def plot_ice_shelf_info(
    ice_shelf: gpd.GeoSeries,
    grav_df: pd.DataFrame | None = None,
    constraints_df: pd.DataFrame | None = None,
    min_dist: xr.DataArray | None = None,
    fig_path: str | None = None,
):
    name = ice_shelf.NAME.iloc[0]

    grd = grav_df.set_index(["northing","easting"]).to_xarray()
    region = vd.get_region((grd.gravity_disturbance.easting.values, grd.gravity_disturbance.northing.values))

    grav_grid = grav_df.set_index(["northing","easting"]).to_xarray()
    grav_grid = grav_grid.where(grav_grid.ice_shelf_mask==True, np.nan)

    anoms = [
        "gravity_disturbance",
        "topo_free_disturbance",
        "starting_gravity",
        "reg",
        "res",
    ]
    anom_titles = [
        "Gravity disturbance",
        "Topo-free disturbance (misfit)",
        "Starting gravity",
        "Regional misfit",
        "Residual misfit",
    ]
    grids = [grav_grid[a] for a in anoms]


    grids.insert(0, min_dist)
    grids.insert(1, grav_grid.error)

    cmaps = (
        ["dense"]+
        ["thermal"]+
        ["viridis"]*(len(anoms)-1)+
        ["balance+h0"]
    )
    # reverse_cpts = [False]*(len(grids)-1)+[True]
    insets = [True]+[False]*(len(grids)-1)
    cbar_labels = (
        [f"median: {int(min_dist.median())} m"]+
        [f"uncertainty; RMSE: {int(utils.rmse(grav_grid.error))} mGal"]+
        [f"stdev: {round(grav_df[grav_df.ice_shelf_mask==True][a].std(),0)} mGal" for a in anoms]
    )
    offshore_points = constraints_df[constraints_df.onshore==False]
    if offshore_points.empty:
        offshore_points = None
    point_sets = (
        [constraints_df]+
        [offshore_points]*(len(grids)-1)
    )
    scalebars = [False]+[True]+[False]*(len(grids)-2)
    titles = (
        ["Constraint proximity"]+
        ["AntGG gravity uncertainty"]+
        anom_titles
    )

    try:
        fig = maps.subplots(
            grids,
            region=region,
            fig_title=f"{name.replace("_", " ")} Ice Shelf",
            cmaps=cmaps,
            # reverse_cpts=reverse_cpts,
            insets=insets,
            robust=True,
            coast=True,
            coast_pen="0.6p,salmon",
            coast_version="measures-v2", # default is depoorter-2013
            hist=True,
            cbar_labels=cbar_labels,
            titles=titles,
            point_sets=point_sets,
            cbar_font="15p,Helvetica,black",
            points_style="c1p",
            scalebars=scalebars,
        )
        fig.show()
        if fig_path is not None:
            fig.savefig(f"{fig_path}{name}_info.png")

    except Exception as e:
        log.error(e)
        log.error(f"Failed to plot {name}")


def load_ice_shelf_info_single(
    ice_shelf: gpd.GeoSeries,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    # convert to geodataframe
    gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

    name = gdf.NAME.iloc[0]

    constraints_df = pd.read_csv(f"{file_path}{name}_constraints.csv.gz")
    grav_df = pd.read_csv(f"{file_path}{name}_grav_anomalies.csv.gz")
    min_dist = xr.open_dataarray(f"{file_path}{name}_min_dist.nc")

    gdf.loc[gdf.index, f"median_constraint_distance"] = min_dist.median().values

    # sample the constraint distance grid into the gravity dataframe
    grav_df = profiles.sample_grids(
        grav_df,
        min_dist,
        sampled_name="constraint_distance",
        verbose="q",
    )

    grav_df["residual_constraint_proximity_ratio"] = grav_df.res / (1/grav_df.constraint_distance)
    grav_df["regional_constraint_proximity_ratio"] = grav_df.reg / (1/grav_df.constraint_distance)

    # subset data to ice shelf
    grav_df_subset=grav_df[grav_df.ice_shelf_mask==True]

    # calculate stats on following columns
    anoms = [
        "gravity_disturbance",
        "partial_topo_free_disturbance",
        "topo_free_disturbance",
        "starting_gravity",
        # "gravity_anomaly",
        # "misfit",
        "reg",
        "res",
        "error",
        "residual_constraint_proximity_ratio",
        "regional_constraint_proximity_ratio",
    ]
    stats_df = pd.DataFrame(
        {
            "rms": [utils.rmse(grav_df_subset[a]) for a in anoms],
            # "mean_absolute": [np.abs(grav_df_subset[a]).mean() for a in anoms],
            "stdev": [grav_df_subset[a].std() for a in anoms],
        },
        index=anoms,
    )

    for row_name, _ in stats_df.iterrows():
        for col_name in stats_df.columns:
            gdf.loc[gdf.index, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]

    if plot:
        if save_plot:
            fig_path = file_path
        else:
            fig_path = None

        plot_ice_shelf_info(
            gdf,
            grav_df=grav_df,
            constraints_df=constraints_df,
            min_dist=min_dist,
            fig_path=fig_path,
        )

    return gdf


def load_ice_shelf_info(
    ice_shelves: gpd.GeoSeries,
    file_path: str | None = None,
    plot: bool = False,
    save_plot: bool = False,
):
    gdf = ice_shelves.copy()

    pbar = tqdm(
        gdf.iterrows(),
        desc="Ice Shelves",
        total=len(gdf),
    )


    shelves = []
    for i, row in pbar:
        pbar.set_description(f"Loading data for {row.NAME}")

        shelves.append(
            load_ice_shelf_info_single(
                row,
                file_path=file_path,
                plot=plot,
                save_plot=save_plot,
        ))

        # for row_name, _ in stats_df.iterrows():
        #     for col_name in stats_df.columns:
        #         gdf.loc[gdf.index, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]

    return pd.concat(shelves)


def add_shelves_to_ensembles(
    x: str,
    y: str,
    ice_shelves: gpd.GeoDataFrame,
    shelves_to_label: list[str] | None = None,
    ax: typing.Any | None = None,
    col_to_add_to_label: str | None = None,
    legend:bool=True,
    legend_cols:int = 2,
    legend_loc:str = "center left",
    legend_bbox_to_anchor:tuple[float, float] = (1.3, 0.5),
):
    gdf = ice_shelves.copy()

    if shelves_to_label is None:
        shelves_to_label = gdf.NAME.unique().tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        xlims = None
        ylims = None
    else:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()

    if xlims is not None and ylims is not None:
        gdf[x] = np.where(gdf[x]<min(xlims), min(xlims), gdf[x])
        gdf[x] = np.where(gdf[x]>max(xlims), max(xlims), gdf[x])
        gdf[y] = np.where(gdf[y]<min(ylims), min(ylims), gdf[y])
        gdf[y] = np.where(gdf[y]>max(ylims), max(ylims), gdf[y])

    texts = []

    for ind, (i, row) in enumerate(gdf.iterrows()):
        if col_to_add_to_label is not None:
            add_to_label = f": {round(row[col_to_add_to_label])} m"
        else:
            add_to_label = ""
        # plot inverted shelves as red stars and red labels
        if row.NAME in inverted_shelves:
            ax.scatter(
                row[x],
                row[y],
                color="red",
                marker="*",
                s=12,
                label=f"{ind+1}) {row.NAME.replace('_', ' ')}{add_to_label}",
                clip_on=False,
                zorder=10,
            )
            texts.append(
                ax.text(
                    row[x],
                    row[y],
                    f"{ind+1}",
                    fontsize=8,
                    color="red",
                    fontweight="normal",
                    path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                )
            )
        else:
            # plot other shelves as black dots and black labels
            if row.NAME in shelves_to_label:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=2,
                    label=f"{ind+1}) {row.NAME.replace('_', ' ')}{add_to_label}",
                    clip_on=False,
                    zorder=10,
                )
                texts.append(
                    ax.text(
                        row[x],
                        row[y],
                        f"{ind+1}",
                        fontsize=8,
                        color="black",
                        fontweight="normal",
                        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                    )
                )
            else:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=2,
                    clip_on=False,
                    zorder=10,
                )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="k", lw=.8),
        ax=ax,
        expand=(1.2, 1.2)
    )

    if legend:
        leg = ax.legend(
            loc=legend_loc,
            title="Ice Shelves",
            bbox_to_anchor=legend_bbox_to_anchor,
            markerscale=0,
            fontsize=8,
            ncol=legend_cols,
        )
        for handle, text in zip(leg.legend_handles, leg.get_texts()):
            text.set_color(handle.get_facecolor()[0])


def ensemble_scatterplot(
    x: str,
    y: str,
    ice_shelves: gpd.GeoDataFrame,
    label_shelves: bool = True,
    shelves_to_label: list[str] | None = None,
    col_to_add_to_label: str | None = None,
    legend:bool=True,
    legend_cols:int = 2,
    legend_loc:str = "center left",
    legend_bbox_to_anchor:tuple[float, float] = (1.3, 0.5),
    xlims: tuple[float, float] | None = None,
    ylims: tuple[float, float] | None = None,
    logx=False,
    logy=False,
):
    gdf = ice_shelves.copy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    if shelves_to_label is None:
        shelves_to_label = gdf.NAME.unique().tolist()

    if xlims is not None:
        gdf = gdf[(gdf[x] >= xlims[0]) & (gdf[x] <= xlims[1])]
    if ylims is not None:
        gdf = gdf[(gdf[y] >= ylims[0]) & (gdf[y] <= ylims[1])]

    texts = []

    for ind, (i, row) in enumerate(gdf.iterrows()):
        if col_to_add_to_label is not None:
            add_to_label = f": {round(row[col_to_add_to_label])} m"
        else:
            add_to_label = ""
        # plot inverted shelves as red stars and red labels
        if row.NAME in inverted_shelves:
            ax.scatter(
                row[x],
                row[y],
                color="red",
                marker="*",
                s=12,
                label=f"{ind+1} {row.NAME.replace('_', ' ')}{add_to_label}",
            )
            if label_shelves:
                texts.append(
                    ax.text(
                        row[x],
                        row[y],
                        f"{ind+1}",
                        fontsize=8,
                        color="red",
                        fontweight="normal",
                        path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                    )
                )
        else:
            # plot other shelves as black dots and black labels
            if row.NAME in shelves_to_label:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=2,
                    label=f"{ind+1} {row.NAME.replace('_', ' ')}{add_to_label}",
                )
                if label_shelves:
                    texts.append(
                        ax.text(
                            row[x],
                            row[y],
                            f"{ind+1}",
                            fontsize=8,
                            color="black",
                            fontweight="normal",
                            path_effects=[patheffects.withStroke(linewidth=1.5, foreground="white")],
                        )
                    )
            else:
                ax.scatter(
                    row[x],
                    row[y],
                    color="black",
                    s=2,
                )

    # ax.set_xlim(xlims)
    # ax.set_ylim(ylims)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="k", lw=.8),
        ax=ax,
        expand=(1.2, 1.2)
    )

    if label_shelves:
        if legend:
            leg = ax.legend(
                loc=legend_loc,
                title="Ice Shelves",
                bbox_to_anchor=legend_bbox_to_anchor,
                markerscale=0,
                fontsize=8,
                ncol=legend_cols,
            )
            for handle, text in zip(leg.legend_handles, leg.get_texts()):
                text.set_color(handle.get_facecolor()[0])

    return fig








# def ice_shelf_stats(
#     ice_shelves: gpd.GeoSeries,
#     file_path: str | None = None,
#     plot: bool = False,
#     save_plot: bool = False,
#     constraint_stats: bool = True,
#     gravity_stats: bool = True,
# ) -> None:
#     gdf = ice_shelves.copy()

#     pbar = tqdm(
#         gdf.iterrows(),
#         desc="Ice Shelves",
#         total=len(gdf),
#     )
#     shelves = []
#     for i, row in pbar:
#         pbar.set_description(f"Loading {row.NAME}")

#         try:
#             shelf = ice_shelf_stats_single(
#                 row,
#                 file_path=file_path,
#                 plot=plot,
#                 save_plot=save_plot,
#                 constraint_stats=constraint_stats,
#                 gravity_stats=gravity_stats,
#             )
#         except FileNotFoundError as e:
#             log.error(e)
#             log.error("Failed to calculate stats for %s", row.NAME)
#             shelf = None
#         shelves.append(shelf)
#     # merge rows into geodataframe
#     return pd.concat(shelves)


# def ice_shelf_stats_single(
#     ice_shelf: gpd.GeoSeries,
#     file_path: str | None = None,
#     plot: bool = False,
#     save_plot: bool = False,
#     constraint_stats: bool = True,
#     gravity_stats: bool = True,
# ) -> gpd.GeoDataFrame:
#     """
#     _summary_

#     Parameters
#     ----------
#     ice_shelf : gpd.GeoSeries
#         _description_
#     file_path : str | None, optional
#         _description_, by default None
#     plot : bool, optional
#         _description_, by default False

#     Returns
#     -------
#     gpd.GeoDataFrame
#         _description_
#     """

#     # convert to geodataframe
#     gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

#     name = gdf.NAME.iloc[0]

#     grav_df = None
#     min_dist = None
#     constraints_df = None

#     if constraint_stats:
#         constraints_df = pd.read_csv(f"{file_path}{name}_constraints.csv.gz")
#         min_dist = xr.open_dataarray(f"{file_path}{name}_min_dist.nc")

#         gdf.loc[gdf.index, f"median_constraint_distance"] = min_dist.median().values

#     if gravity_stats:
#         grav_df = pd.read_csv(f"{file_path}{name}_grav_anomalies.csv.gz")
#         # subset data to ice shelf
#         grav_df_subset=grav_df[grav_df.ice_shelf_mask==True]


#         # calculate stats on following columns
#         anoms = [
#             "gravity_disturbance",
#             "partial_topo_free_disturbance",
#             "topo_free_disturbance",
#             "starting_gravity",
#             # "gravity_anomaly",
#             # "misfit",
#             "reg",
#             "res",
#             "error",
#         ]
#         stats_df = pd.DataFrame(
#             {
#                 "root_mean_square": [utils.rmse(grav_df_subset[a]) for a in anoms],
#                 "mean_absolute": [np.abs(grav_df_subset[a]).mean() for a in anoms],
#                 "stdev": [grav_df_subset[a].std() for a in anoms],
#             },
#             index=anoms,
#         )

#         for row_name, _ in stats_df.iterrows():
#             for col_name in stats_df.columns:
#                 gdf.loc[gdf.index, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]

#     if plot:
#         if save_plot:
#             fig_path = file_path
#         else:
#             fig_path = None
#         plot_ice_shelf_info_single(
#             gdf,
#             grav_df=grav_df,
#             min_dist=min_dist,
#             constraints_df=constraints_df,
#             fig_path=fig_path,
#         )

#     return gdf


# def ice_shelf_save_files_single(
#     ice_shelf: gpd.GeoSeries,
#     bedmap_version: str = "all",
#     grav_spacing: float = 10e3,
#     constraint_spacing: float = 1e3,
#     buffer: float = 10e3,
#     grounded_as_points: bool = False,
#     file_path: str | None = None,
#     plot: bool = False,
#     save_constraints: bool = True,
#     save_gravity: bool = True,
# ) -> None:
#     """
#     For a single ice shelf, download gravity, topography, and constraints data. With
#     these calculate a range of gravity anomalies and save the constraints to a csv, the
#     gravity data to a csv, and the calculated minimum distance between constraints and
#     nearest grid cells to a netcdf. Optionally, make a plot of these results and save
#     to a png.

#     Parameters
#     ----------
#     ice_shelf : gpd.GeoSeries
#         _description_
#     bedmap_version : str, optional
#         _description_, by default "all"
#     grav_spacing : float, optional
#         _description_, by default 10e3
#     constraint_spacing : float, optional
#         _description_, by default 1e3
#     buffer : float, optional
#         _description_, by default 10e3
#     file_path : str | None, optional
#         _description_, by default None
#     plot : bool, optional
#         _description_, by default False
#     """
#     gdf = ice_shelf.copy()

#     name = gdf.NAME

#     min_dist = None
#     constraints_df = None
#     grav_df = None

#     if save_constraints:
#         # calculate distance to constraints
#         try:
#             gdf, _, min_dist, constraints_df = ice_shelf_constraints_distance_single(
#                 gdf,
#                 bedmap_version=bedmap_version,
#                 spacing=constraint_spacing,
#                 buffer=buffer,
#                 grounded_as_points=grounded_as_points,
#                 plot=plot,
#                 fname=file_path+name,
#             )
#             gdf = gdf.iloc[0]
#         except Exception as e:
#             log.error(e)
#             log.error(f"Failed to calculate constraints for {name}")

#     if save_gravity:
#         # load constraints df
#         constraints_df = pd.read_csv(f"{file_path}{name}_constraints.csv.gz")

#         # calculate gravity anomalies
#         gdf, grav_df = ice_shelf_grav_stats_single(
#             gdf,
#             constraints_df=constraints_df,
#             regional_grav_kwargs=dict(
#                 method="constraints",
#                 grid_method="pygmt",
#                 tension_factor=0.25,
#             ),
#             spacing = grav_spacing,
#             plot=False,
#             fname=file_path+name
#         )

#         if plot:
#             plot_ice_shelf_info_single(
#                 gdf,
#                 grav_df=grav_df,
#                 min_dist=min_dist,
#                 constraints_df=constraints_df,
#                 fig_path=file_path,
#             )


# def ice_shelf_save_files(
#     ice_shelves: gpd.GeoSeries,
#     bedmap_version: str = "all",
#     grav_spacing: float = 10e3,
#     constraint_spacing: float = 1e3,
#     buffer: float = 10e3,
#     grounded_as_points: bool = False,
#     file_path: str | None = None,
#     plot: bool = False,
#     save_constraints: bool = True,
#     save_gravity: bool = True,
# ) -> None:
#     """
#     For each ice shelf geometry provided (row of a geodataframe), download gravity,
#     topography, and constraints data. With these calculate a range of gravity anomalies
#     and save the constraints to a csv, the gravity data to a csv, and the calculated
#     minimum distance between constraints and nearest grid cells to a netcdf. Optionally,
#     make plots of these results and save to a png.

#     Parameters
#     ----------
#     ice_shelves : gpd.GeoSeries
#         _description_
#     bedmap_version : str, optional
#         _description_, by default "all"
#     grav_spacing : float, optional
#         _description_, by default 10e3
#     constraint_spacing : float, optional
#         _description_, by default 1e3
#     buffer : float, optional
#         _description_, by default 10e3
#     file_path : str | None, optional
#         _description_, by default None
#     plot : bool, optional
#         _description_, by default False
#     """
#     gdf = ice_shelves.copy()

#     pbar = tqdm(
#         gdf.iterrows(),
#         desc="Ice Shelves",
#         total=len(gdf),
#     )
#     # shelves = []
#     for i, row in pbar:
#     # for i, row in gdf.iterrows():
#         pbar.set_description(f"Processing {row.NAME}")

#         try:
#             _ = ice_shelf_save_files_single(
#                 row,
#                 buffer=buffer,
#                 grav_spacing=grav_spacing,
#                 constraint_spacing=constraint_spacing,
#                 bedmap_version=bedmap_version,
#                 grounded_as_points=grounded_as_points,
#                 plot=plot,
#                 file_path=file_path,
#                 save_constraints=save_constraints,
#                 save_gravity=save_gravity,
#             )
#         except ValueError as e:
#             log.error(e)
#             log.error(f"Error with {row.NAME} ice shelf")
#             continue

#         # add new columns to geodataframe
#         # for row_name, _ in stats_df.iterrows():
#         #     for col_name in stats_df.columns:
#         #         gdf.loc[i, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]
#         # shelves.append(shelf)

#     # merge rows into geodataframe
#     # return pd.concat(shelves)


# def ice_shelf_load_plot_info(
#     ice_shelves: gpd.GeoSeries,
#     file_path: str | None = None,
# ):
#     gdf = ice_shelves.copy()

#     pbar = tqdm(
#         gdf.iterrows(),
#         desc="Ice Shelves",
#         total=len(gdf),
#     )
#     shelves = []
#     for i, row in pbar:
#         pbar.set_description(f"Processing {row.NAME}")

#         grav_df = pd.read_csv(f"{file_path}{row.NAME}_grav_anomalies.csv.gz")
#         constraints_df = pd.read_csv(f"{file_path}{row.NAME}_constraints.csv.gz")
#         min_dist = xr.open_dataarray(f"{file_path}{row.NAME}_min_dist.nc")

#         plot_ice_shelf_info_single(
#             gdf,
#             grav_df=grav_df,
#             min_dist=min_dist,
#             constraints_df=constraints_df,
#             fig_path=file_path,
#         )


# def plot_ice_shelf_info_single(
#     ice_shelf: gpd.GeoSeries,
#     grav_df: pd.DataFrame | None = None,
#     min_dist: xr.DataArray | None = None,
#     constraints_df: pd.DataFrame | None = None,
#     fig_path: str | None = None,
# ):
#     name = ice_shelf.NAME.iloc[0]

#     if grav_df is None:
#         fig = maps.plot_grd(
#             min_dist,
#             title=f"{name.replace("_", " ")} Ice Shelf",
#             cbar_label=f"Constraint median distance; {int(min_dist.median())} (m)",
#             cmap="dense",
#             hist=True,
#             inset=True,
#             robust=True,
#             scalebar=True,
#             coast=True,
#             points=constraints_df,
#             points_style="c1p",
#         )
#         fig.show()
#         return

#     grav_df_subset = grav_df[grav_df.ice_shelf_mask==True]
#     grav_grid = grav_df_subset.set_index(["northing","easting"]).to_xarray()
#     anoms = [
#         "gravity_disturbance",
#         "topo_free_disturbance",
#         "starting_gravity",
#         "reg",
#         "res",
#     ]
#     anom_titles = [
#         "Gravity disturbance",
#         "Topo-free disturbance",
#         "Starting gravity",
#         "Regional misfit",
#         "Residual misfit",
#     ]
#     grids = [grav_grid[a] for a in anoms]

#     if (constraints_df is None) or (min_dist is None):
#         grids.insert(0, grav_grid.error)

#         cmaps = (
#             ["thermal"]+
#             ["viridis"]*(len(anoms)-1)+
#             ["balance+h0"]
#         )
#         reverse_cpts = [False]*(len(grids)-1)+[True]
#         insets = [True]+[False]*(len(grids)-1)
#         cbar_labels = (
#             [f"uncertainty; RMSE: {int(utils.rmse(grav_grid.error))} mGal"]+
#             [f"stdev: {round(grav_df_subset[a].std(),0)} mGal" for a in anoms]
#         )
#         point_sets = None
#         scalebars = [True]+[False]*(len(grids)-1)
#         titles = (
#             ["AntGG gravity uncertainty"]+
#             anom_titles
#         )
#     else:
#         grids.insert(0, min_dist)
#         grids.insert(1, grav_grid.error)

#         cmaps = (
#             ["dense"]+
#             ["thermal"]+
#             ["viridis"]*(len(anoms)-1)+
#             ["balance+h0"]
#         )
#         reverse_cpts = [False]*(len(grids)-1)+[True]
#         insets = [True]+[False]*(len(grids)-1)
#         cbar_labels = (
#             [f"median: {int(min_dist.median())} m"]+
#             [f"uncertainty; RMSE: {int(utils.rmse(grav_grid.error))} mGal"]+
#             [f"stdev: {round(grav_df_subset[a].std(),0)} mGal" for a in anoms]
#         )
#         offshore_points = constraints_df[constraints_df.onshore==False]
#         if offshore_points.empty:
#             offshore_points = None
#         point_sets = (
#             [constraints_df]+
#             [offshore_points]*(len(grids)-1)
#         )
#         scalebars = [False]+[True]+[False]*(len(grids)-2)
#         titles = (
#             ["Constraint proximity"]+
#             ["AntGG gravity uncertainty"]+
#             anom_titles
#         )

#     try:
#         fig = maps.subplots(
#             grids,
#             fig_title=f"{name.replace("_", " ")} Ice Shelf",
#             cmaps=cmaps,
#             reverse_cpts=reverse_cpts,
#             insets=insets,
#             robust=True,
#             coast=True,
#             hist=True,
#             cbar_labels=cbar_labels,
#             titles=titles,
#             point_sets=point_sets,
#             cbar_font="15p,Helvetica,black",
#             points_style="c1p",
#             scalebars=scalebars,
#         )
#         fig.show()
#         if fig_path is not None:
#             fig.savefig(f"{fig_path}{name}_info.png")

#     except Exception as e:
#         log.error(e)
#         log.error(f"Failed to plot {name}")



# def ice_shelf_grav_stats_single(
#     ice_shelf: gpd.GeoSeries,
#     regional_grav_kwargs: dict,
#     constraints_df: pd.DataFrame,
#     buffer: float = 10e3,
#     spacing: float = 10e3,
#     fname: str = None,
#     plot: bool = False,
# ) -> tuple[pd.DataFrame, pd.DataFrame]:

#     # convert to geodataframe
#     gdf = gpd.GeoDataFrame(ice_shelf).T.set_geometry("geometry")

#     # define region around each ice shelf with 10km buffer
#     reg = utils.region_to_bounding_box(gdf.iloc[0].geometry.bounds)
#     reg = vd.pad_region(reg, buffer)

#     log.debug("calculated region %s", reg)
#     log.debug("fetching, subsetting, and resampling datasets")

#     # download gravity data
#     eigen_gravity = fetch.gravity(
#         version="eigen",
#         spacing=spacing,
#         region=reg,
#         registration="g",
#         verbose="q",
#     )
#     antgg_disturbance = fetch.gravity(
#         version="antgg-2021",
#         anomaly_type="DG",
#         spacing=spacing,
#         region=reg,
#         registration="g",
#         verbose="q",
#     )
#     antgg_uncertainty = fetch.gravity(
#         version="antgg-2021",
#         anomaly_type="Err",
#         spacing=spacing,
#         region=reg,
#         registration="g",
#         verbose="q",
#     )

#     # download topography
#     surface = fetch.bedmap2(
#         layer="surface",
#         spacing=spacing,
#         region=reg,
#         fill_nans=True,
#         reference="ellipsoid",  # converts to be referenced to the ellipsoid
#         verbose="q",
#     ).rename({"x":"easting", "y":"northing"})
#     icebase = fetch.bedmap2(
#         layer="icebase",
#         spacing=spacing,
#         region=reg,
#         fill_nans=True,  # fills nans over ocean with 0's (while still reference to the geoid)
#         reference="ellipsoid",  # converts to be referenced to the ellipsoid
#         verbose="q",
#     ).rename({"x":"easting", "y":"northing"})
#     bed = fetch.bedmap2(
#         layer="bed",
#         spacing=spacing,
#         region=reg,
#         fill_nans=True,  # fills nans over ocean with 0's (while still reference to the geoid)
#         reference="ellipsoid",  # converts to be referenced to the ellipsoid
#         verbose="q",
#     ).rename({"x":"easting", "y":"northing"})

#     # make gravity dataframe
#     grav_df = xr.merge(
#         [
#             eigen_gravity.gravity,
#             antgg_disturbance.gravity_disturbance,
#             antgg_disturbance.ellipsoidal_height,
#             antgg_uncertainty.error,
#         ]
#     ).to_dataframe().reset_index().rename(columns={"x":"easting", "y":"northing"})

#     # check layers to cross
#     # anywhere bed is above icebase, set equal to icebase
#     bed = xr.where(bed > icebase, icebase, bed)

#     # anywhere bed is above surface, set equal to surface
#     bed = xr.where(bed > surface, surface, bed)

#     # anywhere icebase is above surface, set equal to surface
#     icebase = xr.where(icebase > surface, surface, icebase)

#     # check it worked
#     assert np.all(surface - icebase) >= 0
#     assert np.all(surface - bed) >= 0
#     assert np.all(icebase - bed) >= 0

#     # set densities
#     air_density = 1
#     ice_density = 917
#     water_density = 1025
#     rock_density = 2670

#     log.debug("calculating terrain mass effect components")
#     # make prism layers for terrain mass effect components
#     ice_surface_prisms = invert4geom_utils.grids_to_prisms(
#         surface=surface,
#         reference=0,
#         density=xr.where(
#             surface >= 0,
#             ice_density - air_density,
#             air_density - ice_density
#         ),
#     )

#     water_surface_prisms = invert4geom_utils.grids_to_prisms(
#         surface=icebase,
#         reference=0,
#         density=xr.where(
#             icebase >= 0,
#             water_density - ice_density,
#             ice_density - water_density
#         ),
#     )

#     rock_surface_prisms = invert4geom_utils.grids_to_prisms(
#         surface=bed,
#         reference=0,
#         density=xr.where(
#             bed >= 0,
#             rock_density - water_density,
#             water_density - rock_density,
#         ),
#     )

#     # calculate terrain mass effect components
#     coords = (grav_df.easting, grav_df.northing, grav_df.ellipsoidal_height)
#     grav_df["ice_surface_grav"] = ice_surface_prisms.prism_layer.gravity(
#         coordinates = coords,
#         field="g_z",
#         progressbar=False,
#     )
#     grav_df["water_surface_grav"] = water_surface_prisms.prism_layer.gravity(
#         coordinates = coords,
#         field="g_z",
#         progressbar=False,
#     )
#     grav_df["rock_surface_grav"] = rock_surface_prisms.prism_layer.gravity(
#         coordinates = coords,
#         field="g_z",
#         progressbar=False,
#     )

#     # calculate anomalies
#     grav_df["terrain_mass_effect"] = grav_df.ice_surface_grav + grav_df.water_surface_grav + grav_df.rock_surface_grav
#     grav_df["partial_topo_free_disturbance"] = grav_df.gravity_disturbance - grav_df.ice_surface_grav - grav_df.water_surface_grav
#     grav_df["topo_free_disturbance"] = grav_df.partial_topo_free_disturbance - grav_df.rock_surface_grav

#     # calculate gravity misfit, regional and residual
#     # requires columns "starting_gravity" and "gravity_anomaly"
#     grav_df["starting_gravity"] = grav_df.rock_surface_grav
#     grav_df["gravity_anomaly"] = grav_df.partial_topo_free_disturbance

#     gr
#     log.debug("calculating gravity misfit and regional/residual components")
#     try:
#         grav_df = regional.regional_separation(
#             grav_df=grav_df,
#             constraints_df=constraint_points,
#             **regional_grav_kwargs,
#         )
#     except:
#         log.error("Error in regional separation")
#         grav_df["misfit"] = grav_df.gravity_anomaly - grav_df.starting_gravity
#         grav_df["reg"] = np.nan
#         grav_df["res"] = np.nan

#     log.debug("calculating stats for each anomaly type")
#     # create mask for ice shelf
#     grav_grid = grav_df.set_index(["northing","easting"]).to_xarray()
#     grav_grid["ice_shelf_mask"] = utils.mask_from_shp(
#         shapefile=gdf,
#         xr_grid=grav_grid.partial_topo_free_disturbance,
#         invert=False,
#     ).rename("ice_shelf_mask")

#     # subset data to ice shelf
#     grav_df = grav_grid.to_dataframe().reset_index()
#     # grav_df["ice_shelf_mask"] = mask.to_dataframe().reset_index().ice_shelf_mask
#     grav_df_subset=grav_df[grav_df.ice_shelf_mask==True]


#     # calculate stats on following columns
#     anoms = [
#         "gravity_disturbance",
#         # "partial_topo_free_disturbance",
#         "topo_free_disturbance",
#         "starting_gravity",
#         # "gravity_anomaly",
#         # "misfit",
#         "reg",
#         "res",
#         "error",

#     ]
#     stats_df = pd.DataFrame(
#         {
#             "root_mean_square": [utils.rmse(grav_df_subset[a]) for a in anoms],
#             "mean_absolute": [np.abs(grav_df_subset[a]).mean() for a in anoms],
#             # "range": [sp.stats.iqr(grav_df_subset[a], rng=(0,100)) for a in anoms],
#             # "range_99": [sp.stats.iqr(grav_df_subset[a], rng=(1,99)) for a in anoms],
#             # "range_95": [sp.stats.iqr(grav_df_subset[a], rng=(5,95)) for a in anoms],
#             "stdev": [grav_df_subset[a].std() for a in anoms],
#         },
#         index=anoms,
#     )

#     if fname is not None:
#         grav_df.to_csv(
#             f"{fname}_grav_anomalies.csv.gz",
#             sep=",",
#             na_rep="",
#             header=True,
#             index=False,
#             encoding="utf-8",
#             compression="gzip",
#         )
#     if plot:
#         grav_grid = grav_df_subset.set_index(["northing","easting"]).to_xarray()
#         grids = [grav_grid[a] for a in anoms]
#         fig = maps.subplots(
#             grids,
#             fig_title=f"{gdf.iloc[0].NAME.replace("_", " ")} Gravity Anomalies",
#             # cpt_lims=utils.get_combined_min_max(grids, robust=True,),
#             robust=True,
#             coast=True,
#             inset=True,
#             hist=True,
#             cbar_label="mGal",
#             titles=anoms,
#             points=constraint_points[constraint_points.onshore==False],
#             points_style="c1p",
#         )
#         fig.show()
#         fig.savefig(f"{fname}_grav_anomalies.png")

#     for row_name, _ in stats_df.iterrows():
#         for col_name in stats_df.columns:
#             gdf.loc[gdf.index, f"{row_name}_{col_name}"] = stats_df.loc[row_name][col_name]

#     return gdf, grav_df




