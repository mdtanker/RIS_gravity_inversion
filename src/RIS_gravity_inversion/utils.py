from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import seaborn as sns
import verde as vd
from invert4geom import utils as inv_utils
from polartoolkit import maps, profiles, utils
from requests import get


def constraint_layout(
    num_constraints,
    shift_stdev,
    region=None,
    shapefile=None,
    plot=False,
):
    if shapefile is not None:
        bounds = gpd.read_file(shapefile).bounds
        region = [bounds.minx, bounds.maxx, bounds.miny, bounds.maxy]
        region = [x.values[0] for x in region]

    x = region[1] - region[0]
    y = region[3] - region[2]
    num_y = int(np.ceil((num_constraints / (x / y)) ** 0.5))

    fudge_factor = 0
    while True:
        num_x = int(np.ceil(num_constraints / num_y)) + fudge_factor

        # create regular grid, with set number of constraint points
        reg = vd.pad_region(region, -15e3)
        x = np.linspace(reg[0], reg[1], int(num_x * 1.1))
        y = np.linspace(reg[2], reg[3], int(num_y * 1.1))
        coords = np.meshgrid(x, y)

        # turn coordinates into dataarray
        da = vd.make_xarray_grid(
            coords,
            data=np.ones_like(coords[0]) * 1e3,
            data_names="upward",
            dims=("northing", "easting"),
        )
        # turn dataarray into dataframe
        df = vd.grid_to_table(da)

        # add randomness to the points
        rand = np.random.default_rng(seed=0)
        constraints = df.copy()
        constraints["northing"] = rand.normal(df.northing, shift_stdev)
        constraints["easting"] = rand.normal(df.easting, shift_stdev)

        # check whether points are inside or outside of shp
        if shapefile is not None:
            gdf = gpd.GeoDataFrame(
                constraints,
                geometry=gpd.points_from_xy(
                    x=constraints.easting, y=constraints.northing
                ),
                crs="EPSG:3031",
            )
            constraints["inside"] = gdf.within(
                gpd.read_file("../data/Ross_Sea_outline.shp").geometry[0]
            )
            constraints = constraints.drop(columns="geometry")
        else:
            constraints["inside"] = True

        # ensure all points are within region
        constraints = utils.points_inside_region(
            constraints, region, names=("easting", "northing")
        )

        # keep only set number of constraints
        try:
            constraints = constraints[constraints.inside].sample(
                n=num_constraints, random_state=0
            )
        except ValueError:
            fudge_factor += 0.1
        else:
            break

    if plot:
        fig = maps.basemap(
            fig_height=8,
            region=region,
        )

        fig.plot(
            x=constraints.easting,
            y=constraints.northing,
            style="c.1c",
            fill="black",
        )

        if shapefile is not None:
            fig.plot(
                shapefile,
                pen="0.2p,black",
            )

        fig.show()

    return constraints


def get_buffer_points(
    buffer_width=10e3,
    grid=None,
    mask=None,
    plot=False,
):
    """
    Create buffer zone of points around ice shelf border and grid with ice shelf masked.
    """
    # get buffered mask
    mask_buffer = mask.buffer(buffer_width)

    # mask grid inside of un-buffered mask
    grid_masked = utils.mask_from_shp(
        shapefile=mask,
        xr_grid=grid,
        masked=True,
    ).rename("upward")

    # mask grid outside of buffered mask
    grid_masked_buffer = utils.mask_from_shp(
        shapefile=mask_buffer,
        xr_grid=grid_masked,
        masked=True,
        invert=False,
    ).rename("upward")

    # create dataframes from grids
    grid_df_outside = vd.grid_to_table(grid_masked).dropna()
    grid_df_buffer = vd.grid_to_table(grid_masked_buffer).dropna()

    if plot is True:
        grid_df_buffer.plot.scatter(x="easting", y="northing")

    return grid_df_buffer, grid_df_outside, grid_masked


def create_starting_bed(
    buffer_and_inside_points,
    masked_bed,
    region,
    spacing,
    method="spline",
    damping=1e-50,
    tension=0,
    weights_col_name=None,
    icebase=None,
    surface=None,
    plot=False,
):
    """
    Create the interpolated bathymetry grid. Interpolate data from sparse constraints
    within the ice shelf area (constraints) and grid points within a buffer zone around
     the ice shelf. Then merge this grid with grid value from outside the ice shelf.
     Since the interpolation is only conducted with a thin buffer zone, instead of all
     of the points outside the shelf, the interpolation is much faster.
    """
    points = buffer_and_inside_points

    if method == "spline":
        weights = None if weights_col_name is None else points[weights_col_name]

        coords = (points.easting, points.northing)
        data = points.upward

        spline = inv_utils.best_spline_cv(
            coordinates=coords,
            data=data,
            weights=weights,
            dampings=damping,
        )

        inner_bed = spline.grid(
            region=region,
            spacing=spacing,
        ).scalars

        inner_bed = inner_bed.assign_attrs(damping=spline.damping_)

    elif method == "surface":
        blocked = pygmt.blockmean(
            data=points[["easting", "northing", "upward"]],
            region=region,
            spacing=spacing,
        )

        inner_bed = pygmt.surface(
            # points[["easting","northing","upward"]],
            blocked,
            spacing=spacing,
            region=region,
            registration="g",
            tension=tension,
        ).rename({"x": "easting", "y": "northing"})

        inner_bed = inner_bed.assign_attrs(tension=tension)

    # merge interpolation of inner / buffer points with outside grid
    inner_bed = inner_bed.where(
        masked_bed.isnull(),
        masked_bed,
    )

    # ensure bed doesn't cross ice base or surface
    inner_bed = ensure_no_crossing(
        inner_bed,
        icebase=icebase,
        surface=surface,
    )

    if plot is True:
        fig = maps.plot_grd(
            inner_bed,
            points=points[points.inside].rename(
                columns={"easting": "x", "northing": "y"}
            ),
        )
        fig.show()

    return inner_bed


def ensure_no_crossing(
    bed,
    icebase=None,
    surface=None,
):
    """
    make sure bed layer doesn't cross icebase or surface
    """
    # ensure bed doesn't cross ice base
    if icebase is not None:
        bed = bed.where(bed <= icebase, icebase)
    # ensure bed doesn't cross surface
    if surface is not None:
        bed = bed.where(bed <= surface, surface)

    return bed


def merge_test_train_to_outside(
    test_train_df,
    constraints,
):
    df = constraints.copy()
    outside_constraints = df[~df.inside].copy()

    fold_col_names = list(
        test_train_df.columns[test_train_df.columns.str.startswith("fold_")]
    )

    # make all outside constraints training points
    cols = {col: "train" for col in fold_col_names}
    new_cols = pd.DataFrame(cols, index=outside_constraints.index)
    outside_constraints = pd.concat([outside_constraints, new_cols], axis=1)

    # merge the outside constraints with the test/train constraints
    return pd.concat([test_train_df, outside_constraints])


def fetch_private_github_file(
    fname, username="mdtanker", fpath="RIS_grav_bath_data/main", output_dir="/data/"
):
    token = os.environ.get("GITHUB_TOKEN")
    if token is None:
        token = getpass("GITHUB_TOKEN: ")

    fpath = (
        f"https://{username}:{token}@raw.githubusercontent.com/{username}/{fpath}/"
        f"{fname}"
    )
    res = get(fpath)

    out_file = f"{output_dir}/{fname}"
    with Path.open(out_file, "wb+") as f:
        f.write(res.content)

    return Path.resolve(out_file)


def make_surface(
    spacing,
    region,
    top,
    checkerboard=True,
    amplitude=100,
    wavelength=10e3,
    plot=True,
):
    """
    Function to create either a flat or checkerboard surface.

    Parameters
    ----------
    spacing : float
        grid spacing
    region : list
        region string [e,w,n,s]
    top : float
        top for flat surface, or baselevel for checkerboard
    checkerboard : bool, optional
        choose whether to return a checkerboard or flat surface, by default True
    amplitude : int, optional
        amplitude of checkerboard, by default 100
    wavelength : float, optional
        checkerboard wavelength in same units as grid, by default 10,000
    plot : bool, optional
        plot the results, by default True

    Returns
    -------
    xarray.DataArray
        the generated surface
    """
    # create a surface with repeating checkerboard pattern
    if checkerboard is True:
        synth = vd.synthetic.CheckerBoard(
            amplitude=amplitude,
            region=region,
            w_east=wavelength,
            w_north=wavelength,
        )

        surface = synth.grid(
            spacing=spacing, data_names="upward", dims=("northing", "easting")
        ).upward

        surface += top

    # create grid of coordinates
    else:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
        )

        # create xarray dataarray from coords with a constant value as defined by 'top'
        surface = vd.make_xarray_grid(
            coords,
            np.ones_like(coords[0]) * top,
            data_names="upward",
            dims=("northing", "easting"),
        ).upward

    if plot is True:
        # plot gravity and percentage contours
        _fig, ax = plt.subplots()
        surface.plot(ax=ax, robust=True)
        ax.set_aspect("equal")

    return surface


def gravity_decay_buffer(
    buffer_perc,
    spacing=1e3,
    interest_region=(-5e3, 5e3, -10e3, 15e3),
    top=2e3,
    checkerboard=False,
    density_contrast=False,
    reference=-4e3,
    obs_height=1200,
    density=2300,
    plot=False,
    percentages=(0.99, 0.95, 0.90),
    **kwargs,
):
    """
    For a given buffer zone width (as percentage of x or y range) and domain parameters,
    calculate the max percent decay of the gravity anomaly within the region of
    interest.
    Decay is defined as the (max-min)/max.
    """
    # get x and y range of interest region
    x_diff = np.abs(interest_region[0] - interest_region[1])
    y_diff = np.abs(interest_region[2] - interest_region[3])

    # pick the bigger range
    max_diff = max(x_diff, y_diff)

    # calc buffer as percentage of width
    buffer_width = max_diff * (buffer_perc / 100)

    # round to nearest multiple of spacing
    def round_to_input(num, multiple):
        return round(num / multiple) * multiple

    # round buffer width to nearest spacing interval
    buffer_width = round_to_input(buffer_width, spacing)

    # define buffer region
    buffer_region = utils.alter_region(interest_region, zoom=-buffer_width)

    # calculate buffer width in terms of number of cells
    buffer_cells = buffer_width / spacing

    # create surface
    surface = make_surface(
        spacing=spacing,
        region=buffer_region,
        top=top,
        checkerboard=checkerboard,
        amplitude=kwargs.get("amplitude", 100),
        wavelength=kwargs.get("wavelength", 10e3),
        plot=False,
    )

    # define what the reference is
    if density_contrast is True:
        # prism reference is mean value of surface
        reference = surface.values.mean()

        # positive densities above, negative below
        dens = surface.copy()
        dens.values[:] = density
        dens = dens.where(surface >= reference, -density)
    else:
        dens = density

    # create prism layer
    flat_prisms = inv_utils.grids_to_prisms(
        surface,
        reference,
        density=dens,
    )

    # create set of observation points
    data = vd.grid_coordinates(
        interest_region,
        spacing=spacing,
        extra_coords=obs_height,
    )

    forward_df = pd.DataFrame(
        {
            "easting": data[0].ravel(),
            "northing": data[1].ravel(),
            "upward": data[2].ravel(),
        }
    )

    # calculate forward gravity of prism layer
    forward_df["forward_total"] = flat_prisms.prism_layer.gravity(
        coordinates=(
            forward_df.easting,
            forward_df.northing,
            forward_df.upward,
        ),
        field="g_z",
        progressbar=kwargs.get("progressbar", False),
    )

    grav = forward_df.set_index(["northing", "easting"]).to_xarray().forward_total

    # get max decay value inside the region of interest
    if density_contrast is True:
        # max_decay = (abs(grav.values.min())-grav.values.max())/(grav.values.max()*2)
        max_decay = 10
    else:
        max_decay = (grav.values.max() - grav.values.min()) / grav.values.max()

    # results = (
    #     f"maximum decay: {int(max_decay*100)}% \n"
    #     f"buffer: {buffer_perc}% / {buffer_width}m / {int(buffer_cells)} cells"
    # )

    if plot is True:
        # plot diagonal profiles
        if kwargs.get("plot_profile", False) is True:
            data_dict = profiles.make_data_dict(
                ["Forward gravity"],
                [grav],
                ["black"],
            )
            # profiles.plot_data(
            #     "points",
            #     start=(interest_region[0], interest_region[2]),
            #     stop=(interest_region[1], interest_region[3]),
            #     data_dict=data_dict,
            # )
            layers_dict = profiles.make_data_dict(
                ["Surface"],
                [surface],
                ["black"],
            )
            fig, _, _ = profiles.plot_profile(
                "points",
                start=(buffer_region[0], (buffer_region[3] - buffer_region[2]) / 2),
                stop=(buffer_region[1], (buffer_region[3] - buffer_region[2]) / 2),
                layers_dict=layers_dict,
                data_dict=data_dict,
                add_map=True,
                map_background=surface,
                inset=False,
                gridlines=False,
            )
            fig.show()

        # plot histogram of gravity decay values
        sns.displot(forward_df.forward_total, kde=True)

        # add lines for various decay percentiles
        col = ["r", "cyan", "k"]
        for i, p in enumerate(percentages):
            plt.axvline(
                forward_df.forward_total.max() * p,
                color=col[i],
                label=f"{p*100}%",
            )
        plt.xlabel("grav")
        plt.ylabel("count")
        plt.title("Gravity decay within region of interest")
        plt.legend()

        # plot gravity and percentage contours
        fig, ax = plt.subplots()
        grav.plot(ax=ax, robust=True)
        ax.set_aspect("equal")

        grav.plot.contour(
            levels=[
                forward_df.forward_total.max() * percentages[0],
                forward_df.forward_total.max() * percentages[1],
                forward_df.forward_total.max() * percentages[2],
            ],
            colors=["k", "cyan", "r"],
        )

        ax.set_title("Forward gravity with decay contours")

    return max_decay, buffer_width, buffer_cells, grav
