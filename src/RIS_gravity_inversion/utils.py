from __future__ import annotations

import os
from getpass import getpass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import verde as vd
from invert4geom import utils as inv_utils
from polartoolkit import profiles, utils
from requests import get


def get_buffer_points(
    buffer_width,
    grid,
    mask,
    plot=False,
):
    """
    Create buffer zone of points around ice shelf border and grid with ice shelf masked.
    """
    # get buffered mask
    mask_buffer = mask.buffer(buffer_width)

    # get only inside the mask
    inside_points = utils.mask_from_shp(
        shapefile=mask,
        xr_grid=grid,
        masked=True,
    ).rename("upward")

    # get between mask and buffer
    buffer_points = utils.mask_from_shp(
        shapefile=mask_buffer,
        xr_grid=inside_points,
        masked=True,
        invert=False,
    ).rename("upward")

    # get only outside the buffered-mask
    outside_points = utils.mask_from_shp(
        shapefile=mask_buffer,
        xr_grid=grid,
        masked=True,
    ).rename("upward")

    # create dataframes from grids
    df_outside = vd.grid_to_table(outside_points).dropna()
    df_buffer = vd.grid_to_table(buffer_points).dropna()

    # label points
    df_outside["inside"] = False
    df_outside["buffer"] = False
    df_buffer["inside"] = False
    df_buffer["buffer"] = True

    if plot:
        df_outside.plot.scatter(x="easting", y="northing", s=1, c="r", label="outside")
        df_buffer.plot.scatter(
            x="easting", y="northing", s=1, c="b", label="buffer", ax=plt.gca()
        )
        plt.legend()
        plt.show()

    return pd.concat((df_buffer, df_outside))


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

    # create topography
    if checkerboard:
        synth = vd.synthetic.CheckerBoard(
            amplitude=kwargs.get("amplitude", 100),
            region=buffer_region,
            w_east=kwargs.get("wavelength", 10e3),
            w_north=kwargs.get("wavelength", 10e3),
        )

        surface = synth.grid(
            spacing=spacing, data_names="upward", dims=("northing", "easting")
        ).upward

        surface += top

    else:
        surface = inv_utils.create_topography(
            method="flat",
            upwards=top,
            region=buffer_region,
            spacing=spacing,
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
