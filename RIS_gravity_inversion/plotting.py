import string

import cmocean  # noqa 401
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly
import plotly.express as px
import pygmt
import pylab
import pyvista as pv
import seaborn as sns
import verde as vd
import xarray as xr
from antarctic_plots import maps, profile, utils
from plotly.subplots import make_subplots

import RIS_gravity_inversion.inversion as inv
import RIS_gravity_inversion.utils as inv_utils
from RIS_gravity_inversion import optimization


def plot_inversion_ensemble_results_profile(
    prism_results,
    grav_results,
    input_forward_column,
    name_prefixes,
    start=None,
    stop=None,
    true_surface=None,
    true_regional=None,
    constraints=None,
    background=None,
    map_buffer=0.01,
    plot_regional_error=True,
    absolute_bed_error=False,
    **kwargs,
):
    # turn dataframe into datasets
    final_surfaces = []
    surface_errors = []
    for i, j in enumerate(prism_results):
        prism_ds = j.set_index(["northing", "easting"]).to_xarray()
        cols = [s for s in j.columns.to_list() if "_layer" in s]
        final = prism_ds[cols[-1]]
        final_surfaces.append(final)
        surface_errors.append(true_surface - final)

    if absolute_bed_error:
        surface_errors = [np.abs(x) for x in surface_errors]

    if plot_regional_error:
        regionals = []
        final_forwards = []
        final_residuals = []
        regional_errors = []
        for i, j in enumerate(grav_results):
            grav_ds = j.set_index(["northing", "easting"]).to_xarray()
            cols = [s for s in j.columns.to_list() if "_forward_grav" in s]
            final_forward = grav_ds[cols[-1]]
            final_forwards.append(final_forward + grav_ds.reg)
            final_residuals.append([final_forward + grav_ds.reg - grav_ds.Gobs_shift])
            regionals.append(grav_ds.reg)
            regional_errors.append(true_regional - grav_ds.reg)

    if start is None:
        reg = vd.get_region((grav_results.easting, grav_results.northing))
        start = [reg[0], reg[2]]
    if stop is None:
        reg = vd.get_region((grav_results.easting, grav_results.northing))
        stop = [reg[1], reg[3]]

    if background is None:
        if true_surface is not None:
            background = true_surface
        else:
            background = final_surfaces[0]

    # extract layers for profile plot
    topo_names = [f"{x} bed error" for x in name_prefixes]
    topo_grids = surface_errors

    if len(topo_grids) <= 4:
        colors = ["black", "blue", "purple", "orange"]
    else:
        colors = [utils.random_color() for x in topo_grids]

    if kwargs.get("data_color", None) is not None:
        data_colors = kwargs.get("data_color")
    else:
        data_colors = colors

    if kwargs.get("layers_color", None) is not None:
        layers_colors = kwargs.get("layers_color")
    else:
        layers_colors = colors

    layers_dict = profile.make_data_dict(
        names=topo_names,
        grids=topo_grids,
        colors=layers_colors,
    )

    if plot_regional_error:
        data_names = [f"{x} regional error" for x in name_prefixes]
        data_grids = regional_errors
        data_dict = profile.make_data_dict(
            names=data_names,
            grids=data_grids,
            colors=data_colors,
        )
    else:
        data_dict = None

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    fig = profile.plot_profile(
        "points",
        fig_height=8,
        start=start,
        stop=stop,
        num=1000,
        fill_layers=False,
        data_height=4,
        # subplot_orientation='horizontal',
        subplot_orientation="vertical",
        add_map=True,
        map_background=background,
        map_cmap="batlowW",
        inset=False,
        gridlines=False,
        coast=False,
        map_buffer=map_buffer,
        data_dict=data_dict,
        layers_dict=layers_dict,
        share_yaxis=True,
        # layer_pen=[3] + [2.6-(.4*x) for x in range(len(layers_dict.items())-1)],
        # data_pen=[3] + [2.6-(.4*x) for x in range(len(data_dict.items())-1)],
        frame=True,
        map_points=constraints,
        map_points_style="x.3c",
        map_points_pen="1p,black",
        map_line_pen="2p,black",
        **kwargs,
    )

    return fig


def plot_regional_gridding_ensemble_profile(
    grav_anomalies,
    name_prefixes,
    start=None,
    stop=None,
    polyline=None,
    true_regional=None,
    constraints=None,
    background=None,
    map_buffer=0.01,
    **kwargs,
):
    # turn dataframe into datasets
    regionals = []
    residuals = []
    regional_errors = []
    for i, j in enumerate(grav_anomalies):
        grav_ds = j.set_index(["northing", "easting"]).to_xarray()
        residuals.append(grav_ds.res)
        regionals.append(grav_ds.reg)
        if true_regional is not None:
            regional_errors.append(true_regional - grav_ds.reg)

    if start is None:
        reg = vd.get_region((grav_anomalies[0].easting, grav_anomalies[0].northing))
        start = [reg[0], reg[2]]
    if stop is None:
        reg = vd.get_region((grav_anomalies[0].easting, grav_anomalies[0].northing))
        stop = [reg[1], reg[3]]

    if background is None:
        if true_regional is not None:
            background = true_regional
        else:
            background = grav_ds.reg

    # extract layers for profile plot
    topo_names = [f"{x}, regional" for x in name_prefixes]
    topo_grids = regionals

    if len(topo_grids) <= 4:
        colors = ["black", "blue", "purple", "orange"]
    else:
        colors = [utils.random_color() for x in topo_grids]

    if kwargs.get("data_color", None) is not None:
        data_colors = kwargs.get("data_color")
    else:
        data_colors = colors

    if kwargs.get("layers_color", None) is not None:
        layers_colors = kwargs.get("layers_color")
    else:
        layers_colors = colors

    layers_dict = profile.make_data_dict(
        names=topo_names, grids=topo_grids, colors=layers_colors
    )

    data_names = [f"{x}, residual" for x in name_prefixes]
    data_grids = residuals

    data_dict = profile.make_data_dict(
        names=data_names, grids=data_grids, colors=data_colors
    )

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    if polyline is None:
        method = "points"
    else:
        method = "polyline"

    fig = profile.plot_profile(
        method=method,
        start=start,
        stop=stop,
        polyline=polyline,
        # num=1000,
        fig_height=8,
        fill_layers=False,
        data_height=4,
        # subplot_orientation='horizontal',
        subplot_orientation="vertical",
        add_map=True,
        map_background=background,
        map_cmap="viridis",
        inset=False,
        gridlines=False,
        coast=False,
        map_buffer=map_buffer,
        data_dict=data_dict,
        layers_dict=layers_dict,
        share_yaxis=True,
        # layer_pen=[3] + [2.6-(.4*x) for x in range(len(layers_dict.items())-1)],
        # data_pen=[3] + [2.6-(.4*x) for x in range(len(data_dict.items())-1)],
        frame=True,
        map_points=constraints,
        map_points_style="x.3c",
        map_points_pen="1p,black",
        map_line_pen="2p,black",
        **kwargs,
    )

    return fig


def plot_inversion_results_profile(
    prism_results,
    grav_results,
    input_forward_column,
    observed_grav_column="Gobs_shift",
    start=None,
    stop=None,
    true_surface=None,
    constraints=None,
    background=None,
    map_buffer=0.01,
    plot_bed_misfit=False,
    layers_frame=None,
    data_frame=None,
    **kwargs,
):
    # turn dataframe into datasets
    prism_ds = prism_results.set_index(["northing", "easting"]).to_xarray()
    grav_ds = grav_results.set_index(["northing", "easting"]).to_xarray()

    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]
    initial_surface = prism_ds.surface
    final_surface = prism_ds[cols[-1]]

    cols = [s for s in grav_results.columns.to_list() if "_final_misfit" in s]
    # final_residual = grav_ds[cols[-1]]

    cols = [s for s in grav_results.columns.to_list() if "_initial_misfit" in s]
    initial_residual = grav_ds[cols[0]]

    cols = [s for s in grav_results.columns.to_list() if "_forward_grav" in s]
    final_forward = grav_ds[cols[-1]]

    if start is None:
        reg = vd.get_region((grav_results.easting, grav_results.northing))
        start = [reg[0], reg[2]]
    if stop is None:
        reg = vd.get_region((grav_results.easting, grav_results.northing))
        stop = [reg[1], reg[3]]

    if background is None:
        background = final_surface

    # extract layers for profile plot
    topo_names = [
        "Starting bed",
        "Inverted bed",
    ]
    topo_grids = [
        initial_surface,
        final_surface,
    ]
    layers_dict = profile.make_data_dict(
        names=topo_names,
        grids=topo_grids,
        colors=["black", "blue"],
    )

    data_names = [
        "Initial residual gravity",
        # "Final residual",
        # "Initial forward",
        "Final forward gravity",
        "Observed gravity",
        #         "Initial misfit",
        #         "Final misfit",
        #         "Final forward",
    ]

    data_grids = [
        initial_residual,
        # final_residual,
        # grav_ds[input_forward_column],
        final_forward + grav_ds.reg,
        grav_ds[observed_grav_column],
        #         grav_ds.misfit,
        #         final_residual+grav_ds.reg,
        #
        #         final_forward,
    ]

    data_dict = profile.make_data_dict(
        names=data_names,
        grids=data_grids,
        colors=["black", "blue", "red", "purple", "gray", "brown"],
    )

    if true_surface is not None:
        layers_dict["True bed"] = dict(name="True bed", grid=true_surface, color="red")
        #     old_dict = layers_dict.copy()
        #     layers_dict = {
        #         "True bed": dict(
        #             name="True bed", grid=true_surface, color="red")
        #         }
        #     layers_dict.update(old_dict)

        if plot_bed_misfit is True:
            data_dict["Bed misfit"] = dict(
                name="Bed misfit", grid=(true_surface - final_surface), color="purple"
            )

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    if layers_frame is None:
        layers_frame = ["nesW", "xafg", "ya+lElevation (m)"]
    if data_frame is None:
        data_frame = ["nesW", "xafg", "ya+lGravity (mGal)"]

    fig = profile.plot_profile(
        "points",
        fig_height=6.5,
        start=start,
        stop=stop,
        num=1000,
        fill_layers=False,
        data_height=3,
        add_map=True,
        map_background=background,
        map_cmap="viridis",
        inset=False,
        gridlines=False,
        coast=False,
        map_buffer=map_buffer,
        data_dict=data_dict,
        layers_dict=layers_dict,
        share_yaxis=True,
        data_pen=[1, 2, 1],
        layer_pen=[1, 2, 1],
        frame=True,
        map_points=constraints,
        map_points_style="x.3c",
        map_points_pen="1p,black",
        map_line_pen="2p,black",
        layers_frame=layers_frame,
        data_frame=data_frame,
        **kwargs,
    )

    return fig


def plot_noise_spacing_ensemble(
    df,
    background="score",
    background_title=None,
    points_color=None,
    points_size=None,
    points_scaling=1,
    points_color_log=False,
    background_color_log=False,
    background_robust=False,
    points_robust=False,
    plot_contours=None,
):
    fig, ax = plt.subplots(figsize=(9, 6))
    df = df.copy()

    df["spacing"] = (df.spacing / 1e3).astype(int)

    noise_levels = df.noise.unique()
    spacings = df.spacing.unique()

    noise_step = noise_levels[1] - noise_levels[0]
    spacing_step = spacings[1] - spacings[0]

    extent = [
        df.noise.min() - noise_step / 2,
        df.noise.max() + noise_step / 2,
        df.spacing.min() - spacing_step / 2,
        df.spacing.max() + spacing_step / 2,
    ]

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    ax.grid(which="major", visible=False)

    if (points_color is not None) or (points_size is not None):
        if points_color is None:
            points_color = "b"
        if points_size is None:
            points_size = 50

        lims = utils.get_min_max(points_color, robust=points_robust)

        if points_color_log is True:
            norm = mpl.colors.LogNorm(
                vmin=lims[0],
                vmax=lims[1],
            )
            vmin = None
            vmax = None
        else:
            norm = None
            vmin = lims[0]
            vmax = lims[1]
        points = ax.scatter(
            df.noise,
            df.spacing,
            s=points_size * points_scaling,
            c=points_color,
            cmap="cmo.deep",
            zorder=10,
            edgecolors="black",
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        if isinstance(points_size, pd.Series):
            kw = dict(prop="sizes", num=3, func=lambda s: s / points_scaling)
            ax.legend(
                *points.legend_elements(**kw),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                title=points_size.name,
            )

        cbar2 = fig.colorbar(points)
        try:
            cbar2.set_label(points_color.name)
        except AttributeError:
            pass

    if background_color_log is True:
        norm = mpl.colors.LogNorm()
    else:
        norm = None

    lims = utils.get_min_max(df[background], robust=background_robust)
    extent = [plt.xlim()[0], plt.xlim()[1], plt.ylim()[1], plt.ylim()[0]]

    plot_background = ax.imshow(
        df.pivot("spacing", "noise", background),
        # alpha=.8,
        extent=extent,
        aspect="auto",
        cmap=pylab.cm.get_cmap("cmo.matter", 10),
        vmin=lims[0],
        vmax=lims[1],
        norm=norm,
        origin="upper",
    )

    cbar = fig.colorbar(plot_background)
    cbar.set_label(background_title)

    ax.set_xlabel("Gravity noise level (%)")
    ax.set_ylabel("Survey line spcaing (km)")

    import scipy.ndimage

    if plot_contours is not None:
        data = df.pivot("spacing", "noise", background)
        data = scipy.ndimage.zoom(data, 5)
        contour = ax.contour(
            data,
            colors="black",
            levels=plot_contours,
            corner_mask=False,
            extent=extent,
            origin="upper",
        )
        cbar.add_lines(contour)

    return fig


def plot_noise_cellsize_ensemble(
    df,
    background="score",
    background_title=None,
    points_color=None,
    points_size=None,
    points_scaling=1,
    points_color_log=False,
    background_color_log=False,
    background_robust=False,
    points_robust=False,
    plot_contours=None,
):
    fig, ax = plt.subplots(figsize=(9, 6))

    noise_levels = df.noise.unique()
    cell_sizes = df.cell_size.unique()

    noise_step = noise_levels[1] - noise_levels[0]
    cell_size_step = cell_sizes[1] - cell_sizes[0]

    extent = [
        df.noise.min() - noise_step / 2,
        df.noise.max() + noise_step / 2,
        df.cell_size.min() - cell_size_step / 2,
        df.cell_size.max() + cell_size_step / 2,
    ]

    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    # ax.set_xticks(np.arange(extent[0], extent[1], noise_step)+noise_step/2)
    # ax.set_yticks(np.arange(extent[2], extent[3], cell_size_step)+cell_size_step/2)
    # ax.set_xticks(np.arange(extent[0], extent[1], noise_step/2), minor=True)
    # ax.set_yticks(np.arange(extent[2], extent[3], cell_size_step/2), minor=True)
    # ax.grid(which='minor', color='white')
    ax.grid(which="major", visible=False)

    #     grid_cell_width = \
    #         (ax.transData.transform((0, cell_sizes[1])) - \
    #         ax.transData.transform((0, cell_sizes[0])))[1]
    #     grid_cell_width = (grid_cell_width/2)**2
    # def normalize(data, low, high):
    #     min_val = data.values.min()
    #     max_val = data.values.max()
    #     x = (high - low) * (((data - min_val) / (max_val - min_val)).clip(0, 1)) + low
    #     return x
    #     size = normalize(
    #         df.outer_bounds,
    #         low=((grid_cell_width)*0.0001),
    #         high=(grid_cell_width*.6))

    if (points_color is not None) or (points_size is not None):
        if points_color is None:
            points_color = "b"
        if points_size is None:
            points_size = 50

        lims = utils.get_min_max(points_color, robust=points_robust)

        if points_color_log is True:
            norm = mpl.colors.LogNorm(
                vmin=lims[0],
                vmax=lims[1],
            )
            vmin = None
            vmax = None
        else:
            norm = None
            vmin = lims[0]
            vmax = lims[1]
        points = ax.scatter(
            df.noise,
            df.cell_size,
            s=points_size * points_scaling,
            c=points_color,
            cmap="cmo.deep",
            zorder=10,
            edgecolors="black",
            norm=norm,
            vmin=vmin,
            vmax=vmax,
        )
        if isinstance(points_size, pd.Series):
            kw = dict(prop="sizes", num=3, func=lambda s: s / points_scaling)
            ax.legend(
                *points.legend_elements(**kw),
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                title=points_size.name,
            )

        cbar2 = fig.colorbar(points)
        try:
            cbar2.set_label(points_color.name)
        except AttributeError:
            pass

    if background_color_log is True:
        norm = mpl.colors.LogNorm()
    else:
        norm = None

    lims = utils.get_min_max(df[background], robust=background_robust)
    extent = [plt.xlim()[0], plt.xlim()[1], plt.ylim()[1], plt.ylim()[0]]

    plot_background = ax.imshow(
        df.pivot("cell_size", "noise", background),
        # alpha=.8,
        extent=extent,
        aspect="auto",
        cmap=pylab.cm.get_cmap("cmo.matter", 10),
        vmin=lims[0],
        vmax=lims[1],
        norm=norm,
        origin="upper",
    )

    cbar = fig.colorbar(plot_background)
    cbar.set_label(background_title)

    ax.set_xlabel("Gravity noise level (%)")
    ax.set_ylabel("Gravity cell size (m) ")

    import scipy.ndimage

    if plot_contours is not None:
        data = df.pivot("cell_size", "noise", background)
        data = scipy.ndimage.zoom(data, 10)
        contour = ax.contour(
            data,
            colors="black",
            levels=plot_contours,
            corner_mask=False,
            extent=extent,
            origin="upper",
        )
        cbar.add_lines(contour)

    # ax.set_title()
    # plt.show()

    return fig


def plot_convergence(results, iter_times=None, logy=False):
    sns.set_theme()

    # get misfit data at end of each iteration
    cols = [s for s in results.columns.to_list() if "_final_misfit" in s]
    iters = len(cols)
    # final_misfits = [np.sqrt(utils.RMSE(results[i])) for i in cols]
    final_misfits = [utils.RMSE(results[i]) for i in cols]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title("Inversion convergence")
    ax1.plot(range(iters), final_misfits, "b-")
    ax1.set_xlabel("Iteration")
    if logy:
        ax1.set_yscale("log")
    ax1.set_ylabel("RMS (mGal)", color="b")
    ax1.tick_params(axis="y", colors="b", which="both")

    if iter_times is not None:
        ax2 = ax1.twinx()
        ax2.plot(range(iters), np.cumsum(iter_times), "g-")
        ax2.set_ylabel("Cumulative time (s)", color="g")
        ax2.tick_params(axis="y", colors="g")
        ax2.grid(False)
    plt.tight_layout()


def plot_ensemble_as_lines(
    results,
    x_col,
    groupby_col,
    x_label=None,
    cbar_label=None,
    logy=False,
    logx=False,
    trend_line=False,
):
    sns.set_theme()

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    # plt.title('')

    grouped = results.groupby(groupby_col)

    norm = plt.Normalize(
        vmin=results[groupby_col].values.min(), vmax=results[groupby_col].values.max()
    )

    for name, group in grouped:
        ax1.plot(
            group[x_col],
            group.RMSE,
            ".-",
            markersize=7,
            color=plt.cm.viridis(norm(name)),
        )
    if trend_line:
        z = np.polyfit(results[x_col], results.RMSE, 1)
        y_hat = np.poly1d(z)(results[x_col])

        ax1.plot(results[x_col], y_hat, "r", lw=1)
        text = f"$slope={z[0]:.3g}$"
        plt.gca().text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
        )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    cbar = plt.colorbar(sm)
    cbar.set_label(cbar_label)

    ax1.set_xlabel(
        x_label,
    )

    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    ax1.set_ylabel("Bed RMSE (m)")

    # plt.legend(loc='best')
    plt.tight_layout()


def plot_constraint_spacing_ensemble(
    results,
    plot_constraints=False,
    plot_starting_error=False,
    highlight_points=None,
    subplot_label=None,
    horizontal_line=None,
    logy=False,
    logx=False,
):
    sns.set_theme()

    df = results.copy()

    df.sort_values("spacing", inplace=True)

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title("Effects of constraint spacing")
    if subplot_label is not None:
        plt.title(subplot_label, loc="left", fontsize=18, weight="bold")

    if horizontal_line is not None:
        plt.axhline(y=horizontal_line, linewidth=2, color="gray", linestyle="dashed")

    ax1.plot(df.spacing / 1e3, df.RMSE, "bd-", markersize=7, label="inverted")
    ax1.set_xlabel(
        "Constraint spacing (km)",
        # color="b",
    )

    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    ax1.set_ylabel("Bed RMSE (m)")
    # ax1.tick_params(axis="x", colors='b', which="both")
    ax1.set_zorder(2)

    if plot_starting_error:
        ax1.plot(
            df.spacing / 1e3,
            df.starting_errors,
            "g.-",
            markersize=10,
            label="starting",
            zorder=1,
        )

    if highlight_points is not None:
        for i, ind in enumerate(highlight_points):
            plt.plot(
                df.spacing.loc[ind] / 1e3,
                df.RMSE.loc[ind],
                "s",
                markersize=12,
                color="b",
                zorder=3,
            )
            plt.annotate(
                string.ascii_lowercase[i + 1],
                (df.spacing.loc[ind] / 1e3, df.RMSE.loc[ind]),
                fontsize=15,
                color="white",
                ha="center",
                va="center",
                zorder=4,
            )

    if plot_constraints:
        ax2 = ax1.twiny()
        ax2.plot(df["Constraints number"], df.RMSE, "g.-", markersize=10)
        ax2.set_xlabel("Number of constraints", color="g")
        ax2.tick_params(axis="x", which="both", colors="g")
        ax2.set_xscale("log")
        ax2.invert_xaxis()
        ax2.grid(False)
        if highlight_points is not None:
            for i in highlight_points:
                ax2.plot(
                    df["Constraints number"].loc[i],
                    df.RMSE.loc[i],
                    "+",
                    markersize=8,
                    color=sns.color_palette()[i],
                    label=f"constraints number, {i}",
                )

    plt.legend(loc="best")
    plt.tight_layout()


def plot_parameter_cv(
    scores, dampings, logx=False, logy=False, param_name="Hyperparameter"
):
    df = pd.DataFrame({"scores": scores, "dampings": dampings})
    df.sort_values(by="dampings", inplace=True)

    best = df.scores.argmin()

    sns.set_theme()

    plt.figure(figsize=(5, 3.5))
    plt.title(f"{param_name} Cross-validation")
    plt.plot(df.dampings, df.scores, marker="o")
    plt.plot(
        df.dampings.iloc[best],
        df.scores.iloc[best],
        "s",
        markersize=10,
        color=sns.color_palette()[3],
        label="Minimum",
    )
    plt.legend(loc="best")
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")
    plt.xlabel(f"{param_name} value")
    plt.ylabel("Root Mean Square Error")

    # plt.ticklabel_format(style='plain')
    plt.tight_layout()


def plot_data_and_layers(
    topo_grids,
    topo_titles,
    grav_grid,
    inversion_region,
    buffer_region=None,
    points=None,
    points_style="c0.05c",
    robust: bool = True,
):
    if buffer_region is None:
        buffer_region = inversion_region

    fig = maps.plot_grd(
        grid=grav_grid,
        fig_height=8,
        cmap="vik",
        region=buffer_region,
        title="Observed gravity",
        cbar_unit="mGal",
        show_region=inversion_region,
        hist=True,
        cbar_yoffset=3,
        robust=robust,
    )
    fig.text(
        position="TL",
        justify="BL",
        text="a)",
        font="18p,Helvetica,black",
        offset="j0/.3",
        no_clip=True,
    )
    for i, j in enumerate(topo_grids):
        fig = maps.plot_grd(
            grid=j,
            fig_height=8,
            cmap="rain",
            reverse_cpt=True,
            region=buffer_region,
            grd2cpt=True,
            title=topo_titles[i],
            cbar_label="elevation",
            cbar_unit="m",
            show_region=inversion_region,
            hist=True,
            cbar_yoffset=3,
            fig=fig,
            origin_shift="xshift",
            points=points,
            points_style=points_style,
            robust=robust,
        )
        fig.text(
            position="TL",
            justify="BL",
            text=f"{string.ascii_lowercase[i+1]})",
            font="18p,Helvetica,black",
            offset="j0/.3",
            no_clip=True,
        )
    return fig


def plotly_subplots(grids, titles):
    figures = []
    for i in grids:
        img = px.imshow(i)
        figures.append(img)

    fig = make_subplots(rows=1, cols=len(figures))

    for i, figure in enumerate(figures):
        for trace in range(len(figure["data"])):
            # add images
            fig.append_trace(
                figure["data"][trace],
                row=1,
                col=i + 1,
            )
            # add titles
            if titles is not None:
                fig.add_annotation(
                    xref="x domain",
                    yref="y domain",
                    x=0.5,
                    y=1.2,
                    showarrow=False,
                    text=f"<b>{titles[i]}</b>",
                    row=1,
                    col=i,
                )

    fig.update_xaxes(
        scaleanchor="y",
        scaleratio=1,
        constrain="domain",
    )

    return fig


def add_light(plotter, prisms):
    # Add a ceiling light
    west, east, south, north = vd.get_region((prisms.easting, prisms.northing))
    easting_center, northing_center = (east + west) / 2, (north + south) / 2
    light = pv.Light(
        position=(easting_center, northing_center, 100e3),
        focal_point=(easting_center, northing_center, 0),
        intensity=1,  # 0 to 1
        light_type="scene light",  # the light doesn't move with the camera
        positional=False,  # the light comes from infinity
        shadow_attenuation=0,  # 0 to 1,
    )
    plotter.add_light(light)


def show_prism_layers(
    prisms: list,
    cmap: str = "viridis",
    color_by: str = "density",
    region: list = None,
    clip_box: bool = True,
    **kwargs,
):
    """
    show prism layers using PyVista

    Parameters
    ----------
    prisms : list
        _description_
    cmap : str, optional
        matplotlib colorscale to use, by default "viridis"
    color_by : str, optional
        either use a variable of the prism_layer dataset, typically 'density' or
        'thickness', or choose 'constant' to have each layer colored by a unique color
        use kwarg `colors` to alter these colors, by default is "density"
    region : list, optional
        region to clip the prisms to, by default is full extent
    clip_box : bool, optional
         choose to clip out cube of 3D plot, by default is True
    """

    # pv.themes.DefaultTheme.server_proxy_enabled()
    # pv.themes.DefaultTheme.server_proxy_prefix('/proxy/')

    # Plot with pyvista
    plotter = pv.Plotter(
        lighting="three_lights",
        # window_size=(1000, 1000),
        notebook=True,
    )

    opacity = kwargs.get("opacity", None)

    for i, j in enumerate(prisms):
        # if region is given, clip model
        if region is not None:
            j = j.sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )

        # turn prisms into pyvist object
        pv_grid = j.prism_layer.to_pyvista()

        # clip corner out of model to help vizualize
        if clip_box is True:
            # extract region from first prism layer
            reg = vd.get_region((j.easting.values, j.northing.values))
            # box_buffer used make box slightly bigger
            box_buffer = kwargs.get("box_buffer", 5e3)
            # set 6 edges of cube to clip out
            bounds = [
                reg[0] - box_buffer,
                reg[0] + box_buffer + ((reg[1] - reg[0]) / 2),
                reg[2] - box_buffer,
                reg[2] + box_buffer + ((reg[3] - reg[2]) / 2),
                np.nanmin(j.bottom),
                np.nanmax(j.top),
            ]
            pv_grid = pv_grid.clip_box(
                bounds,
                invert=True,
            )

        if opacity is not None:
            trans = opacity[i]
        else:
            trans = None

        if color_by == "constant":
            colors = kwargs.get(
                "colors", ["lavender", "aqua", "goldenrod", "saddlebrown", "black"]
            )
            plotter.add_mesh(
                pv_grid,
                color=colors[i],
                smooth_shading=kwargs.get("smooth_shading", False),
                style=kwargs.get("style", "surface"),
                show_edges=kwargs.get("show_edges", False),
                opacity=trans,
            )
        else:
            plotter.add_mesh(
                pv_grid,
                scalars=color_by,
                cmap=cmap,
                flip_scalars=kwargs.get("flip_scalars", False),
                smooth_shading=kwargs.get("smooth_shading", False),
                style=kwargs.get("style", "surface"),
                show_edges=kwargs.get("show_edges", False),
                log_scale=kwargs.get("log_scale", True),
                opacity=trans,
            )
        plotter.set_scale(
            zscale=kwargs.get("zscale", 75)
        )  # exaggerate the vertical coordinate
        plotter.camera_position = kwargs.get("camera_position", "xz")
        plotter.camera.elevation = kwargs.get("elevation", 20)
        plotter.camera.azimuth = kwargs.get("azimuth", -25)
        plotter.camera.zoom(kwargs.get("zoom", 1.2))

    # Add a ceiling light
    add_light(plotter, prisms[i])

    plotter.show_axes()

    plotter.show(jupyter_backend=kwargs.get("backend", "client"))


def plot_prism_layers(
    layers: dict,
    region: list = None,
    cmap: str = "viridis",
    plot_type: str = "2D",
    layers_for_3d: list = None,
    **kwargs,
):
    """
    Plot prism layers resulting from `inversion.grids_to_prism_layers()`. Choose between
    2D, which plots grids of prism thickness, or 3D, which plots the entire prism model
    in 3D using PyVista.

    Parameters
    ----------
    layers : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
            'len': int; length of df
            'prisms': xarray.Dataset; harmonica.prism_layer
    region : list, optional
        GMT-format region to plot for both 2d and 3d plots, by default is buffer region
    cmap : str, optional
        matplotlib colormap to use in plotting
    plot_type : str, optional
        choose between plotting prism layers in '2D' or '3D', by default '2D'
    layers_for_3d: list
        list of layers to include in 3D plot, by default is all layers.
    """

    if plot_type == "3D":
        if layers_for_3d is not None:
            layers_to_plot = dict((k, layers[k]) for k in layers_for_3d)
        else:
            layers_to_plot = layers

        # get list of prism layers
        prism_list = [prism.get("prisms") for prism in layers_to_plot.values()]

        # removed duplicate kwargs before passing to show_prism_layers()
        subset_kwargs = {
            kw: kwargs[kw]
            for kw in kwargs
            if kw
            not in [
                "cmap",
                "color_by",
                "region",
                "clip_box",
            ]
        }

        # plot prisms layers in 3D with pyvista
        show_prism_layers(
            prism_list,
            cmap=kwargs.get("cmap", None),
            color_by=kwargs.get("color_by", "constant"),
            region=region,
            clip_box=kwargs.get("clip_box", True),
            **subset_kwargs,
        )

    elif plot_type == "2D":
        sub_width = 5
        # nrows, ncols = utils.square_subplots(len(layers))
        nrows, ncols = 1, len(layers)

        # setup subplot figure
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width * ncols, sub_width * nrows),
        )

        for i, (k, v) in enumerate(layers.items()):
            # get thickness grid from prisms
            thick = v["prisms"].thickness

            # clip grids to region if specified
            if region is not None:
                thick = thick.sel(
                    easting=slice(region[0], region[1]),
                    northing=slice(region[2], region[3]),
                )
            if ncols < 2:
                thick.plot(
                    ax=ax,
                    robust=True,
                    cmap=cmap,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "anchor": (1, 1),
                        "fraction": 0.05,
                        "pad": 0.04,
                    },
                )
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_aspect("equal")

            else:
                thick.plot(
                    ax=ax[i],
                    robust=True,
                    cmap=cmap,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "anchor": (1, 1),
                        "fraction": 0.05,
                        "pad": 0.04,
                    },
                )
                ax[i].set_title(f"{k} prism thickness")

        if ncols > 1:
            for a in ax:
                a.set_xticklabels([])
                a.set_yticklabels([])
                a.set_xlabel("")
                a.set_ylabel("")
                a.set_aspect("equal")

        fig.tight_layout()


def forward_grav_plotting(
    df_forward: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
    plot_dists: bool = False,
    plot_power_spectrums: bool = False,
    exclude_layers: list = None,
    registration="g",
    block_reduction="pygmt",
    inversion_region=None,
):
    """
    Plot results from forward gravity calculations of prism layers.

    Parameters
    ----------
    df_forward : pd.DataFrame
        Dataframe output from func: inversion.forward_gravity_layers()
    region : list, optional
        GMT format region to use for plots, by default is extent of gravity data
    grav_spacing : float, optional
        spacing of the gravity data to create plots, by default None
    plot_dists : bool, optional
        Choose whether to plot the resulting distributions, by default False
    plot_power_spectrums : bool, optional
        Choose to plot radially average power spectrum of layers, by default False
    exclude_layers : list, optional
        list of layers to exclude from plots, by default None

    """

    # if region not supplied, extract from dataframe
    if region is None:
        region = vd.get_region((df_forward.easting, df_forward.northing))

    # if gravity spacing not supplied, extract from dataframe
    if grav_spacing is None:
        grid = df_forward.set_index(["northing", "easting"]).to_xarray().Gobs
        grav_spacing = float(utils.get_grid_info(grid)[0])

    # drop columns previously used in Bouguer correction
    if exclude_layers is not None:
        cols2drop = ~df_forward.columns.str.contains("|".join(exclude_layers))
        df = df_forward[df_forward.columns[cols2drop]]
    else:
        df = df_forward.copy()

    # get list of columns to grid
    cols_to_grid = [x for x in df.columns.to_list() if "forward" in x]

    sub_width = 5

    # nrows, ncols = utils.square_subplots(len(cols_to_grid))
    nrows, ncols = 1, len(cols_to_grid)

    # setup subplot figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width * ncols, sub_width * nrows),
    )

    # empty list to append grids to
    forward_grids = []

    for i, (col, ax) in enumerate(zip(cols_to_grid, axs.T.ravel())):
        # grid = pygmt.xyz2grd(
        #     data=df[["easting", "northing", col]],
        #     region=region,
        #     spacing=grav_spacing,
        #     registration=registration,
        #     verbose="q",
        # )
        grid = pygmt.surface(
            data=df[["easting", "northing", col]],
            region=region,
            spacing=grav_spacing,
            T=0.25,
            M="0c",
            registration=registration,
        )

        forward_grids.append(grid)

        # plot each grid
        grid.plot(
            ax=ax,
            x="easting",
            y="northing",
            robust=True,
            cmap="RdBu_r",
            cbar_kwargs={
                "orientation": "horizontal",
                "anchor": (1, 1),
                "fraction": 0.05,
                "pad": 0.04,
            },
        )

        if inversion_region is not None:
            ax.add_patch(
                mpl.patches.Rectangle(
                    xy=(inversion_region[0], inversion_region[2]),
                    width=(inversion_region[1] - inversion_region[0]),
                    height=(inversion_region[3] - inversion_region[2]),
                    linewidth=1,
                    fill=False,
                )
            )

        # set column names as titles
        ax.set_title(col)

        # set axes labels and make proportional
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    fig.tight_layout()

    # add grids and names to dictionary
    grid_dict = dict(zip(cols_to_grid, forward_grids))

    if plot_dists is True:
        # get columns to include in plots
        dists = df.drop(["easting", "northing", "upward", "Gobs"], axis=1)
        # reorganize
        data = dists.melt(var_name="layer")
        # layout the subplot grid
        g = sns.FacetGrid(
            data, col="layer", sharex=False, sharey=False
        )  # , col_wrap=5)
        # add histogram for each column
        # g.map(sns.histplot, 'value', binwidth=1, kde=True,)
        # add kernel density estimate plot for each column
        g.map(sns.kdeplot, "value", fill=True, bw_adjust=0.4)
        # groupby to get mean and median
        data_g = data.groupby("layer").agg(["mean", "median"])
        # extract and flatten the axes from the figure
        axes = g.axes.flatten()

        # iterate through each axes
        for ax in axes:
            # extract the species name
            grav_type = ax.get_title().split(" = ")[1]
            # select the data for the species
            d = data_g.loc[grav_type, :]
            # plot the lines
            ax.axvline(x=d.value["mean"], c="k", ls="-", lw=1.5, label="mean")
            ax.axvline(x=d.value["median"], c="grey", ls="--", lw=1.5, label="median")
            ax.legend(fontsize=8)
        # add title
        g.fig.suptitle(
            "Forward gravity histograms", y=1, va="bottom", size=16, fontweight="bold"
        )

        plt.figure()
        sns.histplot(
            data=dists,
            palette="viridis",
            kde=True,
            # stat='count',
            multiple="stack",
            element="step",
        )
        plt.title("Stacked histograms of forward gravity", size=16, fontweight="bold")

    if plot_power_spectrums is True:
        # get columns to include in plots
        power = df.drop(["easting", "northing", "upward", "Gobs"], axis=1)
        # plot radially average power spectrum for each layer
        utils.raps(
            df,
            list(power.columns),
            region=region,
            spacing=grav_spacing,
        )

    return grid_dict


def corrections_plotting(
    df: pd.DataFrame,
    cols_to_grid: list,
    titles: list = None,
    robust: bool = True,
    inversion_region: list = None,
    buffer_region: list = None,
    points: pd.DataFrame = None,
    points_style: str = "c0.05c",
    **kwargs,
):
    """
    Plot results from partial bouguer corrections.
    """

    ds = df.set_index(["northing", "easting"]).to_xarray()

    grids = []
    for i in cols_to_grid:
        grids.append(ds[i])

    if titles is None:
        titles = cols_to_grid

    for i, g in enumerate(grids):
        if i == 0:
            fig = None
            origin_shift = "initialize"
        else:
            origin_shift = "xshift"

        fig = maps.plot_grd(
            grid=g,
            fig_height=9,
            cmap=f"vik+h{np.nanmedian(g)}",
            robust=robust,
            region=buffer_region,
            title=titles[i],
            cbar_unit="mGal",
            show_region=inversion_region,
            hist=True,
            cbar_yoffset=3,
            # grd2cpt=True,
            title_font="16p,Helvetica-bold,black",
            fig=fig,
            origin_shift=origin_shift,
            points=points,
            points_style=points_style,
            shp_mask=kwargs.get("shp_mask", None),
        )
        fig.text(
            position="TL",
            justify="BL",
            text=f"{string.ascii_lowercase[i]})",
            font="18p,Helvetica,black",
            offset="j0/.3",
            no_clip=True,
        )

    return fig


def misfit_plotting(
    df: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
    registration="g",
    constraints: pd.DataFrame = None,
    robust: bool = True,
    points_style="x.2c",
    points_pen="1p",
    **kwargs,
):
    """
    Plot results from anomaly calculations.
    If either region or grav_spacing are provided, the full regions will be gridded.
    If neither are provided, the data is assumed to be regularly spaced and will be
    plotted as is.
    Parameters
    ----------
    df : pd.DataFrame
        Dataframe output from func: inversion.misfit()
    region : list, optional
        GMT format region to use for the plots, by default is extent of gravity data
    grav_spacing : float, optional
        spacing of the gravity data to create plots, by default None
    """

    input_forward_column = kwargs.get("input_forward_column", "forward")
    input_grav_column = kwargs.get("input_grav_column", "grav")

    if (region is None) & (grav_spacing is None):
        ds = df.set_index(["northing", "easting"]).to_xarray()
        grav = ds[input_grav_column]
        forward = ds[input_forward_column]

    else:
        # if inversion region not supplied, extract from dataframe
        if region is None:
            region = vd.get_region((df.easting, df.northing))
        # if gravity spacing not supplied, extract from dataframe
        if grav_spacing is None:
            grid = df.set_index(["northing", "easting"]).to_xarray()[input_grav_column]
            grav_spacing = float(utils.get_grid_info(grid)[0])

        grav = pygmt.surface(
            data=df[["easting", "northing", input_grav_column]],
            region=region,
            spacing=grav_spacing,
            T=0.25,
            # M="0c",
            registration=registration,
        )

        forward = pygmt.surface(
            data=df[["easting", "northing", input_forward_column]],
            region=region,
            spacing=grav_spacing,
            T=0.25,
            # M="0c",
            registration=registration,
        )

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    plot_type = kwargs.get("plot_type", "xarray")
    if plot_type == "xarray":
        cmap = "RdBu_r"
        diff_cmap = "RdBu_r"

        utils.grd_compare(
            grav,
            forward,
            plot=True,
            plot_type=plot_type,
            cmap=cmap,
            robust=robust,
            verbose="e",
            grid1_name="Gravity anomaly",
            grid2_name="Predicted gravity",
            title="Misfit",
            shp_mask=kwargs.get("shp_mask", None),
            hist=kwargs.get("hist", True),
            inset=False,
            cbar_unit="mGal",
            subplot_labels=True,
            RMSE_decimales=1,
            diff_cmap=diff_cmap,
            points=constraints,
            points_style=points_style,
        )
        fig = None

    elif plot_type == "pygmt":
        dif = grav - forward
        rmse = inv_utils.RMSE(dif)

        grids = [grav, dif, forward]
        titles = [
            "Observed gravity",
            f"Misfit, RMSE: {round(rmse,2)} mGal",
            "Predicted gravity",
        ]
        cmaps = ["vik", "vik+h0", "vik"]
        for i, g in enumerate(grids):
            if i == 0:
                fig = None
                origin_shift = "initialize"

            else:
                origin_shift = "xshift"

            fig = maps.plot_grd(
                grid=g,
                fig_height=7,
                cmap=cmaps[i],
                robust=robust,
                title=titles[i],
                cbar_unit="mGal",
                hist=True,
                cbar_yoffset=1.5,
                # grd2cpt=True,
                title_font="18p,Helvetica-bold,black",
                fig=fig,
                origin_shift=origin_shift,
                points=constraints,
                points_style=points_style,
                points_pen=points_pen,
                shp_mask=kwargs.get("shp_mask", None),
            )
            fig.text(
                position="TL",
                justify="BL",
                text=f"{string.ascii_lowercase[i]})",
                font="16p,Helvetica,black",
                offset="j0/.3",
                no_clip=True,
            )
        # fig.show()
    return fig


def anomalies_plotting(
    df_anomalies: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
    registration="g",
    robust=True,
    plot_type="xarray",
    **kwargs,
):
    """
    Plot results from anomaly calculations.

    Parameters
    ----------
    df_anomalies : pd.DataFrame
        Dataframe output from func: inversion.anomalies()
    region : list, optional
        GMT format region to use for the plots, by default is extent of gravity data
    grav_spacing : float, optional
        spacing of the gravity data to create plots, by default None

    Returns
    -------
    list
        Returns a list of gridded anomaly data.
    """

    input_forward_column = kwargs.get("input_forward_column", "forward")
    input_grav_column = kwargs.get("input_grav_column", "grav")
    constraints = kwargs.get("constraints", None)
    shp_mask = kwargs.get("shp_mask", None)

    df = df_anomalies.copy()

    # get columns to include in gridding
    cols_to_grid = [
        input_grav_column,
        "misfit",
        input_forward_column,
        "reg",
        "res",
    ]

    # empty list of grids to append to
    anom_grids = []

    if (region is None) & (grav_spacing is None):
        ds = (
            df.set_index(["northing", "easting"])
            .to_xarray()
            .rename({"easting": "x", "northing": "y"})
        )
        for col in cols_to_grid:
            anom_grids.append(ds[col])

    else:
        for col in cols_to_grid:
            # if inversion region not supplied, extract from dataframe
            if region is None:
                region = vd.get_region((df.easting, df.northing))
            # if gravity spacing not supplied, extract from dataframe
            if grav_spacing is None:
                grid = df.set_index(["northing", "easting"]).to_xarray()[col]
                grav_spacing = float(utils.get_grid_info(grid)[0])

            grid = pygmt.surface(
                data=df[["easting", "northing", col]],
                region=region,
                spacing=grav_spacing,
                T=0.25,
                # M="0c",
                registration=registration,
            )
            anom_grids.append(grid)

    # add grids and names to dictionary
    grid_dict = dict(zip(cols_to_grid, anom_grids))

    # get RMSE value for misfit
    rmse = inv_utils.RMSE(df.res)

    if plot_type == "pygmt":
        _ = utils.grd_compare(
            anom_grids[1],
            anom_grids[3],
            grid1_name="Gravity misfit",
            grid2_name="Regional gravity",
            cbar_label="mGal",
            title="Residual gravity",
            plot=True,
            points=constraints,
            plot_type="pygmt",
            cmap="vik",
            diff_cmap="vik+h0",
            hist=True,
            inset=False,
            points_style="c.15c",
            subplot_labels=True,
            shp_mask=shp_mask,
            robust=robust,
            verbose="q",
        )

    elif plot_type == "xarray":
        # set titles for grids
        plot_titles = [
            "observed gravity (corr)",
            "gravity misfit",
            "forward gravity",
            "regional misfit",
            f"residual misfit: {round(rmse, 2)} mGal",
        ]

        sub_width = 5
        # nrows, ncols = utils.square_subplots(len(cols_to_grid))
        nrows, ncols = 2, 3

        # setup subplot figure
        fig, axs = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width * ncols, sub_width * nrows),
        )

        grav_lims = utils.get_min_max(anom_grids[0], shp_mask)
        forward_lims = utils.get_min_max(anom_grids[2], shp_mask)

        if robust is True:
            grav_lims = (
                np.nanquantile(anom_grids[0], 0.02),
                np.nanquantile(anom_grids[0], 0.98),
            )
            forward_lims = (
                np.nanquantile(anom_grids[2], 0.02),
                np.nanquantile(anom_grids[2], 0.98),
            )

        # get min and max of both grids together
        misfit_vmin = min(grav_lims[0], forward_lims[0])
        misfit_vmax = max(grav_lims[1], forward_lims[1])

        for i, (col, ax) in enumerate(zip(cols_to_grid, axs.ravel())):
            # plot each grid
            if i in [0, 2]:
                anom_grids[i].plot(
                    ax=ax,
                    x="x",
                    y="y",
                    cmap="RdBu_r",
                    vmin=misfit_vmin,
                    vmax=misfit_vmax,
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "anchor": (1, 1),
                        "fraction": 0.05,
                        "pad": 0.04,
                    },
                )
            else:
                if shp_mask is not None:
                    maxabs = vd.maxabs(utils.get_min_max(anom_grids[i], shp_mask))
                    vmin = -maxabs
                    vmax = maxabs
                elif shp_mask is None:
                    vmin, vmax = None, None
                anom_grids[i].plot(
                    ax=ax,
                    x="x",
                    y="y",
                    robust=robust,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="RdBu_r",
                    cbar_kwargs={
                        "orientation": "horizontal",
                        "anchor": (1, 1),
                        "fraction": 0.05,
                        "pad": 0.04,
                    },
                )

            # add subplot titles
            ax.set_title(plot_titles[i])

            # set axes labels and make proportional
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_aspect("equal")

            # add constraint locations as black x's
            if constraints is not None:
                if i in [2, 3, 4]:
                    ax.plot(
                        constraints.easting,
                        constraints.northing,
                        "k.",
                        markersize=kwargs.get("constraint_size", 1),
                        markeredgewidth=1,
                    )

        # add figure title
        fig.suptitle(kwargs.get("title", " "), fontsize=24)

        # delete blank extra subplot
        fig.delaxes(axs[1, 2])

        fig.tight_layout()

    return grid_dict


def inputs_plotting(
    df: pd.DataFrame,
    layer: xr.DataArray,
    robust: bool = True,
    inversion_region: list = None,
    buffer_region: list = None,
    points: pd.DataFrame = None,
    points_style: str = "c0.05c",
    **kwargs,
):
    """
    Plot inputs into the inversion
    """

    residual = df.set_index(["northing", "easting"]).to_xarray().res

    grids = [residual, layer]

    titles = ["Residual misfit", "Starting topography"]

    for i, g in enumerate(grids):
        if i == 0:
            fig = None
            origin_shift = "initialize"
            cmap = "vik+h0"
            cbar_unit = "mGal"
            cbar_label = None
            reverse_cpt = False
        else:
            origin_shift = "xshift"
            cmap = "rain"
            cbar_unit = "m"
            cbar_label = "elevation"
            reverse_cpt = True

        fig = maps.plot_grd(
            grid=g,
            fig_height=9,
            cmap=cmap,
            reverse_cpt=reverse_cpt,
            robust=robust,
            region=buffer_region,
            title=titles[i],
            cbar_unit=cbar_unit,
            cbar_label=cbar_label,
            show_region=inversion_region,
            hist=True,
            cbar_yoffset=3,
            # grd2cpt=True,
            title_font="16p,Helvetica-bold,black",
            fig=fig,
            origin_shift=origin_shift,
            points=points,
            points_style=points_style,
            shp_mask=kwargs.get("shp_mask", None),
        )
        fig.text(
            position="TL",
            justify="BL",
            text=f"{string.ascii_lowercase[i]})",
            font="18p,Helvetica,black",
            offset="j0/.3",
            no_clip=True,
        )

    return fig


def grid_inversion_results(
    misfits,
    topos,
    corrections,
    prisms_ds,
    grav_results,
    region,
    spacing,
    registration,
):
    misfit_grids = []
    for m in misfits:
        grid = pygmt.xyz2grd(
            data=grav_results[["easting", "northing", m]],
            region=region,
            spacing=spacing,
            registration=registration,
            verbose="q",
        )
        misfit_grids.append(grid)

    topo_grids = []
    for t in topos:
        topo_grids.append(
            prisms_ds[t].sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )
        )

    corrections_grids = []
    for m in corrections:
        corrections_grids.append(
            prisms_ds[m].sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )
        )

    return (misfit_grids, topo_grids, corrections_grids)


def plot_inversion_iteration_results(
    grids,
    grav_results,
    topo_results,
    parameters,
    grav_region,
    iterations,
    shp_mask=None,
    constraints=None,
    constraint_size=1,
    topo_cmap_perc=1,
    misfit_cmap_perc=1,
    corrections_cmap_perc=1,
):
    misfit_grids, topo_grids, corrections_grids = grids

    params = parameters.copy()

    # set figure parameters
    sub_width = 5
    nrows, ncols = len(iterations), 3

    # setup subplot figure
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width * ncols, sub_width * nrows),
    )

    # set color limits for each column
    misfit_lims = []
    topo_lims = []
    corrections_lims = []

    for g in misfit_grids:
        misfit_lims.append(utils.get_min_max(g, shapefile=shp_mask))
    for g in topo_grids:
        topo_lims.append(utils.get_min_max(g, shapefile=shp_mask))
    for g in corrections_grids:
        corrections_lims.append(utils.get_min_max(g, shapefile=shp_mask))

    misfit_min = min([i[0] for i in misfit_lims])
    misfit_max = max([i[1] for i in misfit_lims])
    misfit_lim = vd.maxabs(misfit_min, misfit_max) * misfit_cmap_perc

    topo_min = min([i[0] for i in topo_lims]) * topo_cmap_perc
    topo_max = max([i[1] for i in topo_lims]) * topo_cmap_perc

    corrections_min = min([i[0] for i in corrections_lims])
    corrections_max = max([i[1] for i in corrections_lims])
    corrections_lim = (
        vd.maxabs(corrections_min, corrections_max) * corrections_cmap_perc
    )

    for column, j in enumerate(grids):
        for row, y in enumerate(j):
            # if only 1 iteration
            if max(iterations) == 1:
                axes = ax[column]
            else:
                axes = ax[row, column]
            # add iteration number as text
            plt.text(
                -0.1,
                0.5,
                f"Iteration #{iterations[row]}",
                transform=axes.transAxes,
                rotation="vertical",
                ha="center",
                va="center",
                fontsize=20,
            )
            # set colormaps and limits
            if column == 0:  # misfit grids
                cmap = "RdBu_r"
                lims = (-misfit_lim, misfit_lim)
                # lims = (None, None)
                # lims = (-vd.maxabs(j[0]) * perc, vd.maxabs(j[0]) * perc)
                # lims = utils.get_min_max(j[0], shapefile=shp_mask)
                # maxabs = vd.maxabs(
                # utils.get_min_max(j[0], shapefile=shp_mask)
                # )
                # lims = (-maxabs, maxabs)
                robust = True
                # robust = False
                norm = None
            elif column == 1:  # topography grids
                cmap = "gist_earth"
                lims = (topo_min, topo_max)
                # lims = (-vd.maxabs(j[0]) * perc, vd.maxabs(j[0]) * perc)
                # lims = utils.get_min_max(j[0], shapefile=shp_mask)
                robust = True
                # robust = False
                norm = None
            elif column == 2:  # correction grids
                cmap = "RdBu_r"
                lims = (-corrections_lim, corrections_lim)
                # lims = (None, None)
                # lims = utils.get_min_max(j[row], shapefile=shp_mask)
                # maxabs = vd.maxabs(
                #     utils.get_min_max(j[row], shapefile=shp_mask)
                # )
                # lims = (-maxabs, maxabs)
                robust = True
                # robust = False
                # norm = mpl.colors.CenteredNorm(halfrange=corrections_lim, clip=True)
                norm = None
            # plot grids
            j[row].plot(
                ax=axes,
                cmap=cmap,
                norm=norm,
                robust=robust,
                vmin=lims[0],
                vmax=lims[1],
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )

            # add subplot titles
            if column == 0:  # misfit grids
                rmse = inv_utils.RMSE(
                    grav_results[f"iter_{iterations[row]}_initial_misfit"]
                )
                axes.set_title(f"initial misfit RMSE = {round(rmse, 2)} mGal")
            elif column == 1:  # topography grids
                axes.set_title("updated bathymetry")
            elif column == 2:  # correction grids
                rmse = inv_utils.RMSE(
                    topo_results[f"iter_{iterations[row]}_correction"]
                )
                axes.set_title(f"iteration correction RMSE = {round(rmse, 2)} m")

            if constraints is not None:
                if column == 0:  # misfit grids
                    axes.plot(
                        constraints.easting,
                        constraints.northing,
                        "k.",
                        markersize=constraint_size,
                        markeredgewidth=1,
                    )
                elif column == 1:  # topography grids
                    axes.plot(
                        constraints.easting,
                        constraints.northing,
                        "k.",
                        markersize=constraint_size,
                        markeredgewidth=1,
                    )
                elif column == 2:  # correction grids
                    axes.plot(
                        constraints.easting,
                        constraints.northing,
                        "k.",
                        markersize=constraint_size,
                        markeredgewidth=1,
                    )

            # set axes labels and make proportional
            axes.set_xticklabels([])
            axes.set_yticklabels([])
            axes.set_xlabel("")
            axes.set_ylabel("")
            axes.set_aspect("equal")

    # add text with inversion parameter info
    text1, text2, text3 = [], [], []
    params.pop("iter_times")
    for i, (k, v) in enumerate(params.items(), start=1):
        if i <= 6:
            text1.append(f"{k}: {v}\n")
        elif i <= 12:
            text2.append(f"{k}: {v}\n")
        else:
            text3.append(f"{k}: {v}\n")

    text1 = "".join(text1)
    text2 = "".join(text2)
    text3 = "".join(text3)

    # if only 1 iteration
    if max(iterations) == 1:
        plt.text(
            x=0.0,
            y=1.1,
            s=text1,
            transform=ax[0].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text2,
            transform=ax[1].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text3,
            transform=ax[2].transAxes,
        )
    else:
        plt.text(
            x=0.0,
            y=1.1,
            s=text1,
            transform=ax[0, 0].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text2,
            transform=ax[0, 1].transAxes,
        )

        plt.text(
            x=0.0,
            y=1.1,
            s=text3,
            transform=ax[0, 2].transAxes,
        )


def plot_inversion_topo_results(
    prisms_ds,
    shp_mask=None,
    constraints=None,
    inversion_region=None,
    topo_cmap_perc=1,
    constraint_size=1,
):
    initial_topo = prisms_ds.surface

    # list of variables ending in "_layer"
    topos = [s for s in list(prisms_ds.keys()) if "_layer" in s]
    # list of iterations, e.g. [1,2,3,4]
    its = [int(s[5:][:-6]) for s in topos]

    final_topo = prisms_ds[f"iter_{max(its)}_layer"]

    # crop grids to inversion region
    if inversion_region is not None:
        final_topo = final_topo.sel(
            easting=slice(inversion_region[0], inversion_region[1]),
            northing=slice(inversion_region[2], inversion_region[3]),
        )
        initial_topo = initial_topo.sel(
            easting=slice(inversion_region[0], inversion_region[1]),
            northing=slice(inversion_region[2], inversion_region[3]),
        )

    dif = initial_topo - final_topo

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    robust = True

    topo_lims = []
    for g in [initial_topo, final_topo]:
        topo_lims.append(utils.get_min_max(g, shapefile=shp_mask))

    topo_min = min([i[0] for i in topo_lims]) * topo_cmap_perc
    topo_max = max([i[1] for i in topo_lims]) * topo_cmap_perc

    # set figure parameters
    sub_width = 5
    nrows, ncols = 1, 3

    # setup subplot figure
    fig, ax = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width * ncols, sub_width * nrows),
    )

    initial_topo.plot(
        ax=ax[0],
        robust=robust,
        cmap="gist_earth",
        vmin=topo_min,
        vmax=topo_max,
        cbar_kwargs={
            "orientation": "horizontal",
            "anchor": (1, 1),
            "fraction": 0.05,
            "pad": 0.04,
        },
    )
    ax[0].set_title("initial topography")

    dif.plot(
        ax=ax[1],
        robust=True,
        cmap="RdBu_r",
        cbar_kwargs={
            "orientation": "horizontal",
            "anchor": (1, 1),
            "fraction": 0.05,
            "pad": 0.04,
        },
    )
    rmse = inv_utils.RMSE(dif)
    ax[1].set_title(f"difference, RMSE: {round(rmse,2)}m")
    if constraints is not None:
        ax[1].plot(constraints.x, constraints.y, "k.", markersize=constraint_size)
    final_topo.plot(
        ax=ax[2],
        robust=robust,
        cmap="gist_earth",
        vmin=topo_min,
        vmax=topo_max,
        cbar_kwargs={
            "orientation": "horizontal",
            "anchor": (1, 1),
            "fraction": 0.05,
            "pad": 0.04,
        },
    )
    ax[2].set_title("final topography")

    for a in ax:
        a.set_xticklabels([])
        a.set_yticklabels([])
        a.set_xlabel("")
        a.set_ylabel("")
        a.set_aspect("equal")

    fig.tight_layout()

    # _ = utils.grd_compare(
    #     initial_topo,
    #     final_topo,
    #     plot=True,
    #     grid1_name="initial topography",
    #     grid2_name="final topography",

    #     plot_type="xarray",
    #     cmap="gist_earth",
    #     robust=robust,

    #     title="difference",
    #     shp_mask=shp_mask,
    # )


def plot_inversion_grav_results(
    grav_results,
    region,
    spacing,
    registration,
    iterations,
    constraints=None,
    shp_mask=None,
):
    initial_misfit = pygmt.xyz2grd(
        data=grav_results[["easting", "northing", "iter_1_initial_misfit"]],
        region=region,
        spacing=spacing,
        registration=registration,
        verbose="q",
    )

    final_misfit = pygmt.xyz2grd(
        data=grav_results[
            ["easting", "northing", f"iter_{max(iterations)}_final_misfit"]
        ],
        region=region,
        spacing=spacing,
        registration=registration,
        verbose="q",
    )

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    # plot initial misfit - final misfit
    initial_RMSE = inv_utils.RMSE(grav_results["iter_1_initial_misfit"])
    final_RMSE = inv_utils.RMSE(grav_results[f"iter_{max(iterations)}_final_misfit"])

    _ = utils.grd_compare(
        initial_misfit,
        final_misfit,
        plot=True,
        grid1_name=f"Initial misfit: RMSE={round(initial_RMSE, 2)} mGal",
        grid2_name=f"Final misfit: RMSE={round(final_RMSE, 2)} mGal",
        plot_type="xarray",
        cmap="RdBu_r",
        robust=False,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        shp_mask=shp_mask,
        points=constraints,
    )


def plot_inversion_results(
    grav_results,
    topo_results,
    parameters,
    grav_region,
    grav_spacing,
    registration="g",
    iters_to_plot=None,
    constraints=None,
    plot_iter_results=True,
    plot_topo_results=True,
    plot_grav_results=True,
    **kwargs,
):
    # if results are given as filenames (strings), load them
    if isinstance(grav_results, str):
        grav_results = pd.read_csv(
            grav_results,
            sep=",",
            header="infer",
            index_col=None,
            compression="gzip",
        )
    if isinstance(topo_results, str):
        topo_results = pd.read_csv(
            topo_results,
            sep=",",
            header="infer",
            index_col=None,
            compression="gzip",
        )
    if isinstance(parameters, str):
        parameters = np.load(parameters, allow_pickle="TRUE").item()

    prisms_ds = topo_results.set_index(["northing", "easting"]).to_xarray()

    # either set input inversion region or get from input gravity data extent
    if grav_region is None:
        grav_region = vd.get_region((grav_results.easting, grav_results.northing))

    # get lists of columns to grid
    misfits = [s for s in grav_results.columns.to_list() if "initial_misfit" in s]
    topos = [s for s in topo_results.columns.to_list() if "_layer" in s]
    corrections = [s for s in topo_results.columns.to_list() if "_correction" in s]

    # list of iterations, e.g. [1,2,3,4]
    its = [int(s[5:][:-15]) for s in misfits]

    # get on x amonut of iterations to plot
    if iters_to_plot is not None:
        if iters_to_plot > max(its):
            iterations = its
        else:
            iterations = list(np.linspace(1, max(its), iters_to_plot, dtype=int))
    else:
        iterations = its

    # subset columns based on iterations to plot
    misfits = [misfits[i] for i in [x - 1 for x in iterations]]
    topos = [topos[i] for i in [x - 1 for x in iterations]]
    corrections = [corrections[i] for i in [x - 1 for x in iterations]]

    # grid all results
    grids = grid_inversion_results(
        misfits,
        topos,
        corrections,
        prisms_ds,
        grav_results,
        grav_region,
        grav_spacing,
        registration,
    )

    if plot_iter_results is True:
        plot_inversion_iteration_results(
            grids,
            grav_results,
            topo_results,
            parameters,
            grav_region,
            iterations,
            shp_mask=kwargs.get("shp_mask", None),
            constraints=constraints,
            constraint_size=kwargs.get("constraint_size", 2),
            topo_cmap_perc=kwargs.get("topo_cmap_perc", 1),
            misfit_cmap_perc=kwargs.get("misfit_cmap_perc", 1),
            corrections_cmap_perc=kwargs.get("corrections_cmap_perc", 1),
        )

    if plot_topo_results is True:
        plot_inversion_topo_results(
            prisms_ds,
            constraints=constraints,
            inversion_region=grav_region,
            shp_mask=kwargs.get("shp_mask", None),
            topo_cmap_perc=kwargs.get("topo_cmap_perc", 1),
            constraint_size=kwargs.get("constraint_size", 2),
        )

    if plot_grav_results is True:
        plot_inversion_grav_results(
            grav_results,
            grav_region,
            grav_spacing,
            registration,
            iterations,
            constraints=constraints,
            shp_mask=kwargs.get("shp_mask", None),
        )

    return grids


def plot_best_param(study_df, comparison_method, regional_method=None, **kwargs):
    best = study_df.sort_values(by="value").iloc[0]

    if regional_method is None:
        regional_method = best.params_regional_method

    print(f"\n{'':#<10} {regional_method} {'':#>10}")
    print(best)

    rmse, df_anomalies = inv_utils.regional_seperation_quality(
        regional_method=regional_method,
        comparison_method=comparison_method,
        param=best[f"params_{regional_method}"],
        **kwargs,
    )

    if regional_method == "filter":
        title = f"Method: {regional_method} (g{int(best.params_filter/1e3)}km)"
    elif regional_method == "trend":
        title = f"Method: {regional_method} (order={best.params_trend})"
    elif regional_method == "constraints":
        title = f"Method: {regional_method} (tension factor={best.params_constraints})"
    elif regional_method == "eq_sources":
        title = (
            f"Method: {regional_method}"
            f"(Source depth={int(best.params_eq_sources/1e3)}km)"
        )

    score = best.value

    anom_grids = anomalies_plotting(
        df_anomalies,
        region=kwargs.get("inversion_region"),
        grav_spacing=kwargs.get("grav_spacing"),
        title=title + f" Optimization score: {round(score,2)} mGal",
        constraints=kwargs.get("constraints"),
        input_forward_column=kwargs.get("input_forward_column", "forward"),
        input_grav_column=kwargs.get("input_grav_column", "grav"),
    )

    _ = utils.grd_compare(
        kwargs.get("true_regional"),
        anom_grids["reg"],
        plot=True,
        region=kwargs.get("inversion_region"),
        plot_type="xarray",
        cmap="RdBu_r",
        title=title,
        grid1_name="True regional misfit",
        grid2_name="best regional misfit",
    )

    return df_anomalies


def plot_best_params_per_method(studies, comparison_method, **kwargs):
    for k, v in studies.items():
        plot_best_param(studies[k], comparison_method, regional_method=k, **kwargs)


def plot_best_inversion(
    true_surface,
    inversion_region,
    study=None,
    best_params=None,
    grav_spacing=None,
    constraint_points=None,
    plot_iter_results=True,
    plot_topo_results=True,
    plot_grav_results=True,
    plot_type="xarray",
    robust=False,
    results=None,
    **kwargs,
):
    if best_params is None:
        try:
            best_params = optimization.get_best_params_from_study(study)
        except AttributeError:
            pass
    with inv_utils.HiddenPrints():
        results = inv.inversion_RMSE(
            results=results,
            true_surface=true_surface,
            constraints=constraint_points,
            inversion_region=inversion_region,
            plot=True,
            plot_type=plot_type,
            input_grav=kwargs.get("input_grav"),
            input_grav_column=kwargs.get("input_grav_column"),
            prism_layer=kwargs.get("prism_layer"),
            max_iterations=kwargs.get("max_iterations"),
            l2_norm_tolerance=kwargs.get(
                "l2_norm_tolerance", best_params.get("l2_norm_tolerance")
            ),
            delta_l2_norm_tolerance=kwargs.get("delta_l2_norm_tolerance"),
            max_layer_change_per_iter=kwargs.get(
                "max_layer_change_per_iter",
                best_params.get("max_layer_change_per_iter"),
            ),
            deriv_type=kwargs.get("deriv_type", best_params.get("derive_type")),
            solver_type=kwargs.get("solver_type", best_params.get("solver_type")),
            solver_damping=10
            ** [v for k, v in best_params.items() if "damping" in k.lower()][0],
            apply_weights=kwargs.get("apply_weights"),
            weights_after_solving=kwargs.get("weights_after_solving"),
            robust=robust,
        )

    (
        rmse,
        prism_results,
        grav_results,
        params,
        elapsed_time,
        constraints_rmse,
    ) = results

    _ = plot_inversion_results(
        grav_results,
        prism_results,
        params,
        grav_region=inversion_region,
        plot_iter_results=plot_iter_results,
        plot_topo_results=plot_topo_results,
        plot_grav_results=plot_grav_results,
        iters_to_plot=4,
        grav_spacing=grav_spacing,
        constraints=constraint_points,
        registration=kwargs.get("registration"),
    )

    plot_convergence(grav_results, iter_times=params["iter_times"])

    if constraint_points is not None:
        print(
            f"RMSE between surfaces at constraints: " f"{round(constraints_rmse,2)} m"
        )
    return results


def combined_history(
    study,
    target_names,
    include_duration=False,
):
    """
    plot combined optimization history for multiobjective optimizations.
    """
    target_names = target_names.copy()
    figs = []
    for i, j in enumerate(target_names):
        f = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.values[i], target_name=j
        )
        figs.append(f)

    if include_duration is True and "duration" not in target_names:
        f = optuna.visualization.plot_optimization_history(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
        figs.append(f)
        target_names.append("duration")

    if len(target_names) < 2:
        layout = plotly.graph_objects.Layout(
            title="Optimization History Plot",
            yaxis=plotly.graph_objs.layout.YAxis(
                title=target_names[0],
            ),
            xaxis=dict(title="Trial"),
        )
    elif len(target_names) >= 2:
        yaxes = {}
        for i, j in enumerate(target_names, start=1):
            if i == 1:
                pass
            else:
                yax = plotly.graph_objs.layout.YAxis(
                    title=j,
                    overlaying="y",
                    side="left",
                    anchor="free",
                    autoshift=True,
                )
                yaxes[f"yaxis{i}"] = yax
        layout = plotly.graph_objects.Layout(
            title="Optimization History Plot",
            yaxis1=plotly.graph_objs.layout.YAxis(
                title=target_names[0],
                side="right",
            ),
            xaxis=dict(title="Trial"),
            **yaxes,
        )

    # Create figure with secondary x-axis
    fig = plotly.graph_objects.Figure(layout=layout)

    # Add traces
    for i, j in enumerate(target_names):
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=figs[i].data[0]["x"],
                y=figs[i].data[0]["y"],
                name=j,
                mode="markers",
                yaxis=f"y{i+1}",
            )
        )

    fig.update_layout(xaxis_title="Trial", title="Optimization History Plot")

    return fig


def combined_importance(
    study,
    target_names,
    params=None,
    include_duration=False,
):
    """
    plot combined objective value and duration importances. Note, target names must be
    supplied in order which objective function returns them. You can't reorder, or skip
    a value unless it's the last value.
    """
    target_names = target_names.copy()
    figs = []
    try:
        for i, j in enumerate(target_names):
            f = optuna.visualization.plot_param_importances(
                study,
                target=lambda t: t.values[i],
                target_name=j,
                params=params,
            )
            figs.append(f)
        if include_duration is True and "duration" not in target_names:
            f = optuna.visualization.plot_param_importances(
                study,
                target=lambda t: t.duration.total_seconds(),
                target_name="duration",
                params=params,
            )
            figs.append(f)
            target_names.append("duration")

        # Create figure
        fig = plotly.graph_objects.Figure()

        # Add traces
        for i, j in enumerate(target_names):
            fig.add_trace(
                plotly.graph_objects.Bar(
                    x=figs[i].data[0]["x"],
                    y=figs[i].data[0]["y"],
                    name=j,
                    orientation="h",
                )
            )

        fig.update_layout(xaxis_title="Importance", title="Hyperparameter Importance")
    except ValueError:
        print("can't display parameter importances with dynamic search space")
        fig = None
    return fig


def combined_edf(
    study,
    target_names,
    include_duration=False,
):
    """
    plot combined objective value and duration empirical distribution functions
    """
    target_names = target_names.copy()
    figs = []
    for i, j in enumerate(target_names):
        f = optuna.visualization.plot_edf(
            study,
            target=lambda t: t.values[i],
            target_name=j,
        )
        figs.append(f)

    if include_duration is True and "duration" not in target_names:
        f = optuna.visualization.plot_edf(
            study, target=lambda t: t.duration.total_seconds(), target_name="duration"
        )
        figs.append(f)
        target_names.append("duration")

    if len(target_names) < 2:
        layout = plotly.graph_objects.Layout(
            title="Emperical Distribution Function",
            xaxis=plotly.graph_objs.layout.XAxis(
                title=target_names[0],
            ),
            yaxis=dict(title="Cumulative Probability"),
        )

    elif len(target_names) >= 2:
        xaxes = {}
        for i, j in enumerate(target_names, start=1):
            if i == 1:
                pass
            else:
                xax = plotly.graph_objs.layout.XAxis(
                    title=j,
                    overlaying="x",
                    side="bottom",
                    anchor="free",
                    # autoshift=True,
                )
                xaxes[f"xaxis{i}"] = xax
        layout = plotly.graph_objects.Layout(
            title="Emperical Distribution Function",
            xaxis1=plotly.graph_objs.layout.XAxis(
                title=target_names[0],
                side="top",
            ),
            yaxis=dict(title="Cumulative Probability"),
            **xaxes,
        )

    # Create figure with secondary x-axis
    fig = plotly.graph_objects.Figure(layout=layout)

    # Add traces
    for i, j in enumerate(target_names):
        fig.add_trace(
            plotly.graph_objects.Scatter(
                x=figs[i].data[0]["x"],
                y=figs[i].data[0]["y"],
                name=j,
                mode="lines",
                xaxis=f"x{i+1}",
            )
        )

    return fig


def plot_optuna_inversion_figures(
    study,
    target_names,
    include_duration=False,
    params=None,
    seperate_param_importances=False,
    plot_history=True,
    plot_slice=True,
    plot_importance=True,
    plot_edf=True,
    plot_pareto=True,
):
    if plot_history:
        combined_history(
            study,
            target_names,
            include_duration=include_duration,
        ).show()

    if params is None:
        params = [k for k, v in study.get_trials()[0].params.items()]

    if plot_slice:
        for i, j in enumerate(target_names):
            optuna.visualization.plot_slice(
                study, target=lambda t: t.values[i], target_name=j
            ).show()

        if include_duration is True and "duration" not in target_names:
            optuna.visualization.plot_slice(
                study,
                target=lambda t: t.duration.total_seconds(),
                target_name="Execution time",
            ).show()

    if plot_importance:
        if len(params) <= 1:
            pass
        else:
            try:
                if seperate_param_importances is True:
                    combined_importance(
                        study,
                        target_names,
                        params=[
                            "deriv_type",
                            "verde_damping",
                        ],
                        include_duration=include_duration,
                    ).show()

                    combined_importance(
                        study,
                        target_names,
                        params=["deriv_type", "scipy_damping"],
                        include_duration=include_duration,
                    ).show()
                else:
                    combined_importance(
                        study,
                        target_names,
                        params=params,
                        include_duration=include_duration,
                    ).show()
            except AttributeError:
                print("issue with showing importance figure")

    if plot_edf:
        combined_edf(study, target_names, include_duration=include_duration).show()

    if plot_pareto:
        if len(target_names) == 1:
            if "duration" not in target_names:
                if include_duration is True:
                    optuna.visualization.plot_pareto_front(
                        study,
                        targets=lambda t: (t.values[0], t.duration.total_seconds()),
                        target_names=target_names + ["duration"],
                    ).show()

        elif len(target_names) > 1:
            if "duration" not in target_names:
                if include_duration is True:
                    optuna.visualization.plot_pareto_front(
                        study,
                        targets=lambda t: (t.values, t.duration.total_seconds()),
                        target_names=target_names + ["duration"],
                    ).show()
                elif include_duration is False:
                    optuna.visualization.plot_pareto_front(
                        study, target_names=target_names
                    ).show()
            elif "duration" in target_names:
                optuna.visualization.plot_pareto_front(
                    study, target_names=target_names
                ).show()


def merge_mpl_figs(figs, axis, dpi):
    canvases = [f.canvas for f in figs]

    arrays = [np.array(c.buffer_rgba()) for c in canvases]

    a = np.concatenate(arrays, axis=axis)

    backend = mpl.get_backend()
    mpl.use(backend)

    fig, ax = plt.subplots(figsize=(18, 6), dpi=dpi)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.matshow(a)
