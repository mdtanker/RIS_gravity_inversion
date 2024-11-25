from __future__ import annotations

import string

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pyvista as pv
import seaborn as sns
import verde as vd
from invert4geom import plotting, uncertainty
from invert4geom import utils as inv_utils
from plotly.subplots import make_subplots
from polartoolkit import maps, profiles, utils

import RIS_gravity_inversion.uncertainties as uncert

sns.set_theme()


def plot_latin_hypercube(
    params_dict,
    plot_individual_dists=True,
    plot_correlations=True,
    plot_1d_projection=True,
):
    df = pd.DataFrame(
        [params_dict[x]["sampled_values"] for x in params_dict],
    ).transpose()

    df.columns = params_dict.keys()

    # plot individual variables
    if plot_individual_dists is True:
        _fig, axes = plt.subplots(
            1,
            len(df.columns),
            figsize=(3 * len(df.columns), 1.8),
        )

        for i, j in enumerate(df.columns):
            sns.kdeplot(
                ax=axes[i],
                data=df,
                x=j,
            )
            sns.rugplot(ax=axes[i], data=df, x=j, linewidth=2.5, height=0.07)
            axes[i].set_xlabel(j.replace("_", " ").capitalize())
            axes[i].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            axes[i].set_ylabel(None)

        plt.show()

    # n = np.shape(df)[0]
    dim = np.shape(df)[1]

    param_values = df.values

    problem = {
        "num_vars": dim,
        "names": [i.replace("_", " ") for i in df.columns],
        "bounds": [[-1, 1]] * dim,
    }

    # Rescale to the unit hypercube for the analysis
    sample = inv_utils.scale_normalized(param_values, problem["bounds"])

    if plot_correlations:
        # Correlation properties
        [z_mk, pval_mk] = uncert.mann_kendall_test_sample(sample)
        [rho, pval_pr] = uncert.pearson_test_sample(sample)

        correlation_plots(z_mk, pval_mk, problem["names"])
        correlation_plots(rho, pval_pr, problem["names"])

    # Space-filling properties

    # 1D projection
    if plot_1d_projection:
        x = uncert.projection_1D(sample, problem["names"])
        print("\nOn a [0, 1] scale 1D coverage is: " + str(x))

    # 2D projection (manageable number of variables, here fixed at 11)
    if problem["num_vars"] < 11:
        plotting.projection_2d(sample, problem["names"])

    # L2-star discrepancy
    # dl2 = uncert.spac-e_filling_measures_discrepancy(sample)
    # print('\nL2-star discrepancy is: ' + str(dl2))

    # Minimal distance between any two points
    # d = uncert.space_filling_measures_min_distance(sample, 1)
    # print('\nThe minimal distance between two points is: ' + str(d) + '\n')


def correlation_plots(z_test, p_val, var_names):
    """
    Get and plot results for within-sample correlation test, based on
    1) test results z_test (Figure 1)
    2) statistical significance pval (Figure 2)
    other inputs are var_names
    """
    # Local variables
    nvar = len(var_names)
    pval = 1 - (
        p_val + np.matlib.eye(nvar)
    )  # Transformation convenient for plotting below

    ###################################################
    # Figure 1: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))
    res_mat[:, 0:-1] = z_test

    # Center the color scale on 0
    res_mat[0, nvar] = max(np.amax(z_test), -np.amin(z_test))
    res_mat[1, nvar] = -res_mat[0, nvar]

    # Plotting Pearson test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=plt.cm.bwr)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.set_xticks(np.linspace(0.5, nvar - 0.5, num=nvar))
    ax.set_xticklabels(var_names, rotation=20, ha="right")
    ax.set_yticks(np.linspace(0.5, nvar - 0.5, num=nvar))
    ax.set_yticklabels(var_names[::-1])
    ax.tick_params(axis="x", top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=False)
    plt.title("Rank correlation between variables' sampled values", size=13, y=1.07)
    plt.colorbar()
    plt.show()
    # plt.clf()

    ###################################################
    # Figure 2: correlations

    # Matrix to plot
    res_mat = np.zeros((nvar, nvar + 1))

    # Set the thresholds at +-95%, 99%, and 99.9% significance levels
    bin_thresholds = [0.9, 0.95, 0.99, 0.999]
    n_sig = len(bin_thresholds)
    res_mat[:, 0:-1] = uncert.binning(pval, bin_thresholds)

    # Set the color scale
    res_mat[0, nvar] = n_sig

    # Common color map
    cmap = plt.cm.Greys
    cmaplist = [cmap(0)]
    for i in range(n_sig):
        cmaplist.append(cmap(int(255 * (i + 1) / n_sig)))
    mycmap = cmap.from_list("Custom cmap", cmaplist, n_sig + 1)

    # Plot background mesh
    mesh_points = np.linspace(0.5, nvar - 0.5, num=nvar)
    for i in range(nvar):
        plt.plot(
            np.arange(0, nvar + 1),
            mesh_points[i] * np.ones(nvar + 1),
            c="k",
            linewidth=0.3,
            linestyle=":",
        )
        plt.plot(
            mesh_points[i] * np.ones(nvar + 1),
            np.arange(0, nvar + 1),
            c="k",
            linewidth=0.3,
            linestyle=":",
        )

    # Plotting MK test results
    plt.imshow(res_mat, extent=[0, nvar + 1, 0, nvar], cmap=mycmap)

    # Plot specifications
    ax = plt.gca()
    ax.set_xlim(0, nvar)  # Last column only to register min and max values for colorbar
    ax.set_xticks(mesh_points)
    ax.set_xticklabels(var_names, rotation=20, ha="right")
    ax.set_yticks(mesh_points)
    ax.set_yticklabels(var_names[::-1])
    ax.tick_params(axis="x", top=True, bottom=True, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=True, labelleft=True, labelright=False)
    plt.title("Significance of the rank correlations", size=13, y=1.07)
    colorbar = plt.colorbar()
    colorbar.set_ticks(
        np.linspace(res_mat[0, nvar] / 10, 9 * res_mat[0, nvar] / 10, num=n_sig + 1)
    )
    cb_labels = ["None"]
    for i in range(n_sig):
        cb_labels.append(str(bin_thresholds[i] * 100) + "%")
    colorbar.set_ticklabels(cb_labels)
    plt.show()
    # plt.clf()


def uncert_plots(
    results,
    inversion_region,
    spacing,
    bathymetry,
    constraint_points=None,
    weight_by=None,
):
    if (weight_by == "constraints") & (constraint_points is None):
        msg = "must provide constraint_points if weighting by constraints"
        raise ValueError(msg)

    stats_ds = uncertainty.merged_stats(
        results=results,
        plot=True,
        constraints_df=constraint_points,
        weight_by=weight_by,
        region=inversion_region,
    )

    try:
        mean = stats_ds.weighted_mean
        stdev = stats_ds.weighted_stdev
    except AttributeError:
        mean = stats_ds.z_mean
        stdev = stats_ds.z_stdev

    _ = utils.grd_compare(
        bathymetry,
        mean,
        region=vd.pad_region(inversion_region, -spacing),
        plot=True,
        grid1_name="True topography",
        grid2_name="Inverted topography",
        robust=True,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        grounding_line=False,
        reverse_cpt=True,
        cmap="rain",
        points=constraint_points,
        points_style="x.3c",
    )
    _ = utils.grd_compare(
        np.abs(bathymetry - mean),
        stdev,
        region=vd.pad_region(inversion_region, -spacing),
        plot=True,
        grid1_name="True error",
        grid2_name="Stochastic uncertainty",
        cmap="thermal",
        robust=True,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        grounding_line=False,
        points=constraint_points,
        points_style="x.3c",
        points_fill="white",
    )
    return stats_ds


def plot_inversion_ensemble_results_profile(
    prism_results,
    grav_results,
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
    layers_frame=None,
    data_frame=None,
    **kwargs,
):
    # turn dataframe into datasets
    final_surfaces = []
    surface_errors = []
    for j in prism_results:
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
        for j in grav_results:
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
        background = true_surface if true_surface is not None else final_surfaces[0]

    # extract layers for profiles plot
    topo_names = [f"{x} bed error" for x in name_prefixes]
    topo_grids = surface_errors

    if len(topo_grids) <= 4:
        colors = ["black", "blue", "purple", "orange"]
    else:
        colors = [utils.random_color() for x in topo_grids]

    if kwargs.get("data_color") is not None:
        data_colors = kwargs.get("data_color")
    else:
        data_colors = colors

    if kwargs.get("layers_color") is not None:
        layers_colors = kwargs.get("layers_color")
    else:
        layers_colors = colors

    layers_dict = profiles.make_data_dict(
        names=topo_names,
        grids=topo_grids,
        colors=layers_colors,
    )

    if plot_regional_error:
        data_names = [f"{x} regional error" for x in name_prefixes]
        data_grids = regional_errors
        data_dict = profiles.make_data_dict(
            names=data_names,
            grids=data_grids,
            colors=data_colors,
        )
    else:
        data_dict = None

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    if layers_frame is None:
        layers_frame = ["nesW", "xafg", "ya+lbed error (m)"]
    if data_frame is None:
        data_frame = ["neSW", "ya+lregional error (mGal)", "xag+lDistance (m)"]

    fig, _df_layers, _df_data = profiles.plot_profile(
        "points",
        fig_height=8,
        data_height=3.5,
        start=start,
        stop=stop,
        num=1000,
        subplot_orientation="vertical",
        add_map=True,
        map_background=background,
        map_cmap="batlowW",
        inset=False,
        gridlines=False,
        coast=False,
        map_buffer=map_buffer,
        map_points=constraints,
        map_points_style="x.3c",
        map_points_pen="1p,black",
        map_line_pen="2p,black",
        data_dict=data_dict,
        share_yaxis=True,
        data_frame=data_frame,
        # data_pen_thickness=[1, 1.5, 1],
        # data_pen_style=[None,None,"4_2:2p"],
        layers_dict=layers_dict,
        fill_layers=False,
        layers_frame=layers_frame,
        # layer_pen_thickness=[1, 1.5, 1],
        # layers_pen_style=[None,None,"4_2:2p"],
        frame=True,
        start_label="A",
        end_label="A' ",
        **kwargs,
    )
    return fig


def plot_regional_gridding_ensemble_profile(
    grav_anomalies,
    name_prefixes,
    start=None,
    stop=None,
    true_regional=None,
    constraints=None,
    background=None,
    map_buffer=0.01,
    layers_frame=None,
    data_frame=None,
    **kwargs,
):
    # turn dataframe into datasets
    regionals = []
    residuals = []
    regional_errors = []
    for j in grav_anomalies:
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
        background = true_regional if true_regional is not None else grav_ds.reg

    # extract layers for profiles plot
    topo_names = [f"{x}, regional" for x in name_prefixes]
    topo_grids = regionals

    if len(topo_grids) <= 4:
        colors = ["black", "blue", "purple", "orange"]
    else:
        colors = [utils.random_color() for x in topo_grids]

    if kwargs.get("data_color") is not None:
        data_colors = kwargs.get("data_color")
    else:
        data_colors = colors

    if kwargs.get("layers_color") is not None:
        layers_colors = kwargs.get("layers_color")
    else:
        layers_colors = colors

    layers_dict = profiles.make_data_dict(
        names=topo_names, grids=topo_grids, colors=layers_colors
    )

    data_names = [f"{x}, residual" for x in name_prefixes]
    data_grids = residuals

    data_dict = profiles.make_data_dict(
        names=data_names, grids=data_grids, colors=data_colors
    )

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    if layers_frame is None:
        layers_frame = ["nesW", "xafg", "ya+lelevation (m)"]
    if data_frame is None:
        data_frame = ["neSW", "ya+lgravity (mGal)", "xag+lDistance (m)"]

    fig, _df_layers, _df_data = profiles.plot_profile(
        "points",
        fig_height=8,
        data_height=4,
        start=start,
        stop=stop,
        num=1000,
        subplot_orientation="vertical",
        add_map=True,
        map_background=background,
        map_cmap="batlowW",
        inset=False,
        gridlines=False,
        coast=False,
        map_buffer=map_buffer,
        map_points=constraints,
        map_points_style="x.3c",
        map_points_pen="1p,black",
        map_line_pen="2p,black",
        data_dict=data_dict,
        share_yaxis=True,
        data_frame=data_frame,
        data_pen_thickness=[1, 1.5, 1],
        data_pen_style=[None, None, "4_2:2p"],
        layers_dict=layers_dict,
        fill_layers=False,
        layers_frame=layers_frame,
        layer_pen_thickness=[1, 1.5, 1],
        layers_pen_style=[None, None, "4_2:2p"],
        frame=True,
        start_label="A",
        end_label="A' ",
        **kwargs,
    )
    return fig


def plot_inversion_results_profile(
    prism_results,
    grav_results,
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

    initial_surface = prism_ds.starting_topo
    final_surface = prism_ds.topo

    # cols = [s for s in grav_results.columns.to_list() if "_final_misfit" in s]
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

    # extract layers for profiles plot
    topo_names = [
        "Starting bed",
        "Inverted bed",
    ]
    topo_grids = [
        initial_surface,
        final_surface,
    ]
    layers_dict = profiles.make_data_dict(
        names=topo_names,
        grids=topo_grids,
        colors=["black", "blue"],
    )

    data_names = [
        "Initial residual gravity",
        "Final forward gravity",
        "Observed gravity",
    ]

    data_grids = [
        initial_residual,
        final_forward + grav_ds.reg,
        grav_ds[observed_grav_column],
    ]

    data_dict = profiles.make_data_dict(
        names=data_names,
        grids=data_grids,
        colors=["black", "blue", "red", "purple", "gray", "brown"],
    )

    if true_surface is not None:
        layers_dict["True bed"] = dict(name="True bed", grid=true_surface, color="red")

        if plot_bed_misfit is True:
            data_dict["Bed misfit"] = dict(
                name="Bed misfit", grid=(true_surface - final_surface), color="purple"
            )

    if constraints is not None:
        constraints = constraints.rename(columns={"easting": "x", "northing": "y"})

    if layers_frame is None:
        layers_frame = ["nesW", "xafg", "ya+lelevation (m)"]
    if data_frame is None:
        data_frame = ["neSW", "ya+lgravity (mGal)", "xag+lDistance (m)"]

    fig, _df_layers, _df_data = profiles.plot_profile(
        "points",
        fig_height=6.5,
        data_height=3,
        start=start,
        stop=stop,
        num=1000,
        add_map=kwargs.get("map", True),
        map_background=background,
        map_cmap="batlowW",
        inset=False,
        gridlines=False,
        coast=False,
        map_buffer=map_buffer,
        map_points=constraints,
        map_points_style="x.3c",
        map_points_pen="1p,black",
        map_line_pen="2p,black",
        data_dict=data_dict,
        share_yaxis=True,
        data_frame=data_frame,
        data_pen_thickness=[1, 1.5, 1],
        data_pen_style=[None, None, "4_2:2p"],
        layers_dict=layers_dict,
        fill_layers=False,
        layers_frame=layers_frame,
        layer_pen_thickness=[1, 1.5, 1],
        layers_pen_style=[None, None, "4_2:2p"],
        frame=True,
        start_label="A",
        end_label="A' ",
        **kwargs,
    )

    return fig


def plot_2var_ensemble(
    df,
    x,
    y,
    figsize=(9, 6),
    x_title=None,
    y_title=None,
    background="score",
    background_title=None,
    background_cmap="cmo.matter",
    background_lims=None,
    background_cpt_lims=None,
    points_color=None,
    points_share_cmap=False,
    points_size=None,
    points_scaling=1,
    points_label=None,
    points_title=None,
    points_color_log=False,
    points_cmap="cmo.gray_r",
    points_lims=None,
    points_edgecolor="black",
    background_color_log=False,
    background_robust=False,
    points_robust=False,
    plot_contours=None,
    contour_color="black",
    plot_title=None,
    logx=False,
    logy=False,
    flipx=False,
):
    fig, ax = plt.subplots(figsize=figsize)
    df = df.copy()

    ax.grid(which="major", visible=False)

    norm = mpl.colors.LogNorm() if background_color_log is True else None

    cmap = plt.get_cmap(background_cmap)

    if background_lims is None:
        background_lims = utils.get_min_max(df[background], robust=background_robust)
    else:
        cmap.set_under("g")

    grd = df.set_index([y, x]).to_xarray()[background]

    if background_cpt_lims is not None:
        background_lims = background_cpt_lims

    plot_background = grd.plot(
        ax=ax,
        cmap=cmap,
        vmin=background_lims[0],
        vmax=background_lims[1],
        norm=norm,
        edgecolors="w",
        linewidth=0.5,
        add_colorbar=False,
        # xticks=df[x].unique().round(-2),
        # yticks=df[y].unique().astype(int),
    )
    # ax.set_xticks(ax.get_xticks()[1:-1])

    cbar = fig.colorbar(plot_background, extend="both")
    cbar.set_label(background_title)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    # x = df[x].unique()
    # y = df[y].unique()
    # plt.xticks(x[:-1]+0.5)
    # plt.yticks(y[:-1]+0.5)

    if plot_contours is not None:
        contour = grd.plot.contour(
            levels=plot_contours,
            colors=contour_color,
            ax=ax,
        )

        cbar.add_lines(contour)

    if (points_color is not None) or (points_size is not None):
        if points_color is None:
            points_color = "b"
        if points_size is None:
            points_size = 50

        points_cmap = plt.get_cmap(points_cmap)

        if points_share_cmap is True:
            points_cmap = cmap
            vmin = background_lims[0]
            vmax = background_lims[1]
        else:
            if points_lims is None:
                points_lims = utils.get_min_max(df[points_color], robust=points_robust)
            else:
                points_cmap.set_under("g")

            if points_color_log is True:
                norm = mpl.colors.LogNorm(
                    vmin=points_lims[0],
                    vmax=points_lims[1],
                )
                vmin = None
                vmax = None
            else:
                norm = None
                vmin = points_lims[0]
                vmax = points_lims[1]

        points = ax.scatter(
            df[x],
            df[y],
            s=points_size * points_scaling,
            c=df[points_color],
            cmap=points_cmap,
            zorder=10,
            edgecolors=points_edgecolor,
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            label=points_label,
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
        if points_share_cmap is False:
            cbar2 = fig.colorbar(points, extend="both")
            try:  # noqa: SIM105
                cbar2.set_label(points_color.name)
            except AttributeError:
                pass
            if points_title is None:
                points_title = points_label
            cbar2.set_label(points_title)
        # else:
        #     cbar2 = cbar

        if points_label is not None:
            ax.legend(
                loc="lower left",
                bbox_to_anchor=(1, 1),
            )

    if flipx:
        ax.invert_xaxis()

    # ax.ticklabel_format(useOffset=False, style='plain')

    if x_title is None:
        x_title = x
    if y_title is None:
        y_title = y
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)

    ax.set_title(plot_title, fontsize=16)

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
        try:  # noqa: SIM105
            cbar2.set_label(points_color.name)
        except AttributeError:
            pass

    norm = mpl.colors.LogNorm() if background_color_log is True else None

    lims = utils.get_min_max(df[background], robust=background_robust)
    extent = [plt.xlim()[0], plt.xlim()[1], plt.ylim()[1], plt.ylim()[0]]

    plot_background = ax.imshow(
        df.pivot_table("cell_size", "noise", background),
        # alpha=.8,
        extent=extent,
        aspect="auto",
        cmap=plt.get_cmap("cmo.matter", 10),
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
        data = df.pivot_table("cell_size", "noise", background)
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


def plot_ensemble_as_lines(
    results,
    x,
    y,
    groupby_col,
    figsize=(5, 3.5),
    x_lims=None,
    y_lims=None,
    x_label=None,
    y_label="Bathymetry RMSE (m)",
    cbar_label=None,
    logy=False,
    logx=False,
    trend_line=False,
    horizontal_line=None,
    horizontal_line_label=None,
    horizontal_line_label_loc="best",
    horizontal_line_color="gray",
    plot_title=None,
    slope_min_max=False,
    slope_mean=False,
):
    sns.set_theme()

    fig, ax1 = plt.subplots(figsize=figsize)

    grouped = results.groupby(groupby_col)

    norm = plt.Normalize(
        vmin=results[groupby_col].values.min(), vmax=results[groupby_col].values.max()
    )
    slopes = []
    lines = []
    for _, (name, group) in enumerate(grouped):
        ax1.plot(
            group[x],
            group[y],
            ".-",
            markersize=7,
            color=plt.cm.viridis(norm(name)),
        )
        if trend_line and slope_min_max:
            z = np.polyfit(group[x], group[y], 1)
            slopes.append(z[0])
            lines.append(np.poly1d(z)(results[x]))

    if trend_line:
        if slope_min_max:
            text = rf"$min\ slope={min(slopes):.3g}$"
            plt.gca().text(
                0.05,
                0.95,
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmin(slopes)], "r", lw=1)

            text = rf"$max\ slope={max(slopes):.3g}$"
            plt.gca().text(
                0.05,
                0.90,
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmax(slopes)], "r", lw=1)

            # text = f"$mean\ slope={np.mean(slopes):.3g}$"
            # plt.gca().text(
            #     0.05,
            #     0.85,
            #     text,
            #     transform=plt.gca().transAxes,
            #     fontsize=10,
            #     verticalalignment="top",
            # )
            # ax1.plot(results[x], lines[np.argmean(slopes)], "r", lw=1)
        if slope_mean:
            text = rf"$mean\ slope={np.median(slopes):.3g}$"
            plt.gca().text(
                0.05,
                0.85,
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argsort(slopes)[len(slopes) // 2]], "r", lw=1)

        else:
            z = np.polyfit(results[x], results[y], 1)
            y_hat = np.poly1d(z)(results[x])

            ax1.plot(results[x], y_hat, "r", lw=1)
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
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label(cbar_label)

    if horizontal_line is not None:
        plt.axhline(
            y=horizontal_line,
            linewidth=2,
            color=horizontal_line_color,
            linestyle="dashed",
            label=horizontal_line_label,
        )

    ax1.set_xlabel(
        x_label,
    )
    # ax1.set_xticks(list(ax1.get_xticks()) + list(ax1.get_xlim()))

    if x_lims is not None:
        ax1.set_xlim(x_lims)
    if y_lims is not None:
        ax1.set_ylim(y_lims)
    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    ax1.set_ylabel(y_label)

    if horizontal_line is not None:
        plt.legend(loc=horizontal_line_label_loc)

    plt.title(plot_title)
    plt.tight_layout()

    return fig


def plot_1var_ensemble(
    df,
    x,
    y,
    title,
    xlabel,
    ylabel,
    highlight_points=None,
    horizontal_line=None,
    horizontal_line_label=None,
    starting_error=None,
    logy=False,
    logx=False,
):
    sns.set_theme()

    df = df.copy()

    df = df.sort_values(x)

    fig, ax1 = plt.subplots(figsize=(5, 3.5))
    plt.title(title)

    if horizontal_line is not None:
        plt.axhline(
            y=horizontal_line,
            linewidth=2,
            color="gray",
            linestyle="dashed",
            label=horizontal_line_label,
        )

    ax1.plot(df[x], df[y], "bd-", markersize=7, label="inverted")
    ax1.set_xlabel(
        xlabel,
        # color="b",
    )

    if starting_error is not None:
        ax1.plot(
            df[x],
            df[starting_error],
            "g.-",
            markersize=10,
            label="starting",
            zorder=1,
        )

    if logy:
        ax1.set_yscale("log")
    if logx:
        ax1.set_xscale("log")

    ax1.set_ylabel(ylabel)
    # ax1.tick_params(axis="x", colors='b', which="both")
    ax1.set_zorder(2)

    if highlight_points is not None:
        for i, ind in enumerate(highlight_points):
            plt.plot(
                df[x].loc[ind],
                df[y].loc[ind],
                "s",
                markersize=12,
                color="b",
                zorder=3,
            )
            plt.annotate(
                string.ascii_lowercase[i + 1],
                (df[x].loc[ind], df[y].loc[ind]),
                fontsize=15,
                color="white",
                ha="center",
                va="center",
                zorder=4,
            )

    plt.legend(loc="best")
    plt.tight_layout()


def plot_constraint_spacing_ensemble(
    results,
    plot_constraints=False,
    starting_error=None,
    highlight_points=None,
    subplot_label=None,
    horizontal_line=None,
    logy=False,
    logx=False,
):
    sns.set_theme()

    df = results.copy()

    df = df.sort_values("spacing")

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

    if starting_error is not None:
        ax1.plot(
            df.spacing / 1e3,
            df[starting_error],
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
            cmap="batlowW",
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
    region: list | None = None,
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

    opacity = kwargs.get("opacity")

    for i, j in enumerate(prisms):
        # if region is given, clip model
        if region is not None:
            j = j.sel(  # noqa: PLW2901
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3]),
            )

        # turn prisms into pyvist object
        pv_grid = j.prism_layer.to_pyvista()

        # clip corner out of model to help visualize
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

        trans = opacity[i] if opacity is not None else None

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
    region: list | None = None,
    cmap: str = "viridis",
    plot_type: str = "2D",
    layers_for_3d: list | None = None,
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
            layers_to_plot = {k: layers[k] for k in layers_for_3d}
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
            cmap=kwargs.get("cmap"),
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
