from __future__ import annotations

import string

import cmocean.cm as cmo
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from invert4geom import uncertainty
from polartoolkit import utils

import RIS_gravity_inversion.uncertainties as uncert

sns.set_theme()


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
    bathymetry,
    deterministic_bathymetry=None,
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
        region=inversion_region,
        plot=True,
        grid1_name="True topography",
        grid2_name="Stochastic mean inverted topography",
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
        region=inversion_region,
        plot=True,
        grid1_name="Stochastic error",
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

    if deterministic_bathymetry is not None:
        _ = utils.grd_compare(
            np.abs(bathymetry - deterministic_bathymetry),
            stdev,
            region=inversion_region,
            plot=True,
            grid1_name="Deterministic error",
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


def plot_2var_ensemble(
    df,
    x,
    y,
    figsize=(9, 6),
    x_title=None,
    y_title=None,
    background="score",
    background_title=None,
    background_cmap=cmo.matter,
    background_lims=None,
    background_cpt_lims=None,
    points_color=None,
    points_share_cmap=False,
    points_size=None,
    points_scaling=1,
    points_label=None,
    points_title=None,
    points_color_log=False,
    points_cmap=cmo.gray_r,
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
    colorbar: bool = True,
):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
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
        # norm=mpl.colors.BoundaryNorm(
        #   np.linspace(background_lims[0],
        #   background_lims[1],
        #   10), cmap.N),
        edgecolors="w",
        linewidth=0.5,
        add_colorbar=False,
        # xticks=df[x].unique().round(-2),
        # yticks=df[y].unique().astype(int),
    )
    # ax.set_xticks(ax.get_xticks()[1:-1])

    if colorbar:
        cax = fig.add_axes([1, 0.2, 0.05, 0.8])
        cbar = plt.colorbar(plot_background, extend="both", cax=cax)
        cbar.set_label(background_title)

    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")

    # x = df[x].unique()
    # y = df[y].unique()
    # plt.xticks(x[:-1]+0.5)
    # plt.yticks(y[:-1]+0.5)
    y_ticks = list(df[y].unique()[::2])  # .append(df[y].unique()[-1])
    y_ticks.append(df[y].unique()[-1])
    ax.set_yticks(y_ticks)

    # x_ticks = list(df[x].unique()[::2])#.append(df[y].unique()[-1])
    # x_ticks.append(df[x].unique()[-1])
    # ax.set_xlim(df[x].min(), df[x].max())
    # ax.set_xticks(x_ticks)

    if plot_contours is not None:
        contour = grd.plot.contour(
            levels=plot_contours,
            colors=contour_color,
            ax=ax,
        )
        if colorbar:
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
        if (points_share_cmap is False) and (colorbar):
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
    markersize=7,
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
    trend_line_text_loc=(0.05, 0.95),
    flipx=False,
    colorbar: bool = True,
    ax=None,
    plot_elbows=False,
):
    sns.set_theme()

    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize, constrained_layout=False)
    else:
        fig, _ = plt.subplots(figsize=figsize, constrained_layout=False)
        ax1 = ax

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
            markersize=markersize,
            color=plt.cm.viridis(norm(name)),
        )
        if trend_line and slope_min_max:
            z = np.polyfit(group[x], group[y], 1)
            slopes.append(z[0])
            lines.append(np.poly1d(z)(results[x]))
        if plot_elbows:
            from kneebow.rotor import Rotor

            rotor = Rotor()
            rotor.fit_rotate(group[[x, y]])
            elbow_ind = rotor.get_elbow_index()
            ax1.scatter(
                x=group.iloc[elbow_ind][x],
                y=group.iloc[elbow_ind][y],
                marker="*",
                edgecolor="black",
                linewidth=0.5,
                color=plt.cm.viridis(norm(name)),
                s=60,
                zorder=20,
            )
            print(f"Elbow x value: {group[x].iloc[elbow_ind]}")

    if trend_line:
        if slope_min_max:
            text = rf"$min\ slope={min(slopes):.3g}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1],
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )
            ax1.plot(results[x], lines[np.argmin(slopes)], "r", lw=1)

            text = rf"$max\ slope={max(slopes):.3g}$"
            plt.gca().text(
                trend_line_text_loc[0],
                trend_line_text_loc[1] - 0.05,
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
                trend_line_text_loc[0],
                trend_line_text_loc[1] - 0.1,
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
                trend_line_text_loc[0],
                trend_line_text_loc[1],
                text,
                transform=plt.gca().transAxes,
                fontsize=10,
                verticalalignment="top",
            )

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)

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

    if (horizontal_line is not None) & (horizontal_line_label is not None):
        plt.legend(loc=horizontal_line_label_loc)

    if flipx:
        ax1.invert_xaxis()

    if colorbar:
        # pass
        cax = fig.add_axes([0.93, 0.1, 0.05, 0.8])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label)

    plt.title(plot_title)
    # plt.tight_layout()

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
