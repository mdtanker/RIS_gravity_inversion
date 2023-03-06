from typing import TYPE_CHECKING, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import pygmt
import pyvista as pv
import seaborn as sns
import verde as vd
from antarctic_plots import maps, utils, profile
from plotly.subplots import make_subplots
import RIS_gravity_inversion.inversion as inv
import RIS_gravity_inversion.utils as inv_utils
import optuna
import copy
import xarray as xr
import seaborn as sns
import plotly
import os, sys
import warnings


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


def misfit_plotting(
    df: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
    registration="g",
    constraints: pd.DataFrame = None,
    **kwargs,
):
    """
    Plot results from anomaly calculations.

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

    plot_type = kwargs.get('plot_type', 'xarray')
    if plot_type == 'xarray':
        cmap='RdBu_r'
        diff_cmap="RdBu_r"
    elif plot_type == 'pygmt':
        cmap='vik'
        diff_cmap="vik+h0"
    
    if constraints is not None:
        constraints = constraints.rename(columns = {"easting":"x","northing":"y"})
        
    utils.grd_compare(
        grav,
        forward,
        plot=True,
        plot_type=plot_type,
        cmap=cmap,
        robust=False,
        verbose="e",
        grid1_name="Corrected observed gravity",
        grid2_name="Forward gravity",
        title="Gravity misfit",
        shp_mask=kwargs.get("shp_mask", None),
        hist=kwargs.get('hist', True),
        inset=False,
        cbar_unit="mGal",
        subplot_labels=True,
        RMSE_decimates=1,
        diff_cmap=diff_cmap,
        points=constraints,
    )


def anomalies_plotting(
    df_anomalies: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
    registration="g",
    plot_dists: bool = False,
    plot_power_spectrums: bool = False,
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
    plot_dists : bool, optional
        Choose whether to plot the resulting distributions, by default False
    plot_power_spectrums : bool, optional
        Choose to plot radially average power spectrum of layers, by default False

    Returns
    -------
    list
        Returns a list of gridded anomaly data.
    """

    input_forward_column = kwargs.get("input_forward_column", "forward")
    input_grav_column = kwargs.get("input_grav_column", "grav")

    # if inversion region not supplied, extract from dataframe
    if region is None:
        region = vd.get_region((df_anomalies.easting, df_anomalies.northing))

    # if gravity spacing not supplied, extract from dataframe
    if grav_spacing is None:
        grid = df_anomalies.set_index(["northing", "easting"]).to_xarray()[input_grav_column]
        grav_spacing = float(utils.get_grid_info(grid)[0])

    constraints = kwargs.get("constraints", None)

    # get columns to include in gridding
    cols_to_grid = [
        input_grav_column,
        "misfit",
        input_forward_column,
        "reg",
        "res",
    ]

    # get RMSE value for misfit
    rmse = inv.RMSE(df_anomalies.res)

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

    # empty list of grids to append to
    anom_grids = []
    for col in cols_to_grid:
        grid = pygmt.surface(
                data=df_anomalies[["easting", "northing", col]],
                region=region,
                spacing=grav_spacing,
                T=0.25,
                # M="0c",
                registration=registration,
            )
        anom_grids.append(grid)

    shp_mask = kwargs.get("shp_mask", None)

    grav_lims = utils.get_min_max(anom_grids[0], shp_mask)
    forward_lims = utils.get_min_max(anom_grids[2], shp_mask)

    # get min and max of both grids together
    vmin = min(grav_lims[0], forward_lims[0])
    vmax = max(grav_lims[1], forward_lims[1])

    for i, (col, ax) in enumerate(zip(cols_to_grid, axs.ravel())):
        # plot each grid
        if i in [0, 2]:
            anom_grids[i].plot(
                ax=ax,
                x="x",
                y="y",
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )
        else:
            anom_grids[i].plot(
                ax=ax,
                x="x",
                y="y",
                robust=True,
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
                    "kx",
                    markersize=kwargs.get("constraint_size", 4),
                    markeredgewidth=1,
                )
            
    # add figure title
    fig.suptitle(kwargs.get("title", " "), fontsize=24)

    # delete blank extra subplot
    fig.delaxes(axs[1,2])
    
    fig.tight_layout()
    
    # add grids and names to dictionary
    grid_dict = dict(zip(cols_to_grid, anom_grids))

    if plot_dists is True:
        # get column to include in plots
        df = df_anomalies[cols_to_grid]
        # reorganize
        data = df.melt(var_name="gravity type")
        # layout the subplot grid
        g = sns.FacetGrid(
            data, col="gravity type"
        )  # , sharex=False, sharey=False)#, col_wrap=5)
        # add histogram for each column
        # g.map(sns.histplot, 'value', binwidth=1, kde=True,)
        g.map(sns.kdeplot, "value", fill=True, bw_adjust=0.4)
        # groupby to get mean and median
        data_g = data.groupby("gravity type").agg(["mean", "median"])
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

        # seperate plot of misfit components
        df = df_anomalies[
            [
                "grav_corrected",
                "forward_total",
                "misfit",
            ]
        ]
        plt.figure()
        sns.histplot(
            data=df,
            palette="viridis",
            kde=True,
            # stat='count',
            multiple="stack",
            element="step",
        )

    if plot_power_spectrums is True:
        utils.raps(
            df_anomalies,
            ["misfit", "reg", "res"],
            region=region,
            spacing=grav_spacing,
        )

    return grid_dict

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
        topo_grids.append(prisms_ds[t].sel(
            easting=slice(region[0], region[1]),
            northing=slice(region[2], region[3])
        ))

    corrections_grids = []
    for m in corrections:
        corrections_grids.append(prisms_ds[m].sel(
            easting=slice(region[0], region[1]),
            northing=slice(region[2], region[3])
        ))
     
    return (misfit_grids, topo_grids, corrections_grids)


def plot_inversion_iteration_results(
    grids,
    grav_results,
    topo_results,
    parameters,
    grav_region,
    iterations,
    shp_mask = None,
    constraints = None,
    constraint_size = 4,
):
    misfit_grids, topo_grids, corrections_grids = grids

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
    misfit_lims=[]
    topo_lims=[]
    corrections_lims=[]

    for g in misfit_grids:
        grid = g
        misfit_lims.append(utils.get_min_max(grid, shp_mask))
    for g in topo_grids:
        topo_lims.append(utils.get_min_max(g, shp_mask))
    for g in corrections_grids:
        grid = g
        corrections_lims.append(utils.get_min_max(grid, shp_mask))

    misfit_min = min([i[0] for i in misfit_lims])
    misfit_max = max([i[1] for i in misfit_lims])
    misfit_lim = vd.maxabs(misfit_min, misfit_max)

    topo_min = min([i[0] for i in topo_lims])
    topo_max = max([i[1] for i in topo_lims])

    corrections_min = min([i[0] for i in corrections_lims])
    corrections_max = max([i[1] for i in corrections_lims])
    corrections_lim = vd.maxabs(corrections_min, corrections_max)

    for column, j in enumerate(grids):
        for row, y in enumerate(j):
            # if only 1 iteration
            if max(iterations)==1:
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
                # lims = (-vd.maxabs(j[0]) * perc, vd.maxabs(j[0]) * perc)
                # lims = utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                # maxabs = vd.maxabs(
                    # utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                # )
                # lims = (-maxabs, maxabs)
                robust = False
            elif column == 1:  # topography grids
                cmap = "gist_earth"
                lims = (topo_min, topo_max)
                # robust=True
                # lims = (-vd.maxabs(j[0]) * perc, vd.maxabs(j[0]) * perc)
                # lims = utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                robust = False
            elif column == 2:  # correction grids
                cmap = "RdBu_r"
                lims = (-corrections_lim, corrections_lim)
                # lims = utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                # maxabs = vd.maxabs(
                #     utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                # )
                # lims = (-maxabs, maxabs)
                robust = False
            # plot grids
            j[row].plot(
                ax=axes,
                cmap=cmap,
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
                rmse = inv.RMSE(
                    grav_results[f"iter_{iterations[row]}_initial_misfit"]
                )
                axes.set_title(
                    f"initial misfit RMSE = {round(rmse, 2)} mGal"
                )
            elif column == 1:  # topography grids
                axes.set_title("updated bathymetry")
            elif column == 2:  # correction grids
                rmse = inv.RMSE(topo_results[f"iter_{iterations[row]}_correction"])
                axes.set_title(
                    f"iteration correction RMSE = {round(rmse, 2)} m"
                )

            if constraints is not None:
                if column == 0:  # misfit grids
                    axes.plot(
                        constraints.easting,
                        constraints.northing,
                        "k+",
                        markersize=constraint_size,
                        markeredgewidth=1,
                    )
                elif column == 1:  # topography grids
                    axes.plot(
                        constraints.easting,
                        constraints.northing,
                        "r+",
                        markersize=constraint_size,
                        markeredgewidth=1,
                    )
                elif column == 2:  # correction grids
                    axes.plot(
                        constraints.easting,
                        constraints.northing,
                        "k+",
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
    text1, text2, text3 = [],[],[]
    for i, (k, v) in enumerate(parameters.items(), start=1):
        if i <= 4:
            text1.append(f"{k}: {v}\n" )
        elif i <= 9:
            text2.append(f"{k}: {v}\n" )
        else:
            text3.append(f"{k}: {v}\n" )

    text1 = "".join(text1)
    text2 = "".join(text2)
    text3 = "".join(text3)

    # if only 1 iteration
    if max(iterations)==1:
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
    shp_mask = None,
    constraints = None,
):

    initial_topo = prisms_ds.surface
    
    # list of variables ending in "_layer"
    topos = [s for s in list(prisms_ds.keys()) if "_layer" in s]
    # list of iterations, e.g. [1,2,3,4]
    its = [int(s[5:][:-6]) for s in topos]
    
    final_topo = prisms_ds[f"iter_{max(its)}_layer"]

          
    if constraints is not None:
        constraints = constraints.rename(columns = {"easting":"x","northing":"y"})
        
    _ = utils.grd_compare(
        initial_topo,
        final_topo,
        plot=True,
        grid1_name="initial topography",
        grid2_name="final topography",
        plot_type="xarray",
        cmap="gist_earth",
        robust=False,
        hist=True,
        inset=False,
        verbose="q",
        title="difference",
        shp_mask=shp_mask,
    )        

    
def plot_inversion_grav_results(
    grav_results,
    region,
    spacing, 
    registration,
    iterations,
    constraints = None,
    shp_mask = None,
):

    initial_misfit = pygmt.xyz2grd(
        data=grav_results[["easting", "northing", "iter_1_initial_misfit"]],
        region=region,
        spacing=spacing,
        registration=registration,
        verbose="q",
    )

    final_misfit = pygmt.xyz2grd(
        data=grav_results[["easting", "northing", f"iter_{max(iterations)}_final_misfit"]],
        region=region,
        spacing=spacing,
        registration=registration,
        verbose="q",
    )
    
    if constraints is not None:
        constraints = constraints.rename(columns = {"easting":"x","northing":"y"})
        
    # plot initial misfit - final misfit
    initial_RMSE = inv.RMSE(grav_results["iter_1_initial_misfit"])
    final_RMSE = inv.RMSE(grav_results[f"iter_{max(iterations)}_final_misfit"])
    
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
    registration = 'g',
    iters_to_plot = None,
    constraints = None,
    plot_iter_results = True,
    plot_topo_results = True,
    plot_grav_results = True,
    **kwargs
):

    # if results are given as filenames (strings), load into dataframes
    if isinstance(grav_results, str):
        grav_results = pd.read_csv(grav_results)
    if isinstance(topo_results, str):
        topo_results = pd.read_csv(topo_results)

    prisms_ds = topo_results.set_index(['northing','easting']).to_xarray()

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
    else: iterations = its

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
    misfit_grids, topo_grids, corrections_grids = grids
    
    if plot_iter_results is True:
        plot_inversion_iteration_results(
            grids,
            grav_results,
            topo_results,
            parameters,
            grav_region,
            iterations,
            shp_mask = kwargs.get("shp_mask", None),
            constraints = constraints,
            constraint_size = kwargs.get("constraint_size",4),
        )
        
    if plot_topo_results is True:
        plot_inversion_topo_results(
            prisms_ds,
            constraints = constraints,
            shp_mask = kwargs.get("shp_mask", None),
        )

    if plot_grav_results is True:
        plot_inversion_grav_results(
            grav_results,
            grav_region,
            grav_spacing, 
            registration,
            iterations,
            constraints = constraints,
            shp_mask = kwargs.get("shp_mask", None),
        )

    return grids

    """
    Plot the results of the inversion.

    Parameters
    ----------
    grav_results : pd.DataFrame or str
        Input gravity data with inversion results columns. Alternatively provide string
        of csv filename.
    topo_results : pd.DataFrame or str
        Dataframe with corrections and updated geometry of the inversion layer for each
        iteration. Alternatively provide string of csv filename.
    layers_dict : dict
        Nested dict; where each layer is a dict with keys:
            'spacing': int, float; grid spacing
            'fname': str; grid file name
            'rho': int, float; constant density value for layer
            'grid': xarray.DataArray;
            'df': pandas.DataFrame; 2d representation of grid
    active_layer : str
        Layer which was inverted for.
    grav_spacing : int
        Spacing of the gravity data for create plots.
    region : list
        Region to plot
    save_topo_nc : bool
        Choose to save the final inverted topography as a netcdf with an optional kwarg
        filename.
    save_residual_nc : bool
        Choose to save the final residual as a netcdf with an optional kwarg
        filename.
    plot_iters: bool
        plot individual iteration results
    plot_topo_results: bool
        plot final topography results
    plot_grav_results: bool
        plot final gravity results
    ----------------
    shp_mask: str
        shapefile to use as a mask to set the colormap limits
    constraints: pd.DataFrame,
        Locations of constraint points, by default is None.
    topo_fname: str
        Customize the name of the saved netcdf file, by default is 'inverted_topo'.
    residual_fname: str
        Customize the name of the saved netcdf file, by default is 'final_residual'.
    """

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
            f"Method: {regional_method} (Source depth={int(best.params_eq_sources/1e3)}km)"
        )

    score = best.value

    anom_grids = anomalies_plotting(
        df_anomalies,
        region = kwargs.get("inversion_region"),
        grav_spacing = kwargs.get("grav_spacing"),
        title = title + f" Optimization score: {round(score,2)} mGal",
        constraints = kwargs.get("constraints"),
        input_forward_column = kwargs.get('input_forward_column', 'forward') ,
        input_grav_column = kwargs.get('input_grav_column', 'grav'),
    )

    _ = utils.grd_compare(
        kwargs.get("true_regional"),
        anom_grids["reg"],
        plot = True,
        region = kwargs.get("inversion_region"),
        plot_type = "xarray",
        cmap = "RdBu_r",
        title = title,
        grid1_name = "True regional misfit",
        grid2_name = "best regional misfit",
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
    **kwargs,
):
    if best_params is None:
        best_params = opti.get_best_params_from_study(study)
    
    with inv_utils.HiddenPrints():
        prism_results, grav_results, params, elapsed_time = inv.geo_inversion(
            input_grav = kwargs.get('input_grav'),
            input_grav_column = kwargs.get('input_grav_column'),
            prism_layer = kwargs.get('prism_layer'),
            max_iterations = kwargs.get('max_iterations'),
            l2_norm_tolerance = kwargs.get('l2_norm_tolerance'),
            delta_l2_norm_tolerance = kwargs.get('delta_l2_norm_tolerance'),
            max_layer_change_per_iter = best_params.get('max_layer_change_per_iter'),
            deriv_type = best_params.get('deriv_type'),
            solver_type = best_params.get('solver_type'),
            solver_damping = [v for k, v in best_params.items() if 'damping' in k.lower()][0],
        )

    # grid resulting prisms
    ds = prism_results.rename(columns={'easting':'x', 'northing':'y'}).set_index(['y', 'x']).to_xarray()

    # subset the inversion region
    ds_inner = ds.sel(
        x=slice(inversion_region[0], inversion_region[1]),
        y=slice(inversion_region[2], inversion_region[3])
    )

    # get last iteration's layer result
    cols = [s for s in prism_results.columns.to_list() if "_layer" in s]

    final_surface = ds_inner[cols[-1]]
        
    diff_grids = utils.grd_compare(
            true_surface,
            final_surface,
            grid1_name='True',
            grid2_name='Final',
            region=inversion_region,
            plot=True,
            plot_type="xarray",
            # cmap='batlowW',
            cmap="gist_earth",
            robust=True,
            points=constraint_points.rename(columns = {"easting":"x","northing":"y"}),
            hist=True,
            inset=False,
        )

    _ = plot_inversion_results(
        grav_results,
        prism_results,
        params,
        grav_region=inversion_region,
        plot_iter_results = plot_iter_results,
        plot_topo_results = plot_topo_results,
        plot_grav_results = plot_grav_results,
        iters_to_plot=4,
        grav_spacing=grav_spacing,
        constraints=constraint_points,
        )
    
    if constraint_points is not None:
        constraints = profile.sample_grids(
            df=constraint_points,
            grid=diff_grids[0],
            name="surface_difference",
            coord_names = ("easting", "northing"),
        )
        print(f"RMSE between surfaces at constraints: {round(inv.RMSE(constraints.surface_difference), 2)} m")


def combined_history(study, target_names, include_duration=False,):
    """
    plot combined optimization history for multiobjective optimizations.
    """
    target_names = target_names.copy()
    figs = []
    for i,j in enumerate(target_names):
        f = optuna.visualization.plot_optimization_history(study,
            target=lambda t: t.values[i],
            target_name=j)
        figs.append(f)
    
    if include_duration is True and "duration" not in target_names:
        f = optuna.visualization.plot_optimization_history(study,
            target=lambda t: t.duration.total_seconds(),
            target_name="duration")
        figs.append(f)
        target_names.append("duration")
    
    if len(target_names)<2:
        layout = plotly.graph_objects.Layout(
            title="Optimization History Plot",
            yaxis=plotly.graph_objs.layout.YAxis(
                title=target_names[0],
            ),
            xaxis=dict(title="Trial"),
        )
    elif len(target_names)>=2:
        yaxes={}
        for i,j in enumerate(target_names, start=1):
            if i == 1:
                pass
            else:
                yax = plotly.graph_objs.layout.YAxis(
                        title=j,
                        overlaying= 'y',
                        side='left',
                        anchor='free',
                        autoshift=True,)
                yaxes[f"yaxis{i}"] = yax 
        layout = plotly.graph_objects.Layout(
            title="Optimization History Plot",
            yaxis1=plotly.graph_objs.layout.YAxis(
                title=target_names[0],
                side='right',
            ),
            xaxis=dict(title="Trial"),
            **yaxes,
        )
        
    # Create figure with secondary x-axis
    fig = plotly.graph_objects.Figure(layout=layout)
    
    # Add traces
    for i,j in enumerate(target_names):
        fig.add_trace(
            plotly.graph_objects.Scatter(x=figs[i].data[0]['x'], y=figs[i].data[0]['y'],
            name=j,
            mode='markers',
            yaxis=f'y{i+1}',))

    fig.update_layout(xaxis_title='Trial', title="Optimization History Plot")

    return fig


def combined_importance(study, target_names, params=None, include_duration=False,):
    """
    plot combined objective value and duration importances. Note, target names must be supplied in order which objective function returns them. You can't reorder, or skip a value unless it's the last value. 
    """
    target_names = target_names.copy()
    figs = []
    for i,j in enumerate(target_names):
        f = optuna.visualization.plot_param_importances(study,
            target=lambda t: t.values[i],
            target_name=j,
            params=params,                                           
        )
        figs.append(f)
    
    if include_duration is True and "duration" not in target_names:
        f = optuna.visualization.plot_param_importances(study,
            target=lambda t: t.duration.total_seconds(),
            target_name="duration",
            params=params,
        )
        figs.append(f)
        target_names.append("duration")
    
    # Create figure
    fig = plotly.graph_objects.Figure()
    
    # Add traces
    for i,j in enumerate(target_names):
        fig.add_trace(
            plotly.graph_objects.Bar(x=figs[i].data[0]['x'], y=figs[i].data[0]['y'],
            name=j,
            orientation='h',
        ))

    fig.update_layout(xaxis_title='Importance', title="Hyperparameter Importance")

    return fig



def combined_edf(study, target_names, include_duration=False,):
    """
    plot combined objective value and duration empirical distribution functions
    """
    target_names = target_names.copy()
    figs = []
    for i,j in enumerate(target_names):
        f = optuna.visualization.plot_edf(study,
            target=lambda t: t.values[i],
            target_name=j,
        )
        figs.append(f)
    
    if include_duration is True and "duration" not in target_names:
        f = optuna.visualization.plot_edf(study,
            target=lambda t: t.duration.total_seconds(),
            target_name="duration")
        figs.append(f)
        target_names.append("duration")
    
    if len(target_names)<2:
        layout = plotly.graph_objects.Layout(
            title="Emperical Distribution Function",
            xaxis=plotly.graph_objs.layout.XAxis(
                title=target_names[0],
            ),
            yaxis=dict(
                title="Cumulative Probability"
            ),
        )
        
    elif len(target_names)>=2:
        xaxes={}
        for i,j in enumerate(target_names, start=1):
            if i == 1:
                pass
            else:
                xax = plotly.graph_objs.layout.XAxis(
                        title=j,
                        overlaying= 'x',
                        side='bottom',
                        anchor='free',
                        # autoshift=True,
                )
                xaxes[f"xaxis{i}"] = xax 
        layout = plotly.graph_objects.Layout(
            title="Emperical Distribution Function",
            xaxis1=plotly.graph_objs.layout.XAxis(
                title=target_names[0],
                side='top',
            ),
            yaxis=dict(title="Cumulative Probability"),
            **xaxes,
            )
        
    # Create figure with secondary x-axis
    fig = plotly.graph_objects.Figure(layout=layout)
    
    # Add traces
    for i,j in enumerate(target_names):
        fig.add_trace(
            plotly.graph_objects.Scatter(x=figs[i].data[0]['x'], y=figs[i].data[0]['y'],
            name=j,
            mode='lines',
            xaxis=f'x{i+1}',))

    return fig


def plot_optuna_inversion_figures(study, target_names, include_duration=False,):
    
    combined_history(study, target_names, include_duration=include_duration,).show()
    
    for i, j in enumerate(target_names):
        optuna.visualization.plot_slice(study,
            target=lambda t: t.values[i],
            target_name=j).show()
    
    if include_duration is True and "duration" not in target_names:
        optuna.visualization.plot_slice(study,
            target=lambda t: t.duration.total_seconds(),
            target_name="Execution time").show()
        
    combined_importance(study,
        target_names,
        params=[
            "deriv_type",
            "verde_damping",
            ],
        include_duration=include_duration).show()

    combined_importance(study,
        target_names,
        params=[
            "deriv_type",
            "scipy_damping"
            ],
       include_duration=include_duration).show()
    
    combined_edf(study, 
        target_names,
        include_duration=include_duration).show()
    
    if len(target_names) == 1:
        if "duration" not in target_names:
            if include_duration is True:
                optuna.visualization.plot_pareto_front(study,
                    targets=lambda t: (t.values[0], t.duration.total_seconds()),
                    target_names=target_names+["duration"]).show()
           
    elif len(target_names) > 1:
        if "duration" not in target_names:
            if include_duration is True:
                optuna.visualization.plot_pareto_front(study,
                    targets=lambda t: (t.values, t.duration.total_seconds()),
                    target_names=target_names+["duration"]).show()
            elif include_duration is False:
                optuna.visualization.plot_pareto_front(study,
                    target_names=target_names).show()
        elif "duration" in target_names:
             optuna.visualization.plot_pareto_front(study,
                target_names=target_names).show()