import warnings
from typing import Union

import harmonica as hm
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygmt
import pyvista as pv
import seaborn as sns
import verde as vd
import xarray as xr
from antarctic_plots import maps, utils, fetch, profile

def plot_inputs(
    inputs : list,
    grav_spacing : list,
    active_layer : str,
    region : list = None,
    inversion_region : list = None,
    plot_type : str = "xarray",
    **kwargs,
):
    """
    Plot the input data for the inversion

    Parameters
    ----------
    inputs : list
         output from `inversion.import_layers`
    grav_spacing : list
        spacing to use to plot gravity data
    active_layer : str
        plot constraints on this layer if set
    region : list, optional
        region to plot, by default is buffer region
    inversion_region : list, optional
        plot a bounding box of the inversion region, by default None
    plot_type : str, optional
        choose method of plotting; either "xarray" or "pygmt", by default "xarray"
    """

    (   layers,
        grav,
        constraint_grid,
        constraint_points,
        constraint_points_RIS,
    ) = inputs

    # if region not set, get from first grid in layers (buffer region)
    if region is None:
        region = utils.get_grid_info(list(layers.values())[0]["grid"])[1]
        # region = vd.get_region((grav.x, grav.y))

    # grid gravity data
    grav_grid = pygmt.xyz2grd(
        data=grav[['x', 'y', 'Gobs']],
        region=region,
        spacing=grav_spacing,
        registration="p"
    )

    plotting_constraints = kwargs.get("plotting_constraints", constraint_points_RIS)

    if kwargs.get("plot_type", "xarray") == "pygmt":
        fig = maps.plot_grd(
            grid=grav_grid,
            fig_height=8,
            cmap="vik+h0",
            region=region,
            coast=True,
            grd2cpt=True,
            title="Observed gravity",
            cbar_unit="mGal",
            show_region=inversion_region,
            hist=True,
            cbar_yoffset=3,
        )
        if constraint_grid is not None:
            fig = maps.plot_grd(
                grid=constraint_grid,
                fig_height=8,
                cmap="gray",
                region=region,
                coast=True,
                grd2cpt=True,
                cpt_lims=[0,1],
                title="Constraint grid",
                show_region=inversion_region,
                hist=True,
                cbar_yoffset=3,
                fig=fig,
                origin_shift='xshift',
            )

        for i, (k, v) in enumerate(layers.items()):

            fig = maps.plot_grd(
                grid=layers[k]["grid"],
                fig_height=8,
                cmap="rain",
                reverse_cpt=True,
                region=region,
                coast=True,
                grd2cpt=True,
                title=k,
                cbar_label="elevation",
                cbar_unit="m",
                show_region=inversion_region,
                hist=True,
                cbar_yoffset=3,
                fig=fig,
                origin_shift="xshift",
            )
            if k == active_layer:
                if constraint_points is not None:
                    fig.plot(
                        x=plotting_constraints.x,
                        y=plotting_constraints.y,
                        style="c.1c",
                        color="black",
                    )
            else:
                points=None

        fig.show()

    elif kwargs.get("plot_type", "xarray") == "xarray":
        if constraint_grid is not None:
            extra = 2
        else:
            extra = 1

        sub_width = 5
        nrows, ncols = 1, len(layers)+extra

        # setup subplot figure
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width*ncols, sub_width*nrows),
            )

        p = 0
        # clip grav grid to region
        grav_grid = grav_grid.sel(
                x=slice(region[0], region[1]),
                y=slice(region[2], region[3])
                )
        grav_grid.plot(
            ax=ax[p],
            robust=True,
            cmap="RdBu_r",
            cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                    },
        )
        ax[p].set_title("Observed gravity")
        p += 1
        if constraint_grid is not None:
            # clip constraints grid
            constr = constraint_grid.sel(
                x=slice(region[0], region[1]),
                y=slice(region[2], region[3])
                )

            constr.plot(
                ax=ax[p],
                robust=True,
                cmap="copper",
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                    },
            )
            ax[p].set_title("Constraint grid")
        for i, j in enumerate(layers):
            # clip layers grids
            grid = layers[j]["grid"].sel(
                x=slice(region[0], region[1]),
                y=slice(region[2], region[3])
                )
            percent = 1
            lims = (
                np.nanquantile(grid, q=percent / 100),
                np.nanquantile(grid, q=1 - (percent / 100)),
            )
            grid.plot(
                ax=ax[i + extra],
                vmin=lims[0],
                vmax=lims[1],
                cmap="gist_earth",
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                    },
            )
            ax[i + extra].set_title(f"{j} elevation")
            if constraint_points is not None:
                if j == active_layer:
                    ax[i + extra].plot(
                        plotting_constraints.x,
                        plotting_constraints.y,
                        "rx",
                        markersize=1.5,
                        markeredgewidth=0.75,
                    )
        for a in ax:
            if inversion_region is not None:
                a.add_patch(
                    mpl.patches.Rectangle(
                        xy=(inversion_region[0], inversion_region[2]),
                        width=(inversion_region[1] - inversion_region[0]),
                        height=(inversion_region[3] - inversion_region[2]),
                        linewidth=1,
                        fill=False,
                    ))
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xlabel("")
            a.set_ylabel("")
            a.set_aspect("equal")

    if kwargs.get("power_spectrum", False) is True:
        names = []
        grids = []
        for k, v in layers.items():
            names.append(k)
            grids.append(v["grid"])
        names.append("observed_gravity")
        grids.append(grav_grid)
        utils.raps(grids, names, plot_type="pygmt")



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
    cmap : str ="viridis",
    color_by : str ="density",
    region : list = None,
    clip_box : bool = True,
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

    pv.set_jupyter_backend(kwargs.get("backend", "ipyvtklink"))


    # Plot with pyvista
    plotter = pv.Plotter(
        lighting="three_lights",
        # window_size=(1000, 1000),
        notebook=True,
    )

    for i, j in enumerate(prisms):
        # if region is given, clip model
        if region is not None:
            j = j.sel(
                easting=slice(region[0], region[1]),
                northing=slice(region[2], region[3])
                )

        # turn prisms into pyvist object
        pv_grid = j.prism_layer.to_pyvista()

        # clip corner out of model to help vizualize
        if clip_box is True:
            # extract region from first prism layer
            reg = vd.get_region((j.easting.values, j.northing.values))
            # box_buffer used make box slightly bigger
            box_buffer = kwargs.get('box_buffer', 5e3)
            # set 6 edges of cube to clip out
            bounds = [
                reg[0] - box_buffer,
                reg[0] + box_buffer + ((reg[1] - reg[0]) / 2),
                reg[2] - box_buffer,
                reg[2] + box_buffer + ((reg[3] - reg[2]) / 2),
                np.nanmin(j.bottom),
                np.nanmax(j.top)
            ]
            pv_grid = pv_grid.clip_box(
                bounds,
                invert=True,
            )

        if color_by == 'constant':
            colors = kwargs.get("colors",
                ["lavender", "aqua", "goldenrod", "saddlebrown", "black"])
            plotter.add_mesh(
                pv_grid,
                color=colors[i],
                smooth_shading=kwargs.get("smooth_shading", False),
                style=kwargs.get("style", "surface"),
                show_edges=kwargs.get("show_edges", False),
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
    plotter.show()


def plot_prism_layers(
    layers: dict,
    region: list = None,
    cmap: str = 'viridis',
    plot_type: str = '2D',
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
        prism_list = [prism.get('prisms') for prism in layers_to_plot.values()]

        # removed duplicate kwargs before passing to show_prism_layers()
        subset_kwargs = {kw: kwargs[kw] for kw in kwargs if kw
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
            color_by=kwargs.get("color_by", 'constant'),
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
            figsize=(sub_width*ncols, sub_width*nrows),
            )

        for i, (k,v) in enumerate(layers.items()):
            # get thickness grid from prisms
            thick = v['prisms'].thickness

            # clip grids to region if specified
            if region is not None:
                thick = thick.sel(
                    easting=slice(region[0], region[1]),
                    northing=slice(region[2], region[3])
                    )

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

        for a in ax:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_xlabel("")
            a.set_ylabel("")
            a.set_aspect("equal")


def forward_grav_plotting(
    df_forward: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
    plot_dists: bool = False,
    plot_power_spectrums: bool = False,
    exclude_layers: list = None,
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
        region = vd.get_region((df_forward.x, df_forward.y))

    # if gravity spacing not supplied, extract from dataframe
    if grav_spacing is None:
        grid = df_forward.set_index(["y", "x"]).to_xarray().Gobs
        grav_spacing = float(utils.get_grid_info(grid)[0])

    # drop columns previously used in Bouguer correction
    if exclude_layers is not None:
        cols2drop = ~df_forward.columns.str.contains('|'.join(exclude_layers))
        df = df_forward[df_forward.columns[cols2drop]]
    else:
        df = df_forward

    # get list of columns to grid
    cols_to_grid = [x for x in df.columns.to_list() if "forward" in x]

    sub_width = 5

    # nrows, ncols = utils.square_subplots(len(cols_to_grid))
    nrows, ncols = 1, len(cols_to_grid)

    # setup subplot figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width*ncols, sub_width*nrows),
        )

    # empty list to append grids to
    forward_grids = []

    for i, (col, ax) in enumerate(zip(cols_to_grid, axs.T.ravel())):
        grid = pygmt.xyz2grd(
            data=df_forward[["x", "y", col]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose="q",
        )

        forward_grids.append(grid)

        # plot each grid
        grid.plot(
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

        # set column names as titles
        ax.set_title(col)

        # set axes labels and make proportional
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_aspect("equal")

    # add grids and names to dictionary
    grid_dict = dict(zip(cols_to_grid, forward_grids))

    if plot_dists is True:
        # get columns to include in plots
        dists = df.drop(["x", "y", "z", "Gobs"], axis=1)
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
        power = df.drop(["x", "y", "z", "Gobs"], axis=1)
        # plot radially average power spectrum for each layer
        utils.raps(
            df,
            list(power.columns),
            region=region,
            spacing=grav_spacing,
        )

    return grid_dict

def anomalies_plotting(
    df_anomalies: pd.DataFrame,
    region: list = None,
    grav_spacing: float = None,
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

    # if inversion region not supplied, extract from dataframe
    if region is None:
        region = vd.get_region((df_anomalies.x, df_anomalies.y))

    # if gravity spacing not supplied, extract from dataframe
    if grav_spacing is None:
        grid = df_anomalies.set_index(["y", "x"]).to_xarray().Gobs
        grav_spacing = float(utils.get_grid_info(grid)[0])

    constraints = kwargs.get("constraints", None)

    # get columns to include in gridding
    cols_to_grid = [
        # "Gobs",
        "grav_corrected",
        "forward_total",
        "misfit",
        "reg",
        "res",
        ]

    # get RMS value for misfit
    RMS = round(np.sqrt((df_anomalies.res**2).mean(skipna=True)), 2)

    # set titles for grids
    plot_titles = [
        "observed gravity",
        # "partial bouguer gravity",
        "forward gravity",
        "gravity misfit",
        "regional misfit",
        f"residual misfit: {RMS}mGal",
    ]

    sub_width = 5
    # nrows, ncols = utils.square_subplots(len(cols_to_grid))
    nrows, ncols = 2, 3

    # setup subplot figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(sub_width*ncols, sub_width*nrows),
        )

    # empty list of grids to append to
    anom_grids = []

    for i, (col, ax) in enumerate(zip(cols_to_grid, axs.ravel())):
        grid = pygmt.xyz2grd(
            data=df_anomalies[["x", "y", col]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose='q',
        )
        anom_grids.append(grid)

        # plot each grid
        grid.plot(
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
            if i in [2, 3, 4, 5]:
                ax.plot(
                    constraints.x,
                    constraints.y,
                    "kx",
                    markersize=2,
                    markeredgewidth=1,
                )

    # add figure title
    fig.suptitle(kwargs.get("title", " "), fontsize=24)

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

    return anom_grids


def plot_inversion_results(
    grav_results: Union[pd.DataFrame, str],
    topo_results: Union[pd.DataFrame, str],
    layers: dict,
    active_layer: str,
    grav_spacing: int,
    region: list = None,
    save_topo_nc: bool = False,
    save_residual_nc: bool = False,
    plot_iters: bool = True,
    plot_topo_results: bool = True,
    plot_grav_results: bool = True,
    **kwargs,
):
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
    layers : dict
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
        Choose to save the initial residual as a netcdf with an optional kwarg
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
    max_layer_change_per_iter : float
        Use the value set in inversion.geo_inversion(), by default is max absolute value
        of change in first iteration.
    constraints: pd.DataFrame,
        Locations of constraint points, by default is None.
    topo_fname: str
        Customize the name of the saved netcdf file, by default is 'inverted_topo'.
    residual_fname: str
        Customize the name of the saved netcdf file, by default is 'initial_residual'.
    """
    # if results are given as filenames (strings), load into dataframes
    if isinstance(grav_results, str):
        grav_results = pd.read_csv(grav_results)
    if isinstance(topo_results, str):
        topo_results = pd.read_csv(topo_results)

    # either set input inversion region or get from input gravity data extent
    if region is None:
        region = vd.get_region((grav_results.x, grav_results.y))

    # if not supplied, set max correction equal to max absolute value of iter 1
    # correction
    max_abs = vd.maxabs(topo_results.iter_1_correction)
    max_layer_change_per_iter = kwargs.get("max_layer_change_per_iter", max_abs)

    # perc = kwargs.get("perc", 0.7)
    constraints = kwargs.get("constraints", None)

    # get lists of columns to grid
    misfits = [s for s in grav_results.columns.to_list() if "initial_misfit" in s]
    topos = [s for s in topo_results.columns.to_list() if "final_top" in s]
    corrections = [s for s in topo_results.columns.to_list() if "_correction" in s]

    # list of iterations, e.g. [1,2,3,4]
    iterations = [int(s[5:][:-15]) for s in misfits]

    # function to give RMS of series
    def RMS(df):
            return round(np.sqrt(np.nanmean(df**2).item()),2)

    if plot_iters is True:
        misfit_grids = []
        topo_grids = []
        corrections_grids = []

        for i in misfits:
            grid = pygmt.xyz2grd(
                data=grav_results[["x", "y", i]],
                region=region,
                spacing=grav_spacing,
                registration="p",
                verbose="q",
            )
            misfit_grids.append(grid)

        for i in topos:
            grid = pygmt.xyz2grd(
                data=topo_results[["x", "y", i]],
                region=region,
                spacing=layers[active_layer]["spacing"],
                registration="p",
                verbose="q",
            )
            topo_grids.append(grid)

        for i in corrections:
            grid = pygmt.xyz2grd(
                data=topo_results[["x", "y", i]],
                region=region,
                spacing=layers[active_layer]["spacing"],
                registration="p",
                verbose="q",
            )
            corrections_grids.append(grid)

        # set figure parameters
        sub_width = 5
        nrows, ncols = max(iterations), 3

        # setup subplot figure
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width*ncols, sub_width*nrows),
            )

        grids = (misfit_grids, topo_grids, corrections_grids)

        for column, j in enumerate(grids):
            for row, y in enumerate(j):
                # add iteration number as text
                if column == 0:
                    plt.text(
                        -0.1,
                        0.5,
                        f"Iteration #{row+1}",
                        transform=ax[row,column].transAxes,
                        rotation="vertical",
                        ha="center",
                        va="center",
                        fontsize=20,
                    )

                # set colormaps and limits
                if column == 0: # misfit grids
                    cmap = 'RdBu_r'
                    # lims = (-vd.maxabs(j[0]) * perc, vd.maxabs(j[0]) * perc)
                    lims = utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                    robust=False
                elif column == 1: # topography grids
                    cmap = 'gist_earth'
                    # lims = (None,None)
                    # robust=True
                    # lims = (-vd.maxabs(j[0]) * perc, vd.maxabs(j[0]) * perc)
                    lims = utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                    robust=False
                elif column == 2: # correction grids
                    cmap = 'RdBu_r'
                    # lims = (-max_layer_change_per_iter, max_layer_change_per_iter)
                    lims = utils.get_min_max(j[0], kwargs.get("shp_mask", None))
                    robust=False

                # plot grids
                j[row].plot(
                    ax=ax[row,column],
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
                if column == 0: # misfit grids
                    ax[row,column].set_title(
                        "initial misfit RMS = "
                        f"{RMS(grav_results[f'iter_{row+1}_initial_misfit'])}mGal")
                elif column == 1: # topography grids
                    ax[row,column].set_title("updated bathymetry")
                elif column == 2: # correction grids
                    ax[row,column].set_title(
                        "iteration correction RMS = "
                        f"{RMS(topo_results[f'iter_{row+1}_correction'])}m")

                if constraints is not None:
                    if column == 0: # misfit grids
                        ax[row,column].plot(constraints.x, constraints.y, "k+")
                    elif column == 1: # topography grids
                        ax[row,column].plot(constraints.x, constraints.y, "r+")
                    elif column == 2: # correction grids
                        ax[row,column].plot(constraints.x, constraints.y, "k+")


                # set axes labels and make proportional
                ax[row,column].set_xticklabels([])
                ax[row,column].set_yticklabels([])
                ax[row,column].set_xlabel("")
                ax[row,column].set_ylabel("")
                ax[row,column].set_aspect("equal")

    if plot_topo_results is True:
        initial_topo = layers[active_layer]["grid"]

        final_topo = pygmt.xyz2grd(
                data=topo_results[["x", "y", f"iter_{max(iterations)}_final_top"]],
                region=region,
                spacing=layers[active_layer]["spacing"],
                registration="p",
                verbose="q",
            )

        utils.grd_compare(
            initial_topo,
            final_topo,
            plot=True,
            plot_type="xarray",
            cmap="gist_earth",
            robust=False,
            verbose="q",
            grid1_name="initial topography",
            grid2_name="final topography",
            title="difference",
            shp_mask=kwargs.get("shp_mask", None),
            )

    if plot_grav_results is True:
        Gobs = pygmt.xyz2grd(
            data=grav_results[["x", "y", "Gobs"]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose="q",
        )

        final_forward_total = pygmt.xyz2grd(
            data=grav_results[["x", "y", f"iter_{max(iterations)}_forward_total"]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose="q",
        )

        initial_forward_total = pygmt.xyz2grd(
            data=grav_results[["x", "y", "forward_total"]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose="q",
        )

        initial_misfit = pygmt.xyz2grd(
            data=grav_results[["x", "y", "iter_1_initial_misfit"]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose="q",
        )

        final_misfit = pygmt.xyz2grd(
            data=grav_results[["x", "y", f"iter_{max(iterations)}_final_misfit"]],
            region=region,
            spacing=grav_spacing,
            registration="p",
            verbose="q",
        )

        # plot observerd - forward
        utils.grd_compare(
            Gobs,
            final_forward_total,
            plot=True,
            plot_type="xarray",
            cmap="RdBu_r",
            robust=False,
            verbose="q",
            grid1_name="Observed gravity",
            grid2_name="Final total forward gravity",
            title="difference",
            shp_mask=kwargs.get("shp_mask", None),
            )

        # plot initial forward - final forward
        utils.grd_compare(
            initial_forward_total,
            final_forward_total,
            plot=True,
            plot_type="xarray",
            cmap="RdBu_r",
            robust=False,
            verbose="q",
            grid1_name="Initial total forward gravity",
            grid2_name="Final total forward gravity",
            title="difference",
            shp_mask=kwargs.get("shp_mask", None),
            )

        # plot initial misfit - final misfit
        initial_rms = RMS(grav_results[f'iter_1_initial_misfit'])
        final_rms = RMS(grav_results[f'iter_{max(iterations)}_final_misfit'])
        utils.grd_compare(
            initial_misfit,
            final_misfit,
            plot=True,
            plot_type="xarray",
            cmap="RdBu_r",
            robust=False,
            verbose="q",
            grid1_name=f"Initial misfit: RMS={initial_rms}mGal",
            grid2_name=f"Final misfit: RMS={final_rms}mGal",
            title="difference",
            shp_mask=kwargs.get("shp_mask", None),
            )


    if save_topo_nc is True:
        final_topo.to_netcdf(
            f"results/{kwargs.get('topo_fname','inverted_topo')}.nc"
        )

    if save_residual_nc is True:
        initial_residual.to_netcdf(
            f"results/{kwargs.get('residual_fname','initial_residual')}.nc"
        )
