from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
from invert4geom import synthetic as inv_synthetic
from invert4geom import uncertainty
from invert4geom import utils as inv_utils
from polartoolkit import fetch, maps, profiles, utils

import RIS_gravity_inversion.gravity_processing as gravity_processing


def load_synthetic_model(
    spacing: float = 1e3,
    inversion_region: tuple[float, float, float, float] = (
        -40e3,
        260e3,
        -1800e3,
        -1400e3,
    ),
    buffer: float = 0,
    zref: float = 0,
    bathymetry_density_contrast: float = 1476,
    basement_density_contrast: float = 100,
    basement: bool = False,
    gravity_noise: float | None = None,
    gravity_noise_wavelength: float = 50e3,
    plot_topography: bool = True,
    plot_gravity: bool = True,
    just_topography: bool = False,
) -> tuple[xr.DataArray, pd.DataFrame]:
    """
    Function to perform all necessary steps to create a synthetic model for the examples
    in the documentation.

    Parameters
    ----------
    spacing : float, optional
        spacing of the grid and gravity, by default 1e3
    buffer : float, optional
        buffer to add around the region, by default 0. Buffer region used for creating
        topography and prisms, while inner region used for extent of gravity and
        constraints.
    zref : float , optional
        reference level to use, by default 0
    bathymetry_density_contrast : float, optional
        density contrast between bathymetry and water, by default 1476, (2500 - 1024)
    basement_density_contrast : float, optional
        density contrast between basement and water, by default 100
    basement : bool, optional
        set to True to include a basement model for the regional gravity field, by
        default False
    gravity_noise : float | None, optional
        decimal percentage noise level to add to gravity data, by default None
    gravity_noise_wavelength : float, optional
        wavelength of noise in km to add to gravity data, by default 50e3
    plot_topography : bool, optional
        plot the topography, by default False
    plot_gravity : bool, optional
        plot the gravity data, by default True
    just_topography : bool, optional
        return only the topography, by default False

    Returns
    -------
    true_topography : xarray.DataArray
        the true topography
    grav_df : pandas.DataFrame
        the gravity data
    """
    # inversion_region = (-40e3, 260e3, -1800e3, -1400e3)
    registration = "g"

    buffer_region = (
        vd.pad_region(inversion_region, buffer) if buffer != 0 else inversion_region
    )

    # get Ross Sea bathymetry and basement data
    bathymetry_grid = fetch.ibcso(
        layer="bed",
        reference="ellipsoid",
        region=buffer_region,
        spacing=spacing,
        registration=registration,
    ).rename({"x": "easting", "y": "northing"})

    if just_topography is True:
        return bathymetry_grid, None, None

    if basement is True:
        sediment_thickness = fetch.sediment_thickness(
            version="lindeque-2016",
            region=buffer_region,
            spacing=spacing,
            registration=registration,
        ).rename({"x": "easting", "y": "northing"})
        basement_grid = bathymetry_grid - sediment_thickness
        basement_grid = xr.where(
            basement_grid > bathymetry_grid, bathymetry_grid, basement_grid
        )
    else:
        basement_grid = None

    if plot_topography is True:
        fig = maps.plot_grd(
            bathymetry_grid,
            show_region=inversion_region,
            fig_height=10,
            title="Bathymetry",
            hist=True,
            cbar_yoffset=1,
            cmap="rain",
            reverse_cpt=True,
            cbar_label="elevation (m)",
            robust=True,
        )

        if basement is True:
            fig = maps.plot_grd(
                basement_grid,
                show_region=inversion_region,
                fig_height=10,
                title="Basement",
                hist=True,
                cbar_yoffset=1,
                cmap="rain",
                reverse_cpt=True,
                cbar_label="elevation (m)",
                fig=fig,
                origin_shift="xshift",
                robust=True,
                scalebar=True,
                scale_position="n-.05/-.03",
            )

        fig.show()

    # calculate forward gravity effects

    # bathymetry
    density_grid = xr.where(
        bathymetry_grid >= zref,
        bathymetry_density_contrast,
        -bathymetry_density_contrast,
    )
    bathymetry_prisms = inv_utils.grids_to_prisms(
        bathymetry_grid,
        zref,
        density=density_grid,
    )

    # basement
    if basement is True:
        density_grid = xr.where(
            basement_grid >= zref,
            basement_density_contrast,
            -basement_density_contrast,
        )
        basement_prisms = inv_utils.grids_to_prisms(
            basement_grid,
            zref,
            density=density_grid,
        )

    # make pandas dataframe of locations to calculate gravity
    # this represents the station locations of a gravity survey
    # create lists of coordinates
    coords = vd.grid_coordinates(
        region=inversion_region,
        spacing=spacing,
        pixel_register=False,
        extra_coords=1000,  # survey elevation
    )

    # grid the coordinates
    observations = vd.make_xarray_grid(
        (coords[0], coords[1]),
        data=coords[2],
        data_names="upward",
        dims=("northing", "easting"),
    ).upward

    grav_df = vd.grid_to_table(observations)

    grav_df["bathymetry_grav"] = bathymetry_prisms.prism_layer.gravity(
        coordinates=(
            grav_df.easting,
            grav_df.northing,
            grav_df.upward,
        ),
        field="g_z",
        progressbar=True,
    )

    if basement is True:
        grav_df["basement_grav"] = basement_prisms.prism_layer.gravity(
            coordinates=(
                grav_df.easting,
                grav_df.northing,
                grav_df.upward,
            ),
            field="g_z",
            progressbar=True,
        )
        grav_df["basement_grav"] = grav_df.basement_grav - grav_df.basement_grav.mean()
    else:
        grav_df["basement_grav"] = 0

    # add forward gravity fields together to get the observed gravity
    grav_df["disturbance"] = grav_df.bathymetry_grav + grav_df.basement_grav

    # contaminate gravity with short and long-wavelength random noise
    if gravity_noise is not None:
        # long-wavelength noise
        grav_df["noise_free_disturbance"] = grav_df.disturbance
        cont = inv_synthetic.contaminate_with_long_wavelength_noise(
            grav_df.set_index(["northing", "easting"]).to_xarray().disturbance,
            coarsen_factor=None,
            spacing=gravity_noise_wavelength,
            noise_as_percent=False,
            noise=gravity_noise,
        )
        df = vd.grid_to_table(cont.rename("disturbance")).reset_index(drop=True)
        grav_df = pd.merge(  # noqa: PD015
            grav_df.drop(columns=["disturbance"], errors="ignore"),
            df,
            on=["easting", "northing"],
        )

        # short-wavelength noise
        cont = inv_synthetic.contaminate_with_long_wavelength_noise(
            grav_df.set_index(["northing", "easting"]).to_xarray().disturbance,
            coarsen_factor=None,
            spacing=spacing * 2,
            noise_as_percent=False,
            noise=gravity_noise,
        )
        df = vd.grid_to_table(cont.rename("disturbance")).reset_index(drop=True)
        grav_df = pd.merge(  # noqa: PD015
            grav_df.drop(columns=["disturbance"], errors="ignore"),
            df,
            on=["easting", "northing"],
        )

        grav_df["uncert"] = gravity_noise

    grav_df["gravity_anomaly"] = grav_df.disturbance

    if plot_gravity is True:
        grav_grid = grav_df.set_index(["northing", "easting"]).to_xarray()

        fig = maps.plot_grd(
            grav_grid.bathymetry_grav,
            region=inversion_region,
            fig_height=10,
            title="Bathymetry gravity",
            hist=True,
            cbar_yoffset=1,
            cbar_label="mGal",
            robust=True,
        )

        if basement is True:
            fig = maps.plot_grd(
                grav_grid.basement_grav,
                region=inversion_region,
                fig_height=10,
                title="Basement gravity",
                hist=True,
                cbar_yoffset=1,
                cbar_label="mGal",
                fig=fig,
                origin_shift="xshift",
                robust=True,
            )

        fig = maps.plot_grd(
            grav_grid.disturbance,
            region=inversion_region,
            fig_height=10,
            title="Combined gravity",
            hist=True,
            cbar_yoffset=1,
            cbar_label="mGal",
            fig=fig,
            origin_shift="xshift",
            robust=True,
        )

        fig.show()

        if gravity_noise is not None:
            _ = utils.grd_compare(
                grav_grid.noise_free_disturbance,
                grav_grid.disturbance,
                fig_height=10,
                plot=True,
                grid1_name="Gravity",
                grid2_name=f"with {gravity_noise} mGal noise",
                title="Difference",
                title_font="18p,Helvetica-Bold,black",
                cbar_unit="mGal",
                cbar_label="gravity",
                # RMSE_decimals=0,
                region=inversion_region,
                inset=False,
                hist=True,
                cbar_yoffset=1,
                label_font="16p,Helvetica,black",
            )
    return bathymetry_grid, basement_grid, grav_df


def constraint_layout_number(
    num_constraints=None,
    latin_hypercube=False,
    shape=None,
    spacing=None,
    shift_stdev=0,
    region=None,
    shapefile=None,
    padding=None,
    add_outside_points=False,
    grid_spacing=None,
    plot=False,
    seed=0,
):
    full_region = region

    if shapefile is not None:
        bounds = gpd.read_file(shapefile).bounds
        region = [bounds.minx, bounds.maxx, bounds.miny, bounds.maxy]
        region = [x.values[0] for x in region]

    x = region[1] - region[0]
    y = region[3] - region[2]

    if (shape is None) and (num_constraints is None) and (spacing is None):
        msg = "must provide either shape, num_constraints, or spacing"
        raise ValueError(msg)

    if padding is not None:
        region = vd.pad_region(region, padding)

    width = region[1] - region[0]
    height = region[3] - region[2]

    if num_constraints == 0:
        constraints = pd.DataFrame(columns=["easting", "northing", "upward", "inside"])
    elif latin_hypercube:
        if num_constraints is None:
            msg = "need to set number of constraints if using latin hypercube"
            raise ValueError(msg)
        coord_dict = {
            "easting": {
                "distribution": "uniform",
                "loc": region[0],  # lower bound
                "scale": width,  # range
            },
            "northing": {
                "distribution": "uniform",
                "loc": region[2],  # lower bound
                "scale": height,  # range
            },
        }
        sampled_coord_dict = uncertainty.create_lhc(
            n_samples=num_constraints,
            parameter_dict=coord_dict,
            criterion="maximin",
        )
        constraints = pd.DataFrame(
            {
                "easting": sampled_coord_dict["easting"]["sampled_values"],
                "northing": sampled_coord_dict["northing"]["sampled_values"],
                "upward": np.ones_like(sampled_coord_dict["northing"]["sampled_values"])
                * 1e3,
            }
        )

    else:
        fudge_factor = 0
        while True:
            if num_constraints is not None:
                num_y = int(np.ceil((num_constraints / (x / y)) ** 0.5))
                num_x = int(np.ceil(num_constraints / num_y)) + fudge_factor
            elif shape is not None:
                num_x = shape[0]
                num_y = shape[1]

            if spacing is not None:
                pad = (0, 0)
            else:
                # if (num_x % 2 == 0) and (num_y % 2 == 0):
                pad = (-height / (num_y) / 2, -width / (num_x) / 2)
                # elif num_x % 2 == 0:
                #     pad = (0, -width/(num_x)/2)
                # elif num_y % 2 == 0:
                #     pad = (-height/(num_y)/2, 0)
                # else:
                #     pad = (0, 0)

            reg = vd.pad_region(region, pad)

            if spacing is not None:
                x = np.arange(reg[0], reg[1], spacing[0])
                y = np.arange(reg[2], reg[3], spacing[1])

                # center of region
                x_reg_mid = (reg[1] - reg[0]) / 2
                y_reg_mid = (reg[3] - reg[2]) / 2

                # center of arrays
                x_mid = (x[-1] - x[0]) / 2
                y_mid = (y[-1] - y[0]) / 2

                # shift to be centered
                xshift = x_reg_mid - x_mid
                yshift = y_reg_mid - y_mid
                x += xshift
                y += yshift
            else:
                if num_x == 1:
                    x = [(reg[1] + reg[0]) / 2]
                else:
                    x = np.linspace(reg[0], reg[1], num_x)
                if num_y == 1:
                    y = [(reg[3] + reg[2]) / 2]
                else:
                    y = np.linspace(reg[2], reg[3], num_y)

            # if len(x) == 1:
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
            rand = np.random.default_rng(seed=seed)
            constraints = df.copy()
            constraints["northing"] = rand.normal(df.northing, shift_stdev)
            constraints["easting"] = rand.normal(df.easting, shift_stdev)

            # keep only set number of constraints
            if shape is not None or spacing is not None:
                break
            try:
                constraints = constraints.sample(n=num_constraints, random_state=seed)
            except ValueError:
                fudge_factor += 0.1
            else:
                break

    # check whether points are inside or outside of shp
    if shapefile is not None:
        gdf = gpd.GeoDataFrame(
            constraints,
            geometry=gpd.points_from_xy(x=constraints.easting, y=constraints.northing),
            crs="EPSG:3031",
        )
        constraints["inside"] = gdf.within(gpd.read_file(shapefile).geometry[0])
        try:  # noqa: SIM105
            constraints = constraints.drop(columns="geometry")
        except KeyError:
            pass
        # drop outside constraints
        constraints = constraints[constraints.inside]

    # ensure all points are within region
    constraints = utils.points_inside_region(
        constraints, region, names=("easting", "northing")
    )

    constraints = constraints.drop(columns="upward")

    if add_outside_points:
        constraints["inside"] = True

        # make empty grid
        coords = vd.grid_coordinates(
            region=full_region,
            spacing=grid_spacing,
            pixel_register=False,
        )
        grd = vd.make_xarray_grid(
            coords,
            data=np.ones_like(coords[0]) * 1e3,
            data_names="upward",
            dims=("northing", "easting"),
        ).upward
        # mask to shapefile
        masked = utils.mask_from_shp(
            shapefile=shapefile,
            xr_grid=grd,
            masked=True,
        ).rename("upward")
        outside_constraints = vd.grid_to_table(masked).dropna()
        outside_constraints = outside_constraints.drop(columns="upward")
        outside_constraints["inside"] = False

        constraints = pd.concat([outside_constraints, constraints], ignore_index=True)

    if plot:
        fig = maps.basemap(
            fig_height=8,
            region=full_region,
            frame=True,
        )

        fig.plot(
            x=constraints.easting,
            y=constraints.northing,
            style="c2p",
            fill="black",
        )

        if shapefile is not None:
            fig.plot(
                shapefile,
                pen="0.2p,black",
            )

        fig.show()

    return constraints


def constraint_layout_spacing(
    spacing,
    shift_stdev=0,
    region=None,
    shapefile=None,
    padding=None,
    plot=False,
):
    if shapefile is not None:
        bounds = gpd.read_file(shapefile).bounds
        region = [bounds.minx, bounds.maxx, bounds.miny, bounds.maxy]
        region = [x.values[0] for x in region]

    # create regular grid, with set number of constraint points
    reg = vd.pad_region(region, padding) if padding is not None else region

    # start grid from edges
    # x = np.arange(reg[0], reg[1], spacing)
    # y = np.arange(reg[2], reg[3], spacing)

    # start grid from center
    x_mid = reg[0] + (reg[1] - reg[0]) / 2
    y_mid = reg[2] + (reg[3] - reg[2]) / 2
    x1 = np.arange(x_mid, reg[0], -spacing)
    x2 = np.arange(x_mid + spacing, reg[1], spacing)
    y1 = np.arange(y_mid, reg[2], -spacing)
    y2 = np.arange(y_mid + spacing, reg[3], spacing)
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

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
            geometry=gpd.points_from_xy(x=constraints.easting, y=constraints.northing),
            crs="EPSG:3031",
        )
        constraints["inside"] = gdf.within(gpd.read_file(shapefile).geometry[0])
        constraints = constraints.drop(columns="geometry")
    else:
        constraints["inside"] = True

    # drop outside constraints
    constraints = constraints[constraints.inside]

    # ensure all points are within region
    constraints = utils.points_inside_region(
        constraints, region, names=("easting", "northing")
    )

    if plot:
        fig = maps.basemap(
            fig_height=8,
            region=region,
            frame=True,
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


def airborne_survey(
    along_line_spacing: float,
    grav_observation_height: float,
    region: tuple[float, float, float, float],
    NS_line_spacing: float | None = None,
    EW_line_spacing: float | None = None,
    NS_line_number: float | None = None,
    EW_line_number: float | None = None,
    padding: float | None = None,
    NS_lines_to_remove: list[int] | None = None,
    EW_lines_to_remove: list[int] | None = None,
    grav_grid: xr.DataArray | None = None,
    plot: bool = False,
):
    if padding is not None:
        region = vd.pad_region(region, padding)

    width = region[1] - region[0]
    height = region[3] - region[2]

    # center of region
    x_reg_mid = (region[1] - region[0]) / 2
    y_reg_mid = (region[3] - region[2]) / 2

    # simulate N-S tie lines
    if NS_line_spacing is not None:
        x = np.arange(region[0], region[1], NS_line_spacing)
        # center of arrays
        x_mid = (x[-1] - x[0]) / 2
        # shift to be centered
        xshift = x_reg_mid - x_mid
        x += xshift
    elif NS_line_number is not None:
        if NS_line_number == 1:
            x = [(region[1] + region[0]) / 2]
        else:
            pad = (0, -width / (NS_line_number) / 2)
            reg = vd.pad_region(region, pad)
            x = np.linspace(reg[0], reg[1], NS_line_number)

    y = np.arange(region[2], region[3], along_line_spacing)

    # remove select N-S lines,starting from left
    if NS_lines_to_remove is not None:
        x = np.delete(x, NS_lines_to_remove)

    # calculate median spacing
    # NS_points = [[0, i] for i in x]
    # NS_median_spacing = (
    #     np.median(
    #         vd.median_distance(
    #             NS_points,
    #             k_nearest=1,
    #         )
    #     )
    #     / 1e3
    # )
    coords = np.meshgrid(x, y)

    # turn coordinates into dataarray
    ties = vd.make_xarray_grid(
        coords,
        data=np.ones_like(coords[0]) * grav_observation_height,
        data_names="upward",
        dims=("northing", "easting"),
    )
    # turn dataarray into dataframe
    df_ties = vd.grid_to_table(ties)

    # give each tie line a number starting at 1000 in increments of 10
    df_ties["line"] = np.nan
    for i, j in enumerate(df_ties.easting.unique()):
        df_ties["line"] = np.where(df_ties.easting == j, 1000 + i * 10, df_ties.line)

    # simulate E-W flight line
    if EW_line_spacing is not None:
        y = np.arange(region[2], region[3], EW_line_spacing)
        # center of arrays
        y_mid = (y[-1] - y[0]) / 2
        # shift to be centered
        yshift = y_reg_mid - y_mid
        y += yshift
    elif EW_line_number is not None:
        if EW_line_number == 1:
            y = [(region[2] + region[3]) / 2]
        else:
            pad = (-height / (EW_line_number) / 2, 0)
            reg = vd.pad_region(region, pad)
            y = np.linspace(reg[2], reg[3], EW_line_number)

    x = np.arange(region[0], region[1], along_line_spacing)

    # remove select E-W lines, starting from bottom
    if EW_lines_to_remove is not None:
        y = np.delete(y, EW_lines_to_remove)

    coords = np.meshgrid(x, y)

    # turn coordinates into dataarray
    lines = vd.make_xarray_grid(
        coords,
        data=np.ones_like(coords[0]) * grav_observation_height,
        data_names="upward",
        dims=("northing", "easting"),
    )
    # turn dataarray into dataframe
    df_lines = vd.grid_to_table(lines)

    # give each line a number starting at 0 in increments of 10
    df_lines["line"] = np.nan
    for i, j in enumerate(df_lines.northing.unique()):
        df_lines["line"] = np.where(df_lines.northing == j, i + 1 * 10, df_lines.line)

    # merge dataframes
    df = pd.concat([df_ties, df_lines])

    # add a time column
    df["time"] = np.nan
    for i in df.line.unique():
        if i >= 1000:
            time = df[df.line == i].sort_values("northing").reset_index().index.values
        else:
            time = df[df.line == i].sort_values("easting").reset_index().index.values
        df.loc[df.line == i, "time"] = time

    # convert to geopandas
    df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(x=df.easting, y=df.northing),
        crs="EPSG:3031",
    )

    # calculate distance along each line
    df["dist_along_line"] = gravity_processing.distance_along_line(
        df,
        line_col_name="line",
        time_col_name="time",
    )

    # calculate median spacing
    # np.median(vd.median_distance(
    #     (constraint_points.easting, constraint_points.northing),
    #     k_nearest=1,
    # ))/1e3

    # df["median_spacing"] = median_spacing

    # sample gravity at points and regrid
    if grav_grid is not None:
        df = profiles.sample_grids(
            df,
            grav_grid,
            "gravity_anomaly",
            coord_names=("easting", "northing"),
        )

    if plot:
        fig = maps.basemap(
            region=vd.pad_region(region, 0.1 * (region[1] - region[0])),
            frame=True,
        )
        fig.plot(
            x=[region[0], region[0], region[1], region[1], region[0]],
            y=[region[2], region[3], region[3], region[2], region[2]],
            pen=".5p,black",
            label="inversion region",
        )

        if grav_grid is not None:
            pygmt.makecpt(
                cmap="viridis",
                series=[df.gravity_anomaly.min(), df.gravity_anomaly.max()],
                background=True,
            )
            fig.plot(
                x=df.easting,
                y=df.northing,
                style="c0.1c",
                fill=df.gravity_anomaly,
                cmap=True,
            )
            maps.add_colorbar(fig, cbar_label="mGal", cbar_yoffset=1)
        fig.plot(
            df[["easting", "northing"]],
            style="p",
            fill="red",
            label="observation points",
        )

        fig.legend()
        fig.show()

    return df
