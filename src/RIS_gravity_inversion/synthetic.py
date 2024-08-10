import geopandas as gpd
import numpy as np
import verde as vd
from polartoolkit import maps, utils


def constraint_layout_number(
    num_constraints,
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

    x = region[1] - region[0]
    y = region[3] - region[2]
    num_y = int(np.ceil((num_constraints / (x / y)) ** 0.5))

    fudge_factor = 0
    while True:
        num_x = int(np.ceil(num_constraints / num_y)) + fudge_factor

        # create regular grid, with set number of constraint points
        if padding is not None:
            reg = vd.pad_region(region, padding)
        else:
            reg = region
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
            constraints["inside"] = gdf.within(gpd.read_file(shapefile).geometry[0])
            try:
                constraints.drop(columns="geometry", inplace=True)
            except KeyError:
                pass
            # drop outside constraints
            constraints = constraints[constraints.inside]
        # else:
        # constraints["inside"] = True

        # ensure all points are within region
        constraints = utils.points_inside_region(
            constraints, region, names=("easting", "northing")
        )

        # keep only set number of constraints
        try:
            constraints = constraints.sample(n=num_constraints, random_state=0)
        except ValueError:
            fudge_factor += 0.1
        else:
            break

    constraints.drop(columns="upward", inplace=True)

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
    if padding is not None:
        reg = vd.pad_region(region, padding)
    else:
        reg = region

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
        constraints.drop(columns="geometry", inplace=True)
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
