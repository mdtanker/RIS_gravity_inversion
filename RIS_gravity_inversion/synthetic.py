import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import verde as vd
from antarctic_plots import maps, utils

import RIS_gravity_inversion.utils as inv_utils


def gaussian2d(x, y, sigma_x, sigma_y, x0=0, y0=0, angle=0.0):
    """
    From Fatiando-Legacy
    Non-normalized 2D Gaussian function

    Parameters:

    * x, y : float or arrays
        Coordinates at which to calculate the Gaussian function
    * sigma_x, sigma_y : float
        Standard deviation in the x and y directions
    * x0, y0 : float
        Coordinates of the center of the distribution
    * angle : float
        Rotation angle of the gaussian measure from the x axis (north) growing
        positive to the east (positive y axis)

    Returns:

    * gauss : array
        Gaussian function evaluated at *x*, *y*

    """
    theta = -1 * angle * np.pi / 180.0
    tmpx = 1.0 / sigma_x**2
    tmpy = 1.0 / sigma_y**2
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    a = tmpx * costheta + tmpy * sintheta**2
    b = (tmpy - tmpx) * costheta * sintheta
    c = tmpx * sintheta**2 + tmpy * costheta**2
    xhat = x - x0
    yhat = y - y0
    return np.exp(-(a * xhat**2 + 2.0 * b * xhat * yhat + c * yhat**2))


def exponential_surface(
    x,
    y,
    region,
    base_level,
    scaling,
    decay,
    x_shift=0,
    y_shift=0,
):
    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    x_center = x - (region[0] + x_range / 2)
    y_center = y - (region[2] + y_range / 2)

    func = np.exp(-((x_center - x_shift) ** 2 + (y_center - y_shift) ** 2) / decay)

    return base_level + (scaling * func)


def synthetic_topography_upper(
    spacing,
    region,
    low=0,
    high=1,
    registration="g",
):
    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        pixel_register=pixel_register,
    )

    # create topographic features
    feature = (60e3 - x) ** 2 + (y - 30e3) ** 2

    grid = vd.make_xarray_grid(
        (x, y),
        feature,
        data_names="z",
        dims=("y", "x"),
    ).z

    grid = inv_utils.normalize_xarray(grid, low=low, high=high)

    return grid


def synthetic_topography_regional(
    spacing,
    region,
    plot_individuals=False,
    registration="g",
    scale=1,
):
    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        pixel_register=pixel_register,
    )

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    feature1 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 2,
            sigma_y=y_range * 2,
            x0=region[0] + x_range,
            y0=region[2] + y_range * 0.5,
            angle=10,
        )
        * -150
        * scale
    ) - 3500
    feature2 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 3,
            sigma_y=y_range * 0.4,
            x0=region[0] + x_range * 0.2,
            y0=region[2] + y_range * 0.4,
            angle=-10,
        )
        * -100
        * scale
    )
    feature3 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.2,
            sigma_y=y_range * 7,
            x0=region[0] + x_range * 0.8,
            y0=region[2] + y_range * 0.7,
            angle=-80,
        )
        * 150
        * scale
    )

    features = [feature1, feature2, feature3]

    topo = sum(features)

    grid = vd.make_xarray_grid(
        (x, y),
        topo,
        data_names="z",
        dims=("y", "x"),
    ).z

    if plot_individuals is True:
        sub_width = 5
        nrows, ncols = 1, len(features)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width * ncols, sub_width * nrows),
        )
        for i, (f, ax) in enumerate(zip(features, ax.T.ravel())):
            feature = vd.make_xarray_grid((x, y), f, data_names="z", dims=("y", "x")).z
            feature.plot(
                ax=ax,
                x="x",
                y="y",
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )
            # set axes labels and make proportional
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_aspect("equal")

    return grid


def synthetic_topography(
    spacing,
    region,
    plot_individuals=False,
    registration="g",
):
    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        pixel_register=pixel_register,
    )

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    # regional
    f1 = exponential_surface(
        x,
        y,
        region,
        base_level=-300,
        scaling=-500,
        decay=1e12,
        x_shift=x_range * 0.4,
        y_shift=y_range * -0.2,
    )
    f2 = exponential_surface(
        x,
        y,
        region,
        base_level=0,
        scaling=200,
        decay=1e11,
        x_shift=x_range * -0.2,
        y_shift=y_range * -0.3,
    )

    # high-frequency
    # circular
    f3 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.015,
            sigma_y=y_range * 0.015,
            x0=region[0] + x_range * 0.35,
            y0=region[2] + y_range * 0.5,
        )
        * -100
    )
    f4 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.02,
            sigma_y=y_range * 0.02,
            x0=region[0] + x_range * 0.65,
            y0=region[2] + y_range * 0.5,
        )
        * 200
    )
    f5 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.03,
            sigma_y=y_range * 0.03,
            x0=region[0] + x_range * 0.5,
            y0=region[2] + y_range * 0.35,
        )
        * 50
    )
    f6 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.04,
            sigma_y=y_range * 0.04,
            x0=region[0] + x_range * 0.5,
            y0=region[2] + y_range * 0.65,
        )
        * -300
    )

    # elongate
    f7 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.25,
            sigma_y=y_range * 0.03,
            x0=region[0] + x_range * 0.3,
            y0=region[2] + y_range * 0.7,
            angle=45,
        )
        * -300
    )
    f8 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.7,
            sigma_y=y_range * 0.02,
            x0=region[0] + x_range * 0.7,
            y0=region[2] + y_range * 0.7,
            angle=-45,
        )
        * 50
    )
    f9 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.7,
            sigma_y=y_range * 0.1,
            x0=region[0] + x_range * 0.3,
            y0=region[2] + y_range * 0.3,
            angle=-45,
        )
        * -100
    )
    f10 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.7,
            sigma_y=y_range * 0.08,
            x0=region[0] + x_range * 0.7,
            y0=region[2] + y_range * 0.3,
            angle=45,
        )
        * 200
    )

    features = [
        f1,
        f2,
        f3,
        f4,
        f5,
        f6,
        f7,
        f8,
        f9,
        f10,
    ]

    topo = sum(features)

    grid = vd.make_xarray_grid(
        (x, y),
        topo,
        data_names="z",
        dims=("y", "x"),
    ).z

    if plot_individuals is True:
        sub_width = 5
        nrows, ncols = 2, int(len(features) / 2)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width * ncols, sub_width * nrows),
        )
        for i, (f, ax) in enumerate(zip(features, ax.T.ravel())):
            feature = vd.make_xarray_grid((x, y), f, data_names="z", dims=("y", "x")).z
            feature.plot(
                ax=ax,
                x="x",
                y="y",
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )
            # set axes labels and make proportional
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_aspect("equal")

    return grid


def synthetic_topography_simple(
    spacing,
    region,
    plot_individuals=False,
    registration="g",
):
    if registration == "g":
        pixel_register = False
    elif registration == "p":
        pixel_register = True

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(
        region=region,
        spacing=spacing,
        pixel_register=pixel_register,
    )

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    # regional
    f1 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 1.6,
            sigma_y=y_range * 1.6,
            x0=region[0] + x_range * 0.9,
            y0=region[2] + y_range * 0.3,
        )
        * -800
    )

    # high-frequency
    # circular
    f2 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.03,
            sigma_y=y_range * 0.03,
            x0=region[0] + x_range * 0.35,
            y0=region[2] + y_range * 0.5,
        )
        * -100
    )
    f3 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.08,
            sigma_y=y_range * 0.08,
            x0=region[0] + x_range * 0.65,
            y0=region[2] + y_range * 0.5,
        )
        * 200
    )

    # elongate
    f4 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.5,
            sigma_y=y_range * 0.06,
            x0=region[0] + x_range * 0.3,
            y0=region[2] + y_range * 0.7,
            angle=45,
        )
        * -300
    )
    f5 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 1.4,
            sigma_y=y_range * 0.04,
            x0=region[0] + x_range * 0.7,
            y0=region[2] + y_range * 0.7,
            angle=-45,
        )
        * 50
    )

    features = [
        f1,
        f2,
        f3,
        f4,
        f5,
    ]

    topo = sum(features)

    grid = vd.make_xarray_grid(
        (x, y),
        topo,
        data_names="z",
        dims=("y", "x"),
    ).z

    if plot_individuals is True:
        sub_width = 5
        nrows, ncols = 2, int(len(features) / 2)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(sub_width * ncols, sub_width * nrows),
        )
        for i, (f, ax) in enumerate(zip(features, ax.T.ravel())):
            feature = vd.make_xarray_grid((x, y), f, data_names="z", dims=("y", "x")).z
            feature.plot(
                ax=ax,
                x="x",
                y="y",
                cbar_kwargs={
                    "orientation": "horizontal",
                    "anchor": (1, 1),
                    "fraction": 0.05,
                    "pad": 0.04,
                },
            )
            # set axes labels and make proportional
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_aspect("equal")

    return grid


def contaminate(
    data,
    stddev,
    percent=False,
    percent_as_max_abs=True,
    return_stddev=False,
    seed=None,
):
    """
    From Fatiando-legacy

    Add pseudorandom gaussian noise to an array.

    Noise added is normally distributed with zero mean.

    Parameters:

    * data : array or list of arrays
        Data to contaminate
    * stddev : float or list of floats
        Standard deviation of the Gaussian noise that will be added to *data*
    * percent : True or False
        If ``True``, will consider *stddev* as a decimal percentage and the
        standard deviation of the Gaussian noise will be this percentage of
        the maximum absolute value of *data*
    * return_stddev : True or False
        If ``True``, will return also the standard deviation used to
        contaminate *data*
    * seed : None or int
        Seed used to generate the pseudo-random numbers. If `None`, will use a
        different seed every time. Use the same seed to generate the same
        random sequence to contaminate the data.

    Returns:

    if *return_stddev* is ``False``:

    * contam : array or list of arrays
        The contaminated data array

    else:

    * results : list = [contam, stddev]
        The contaminated data array and the standard deviation used to
        contaminate it.

    Examples:

    >>> import np as np
    >>> data = np.ones(5)
    >>> noisy = contaminate(data, 0.1, seed=0)
    >>> print noisy
    [ 1.03137726  0.89498775  0.95284582  1.07906135  1.04172782]
    >>> noisy, std = contaminate(data, 0.05, seed=0, percent=True,
    ...                          return_stddev=True)
    >>> print std
    0.05
    >>> print noisy
    [ 1.01568863  0.94749387  0.97642291  1.03953067  1.02086391]
    >>> data = [np.zeros(5), np.ones(3)]
    >>> noisy = contaminate(data, [0.1, 0.2], seed=0)
    >>> print noisy[0]
    [ 0.03137726 -0.10501225 -0.04715418  0.07906135  0.04172782]
    >>> print noisy[1]
    [ 0.81644754  1.20192079  0.98163167]

    """
    np.random.seed(seed)
    # Check if dealing with an array or list of arrays
    if not isinstance(stddev, list):
        stddev = [stddev]
        data = [data]

    contam = []
    for i in range(len(stddev)):
        if stddev[i] == 0.0:
            contam.append(data[i])
            continue
        if percent:
            if percent_as_max_abs:
                stddev[i] = stddev[i] * max(abs(data[i]))
            else:
                stddev[i] = stddev[i] * abs(data[i])
        if percent_as_max_abs is True:
            print(f"Standard deviation used for noise: {stddev}")
        noise = np.random.normal(scale=stddev[i], size=len(data[i]))
        # Subtract the mean so that the noise doesn't introduce a systematic
        # shift in the data
        noise -= noise.mean()
        contam.append(np.array(data[i]) + noise)
    np.random.seed()
    if len(contam) == 1:
        contam = contam[0]
        stddev = stddev[0]
    if return_stddev:
        return [contam, stddev]
    else:
        return contam


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
            constraints.drop(columns="geometry", inplace=True)
        else:
            constraints["inside"] = True

        # drop outside constraints
        constraints = constraints[constraints.inside]

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
