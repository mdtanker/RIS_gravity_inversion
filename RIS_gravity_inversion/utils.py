import matplotlib.pyplot as plt
import numpy as np
import verde as vd


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


def synthetic_topography_simple(
    spacing,
    region,
    plot_individuals=False,
):

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(region=region, spacing=spacing)

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    feature1 = exponential_surface(
        x, y, region, -3500, -600, 1e11, x_range * 0.4, y_range * -0.2
    )
    feature2 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 3,
            sigma_y=y_range * 0.4,
            x0=region[0] + x_range * 0.2,
            y0=region[2] + y_range * 0.6,
            angle=10,
        )
        * -600
    )
    feature3 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.2,
            sigma_y=y_range * 7,
            x0=region[0] + x_range * 0.8,
            y0=region[2] + y_range * 0.3,
            angle=80,
        )
        * 1000
    )

    features = [feature1, feature2, feature3]

    topo = sum(features)

    grid = vd.make_xarray_grid((x, y), topo, data_names="z", dims=("y", "x")).z

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
):

    # create grid of coordinates
    (x, y) = vd.grid_coordinates(region=region, spacing=spacing)

    # get x and y range
    x_range = abs(region[1] - region[0])
    y_range = abs(region[3] - region[2])

    # create topographic features
    # regional
    feature1 = exponential_surface(
        x, y, region, -500, -600, 1e11, x_range * 0.4, y_range * -0.2
    )
    feature2 = exponential_surface(
        x, y, region, -500, 300, 1e10, x_range * -0.2, y_range * -0.3
    )

    # high-frequency
    feature3 = exponential_surface(
        x, y, region, -500, -150, 1e8, x_range * 0.2, y_range * 0.2
    )
    feature4 = exponential_surface(
        x, y, region, -500, 250, 5e7, x_range * -0.1, y_range * -0.2
    )
    feature5 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 0.03,
            sigma_y=y_range * 0.5,
            x0=region[0] + x_range * 0.2,
            y0=region[2] + y_range * 0.4,
            angle=20,
        )
        * -400
    )
    feature6 = (
        gaussian2d(
            x,
            y,
            sigma_x=x_range * 1,
            sigma_y=y_range * 0.05,
            x0=region[0] + x_range * 0.8,
            y0=region[2] + y_range * 0.4,
            angle=45,
        )
        * 200
    )

    features = [feature1, feature2, feature3, feature4, feature5, feature6]

    topo = sum(features)

    grid = vd.make_xarray_grid((x, y), topo, data_names="z", dims=("y", "x")).z

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


def contaminate(data, stddev, percent=False, return_stddev=False, seed=None):
    r"""
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
            stddev[i] = stddev[i] * max(abs(data[i]))
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
