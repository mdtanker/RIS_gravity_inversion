# %%
import numpy as np
import pandas as pd
import pytest
import verde as vd

from RIS_gravity_inversion import regional


def dummy_grid():
    (x, y, z) = vd.grid_coordinates(
        region=[-100, 100, 200, 400],
        spacing=100,
        extra_coords=20,
    )

    # create topographic features
    misfit = y**2

    grid = vd.make_xarray_grid(
        (x, y),
        (misfit, z),
        data_names=("misfit", "upward"),
        dims=("northing", "easting"),
    )

    return grid


def dummy_df():
    df = dummy_grid().to_dataframe().reset_index()
    df["grav"] = 20000
    return df


def test_regional_trend():
    """
    test the regional_trend function
    """
    anomalies = dummy_df()

    df = regional.regional_trend(
        trend=0,
        misfit_grid=dummy_grid().rename({"easting": "x", "northing": "y"}).misfit,
        anomalies=anomalies,
        crs="3031",
    )
    print(df)
    # test  whether regional field has been remove correctly
    # by whether the means of the reg and misfit are similar
    assert np.mean(df.reg) == pytest.approx(np.mean(df.misfit), rel=1e-10)


def test_regional_filter():
    """
    test the regional_filter function
    """
    anomalies = dummy_df()

    df = regional.regional_filter(
        filter_width="g300",
        misfit_grid=dummy_grid().rename({"easting": "x", "northing": "y"}).misfit,
        anomalies=anomalies,
        registration="g",
    )

    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)

    # test  whether regional field has been remove correctly
    # by whether the limits of the regional are smaller than the limits of the gravity
    # with a 10% margin
    assert reg_range < misfit_range * 0.9


def test_regional_constraints():
    """
    test the regional_constraints function
    """
    anomalies = dummy_df()
    points = pd.DataFrame(
        {"easting": [-50, -30, 0, 10, 50], "northing": [210, 280, 240, 360, 310]}
    )
    df = regional.regional_constraints(
        constraint_points=points,
        misfit_grid=dummy_grid().rename({"easting": "x", "northing": "y"}).misfit,
        anomalies=anomalies,
        region=[-100, 100, 200, 400],
        spacing=100,
        grid_method="verde",
    )

    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    # test  whether regional field has been remove correctly
    # by whether the regional values are close the the grav values at the constraints
    assert reg_range < misfit_range * 0.9


def test_regional_eq_sources():
    """
    test the regional_eq_sources function
    """
    anomalies = dummy_df()
    anomalies["misfit"] = np.random.normal(100, 100, len(anomalies))

    df = regional.regional_eq_sources(
        source_depth=1000e3,
        anomalies=anomalies,
    )
    print(df)
    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    print(reg_range, misfit_range)
    # test  whether regional field has been remove correctly
    # by whether the regional values are close the the grav values at the constraints
    assert reg_range < misfit_range * 0.9
