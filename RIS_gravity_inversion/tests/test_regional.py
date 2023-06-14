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


@pytest.mark.parametrize("test_input", ["pygmt", "rioxarray", "verde"])
def test_regional_trend(test_input):
    """
    test the regional_trend function
    """
    anomalies = dummy_df()

    df = regional.regional_trend(
        trend=0,
        misfit_grid=dummy_grid().misfit,
        anomalies=anomalies,
        fill_method=test_input,
    )

    assert len(df.misfit) == len(df.reg)

    # test  whether regional field has been remove correctly
    # by whether the means of the reg and misfit are similar
    assert np.mean(df.reg) == pytest.approx(np.mean(df.misfit), rel=1e-10)

    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    # test  whether regional field has been remove correctly
    # by whether the limits of the regional are smaller than the limits of the gravity
    # with a 10% margin
    assert reg_range < (misfit_range * 0.9)

    # test that the regional values are between the misfit values
    assert np.max(df.reg) < np.max(df.misfit)
    assert np.min(df.reg) > np.min(df.misfit)


def test_regional_filter():
    """
    test the regional_filter function
    """
    anomalies = dummy_df()

    df = regional.regional_filter(
        filter_width="g300",
        misfit_grid=dummy_grid().misfit,
        anomalies=anomalies,
        registration="g",
    )

    assert len(df.misfit) == len(df.reg)

    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)

    # test  whether regional field has been remove correctly
    # by whether the limits of the regional are smaller than the limits of the gravity
    # with a 10% margin
    assert reg_range < (misfit_range * 0.9)
    # test that the mean regional value is in the range of the misfit values
    assert np.mean(df.reg) < np.max(df.misfit)
    assert np.mean(df.reg) > np.min(df.misfit)


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
        misfit_grid=dummy_grid().misfit,
        anomalies=anomalies,
        region=[-100, 100, 200, 400],
        spacing=100,
        grid_method="verde",
    )

    assert len(df.misfit) == len(df.reg)

    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    # test  whether regional field has been remove correctly
    # by whether the range of regional values are lower than the range of misfit values
    assert reg_range < misfit_range * 0.9
    # test that the mean regional value is in the range of the misfit values
    assert np.max(df.reg) < np.max(df.misfit)
    assert np.min(df.reg) > np.min(df.misfit)
    # test that the mean regional value is in the range of the misfit values
    assert np.mean(df.reg) < np.max(df.misfit)
    assert np.mean(df.reg) > np.min(df.misfit)


def test_regional_eq_sources():
    """
    test the regional_eq_sources function
    """
    anomalies = dummy_df()
    anomalies["misfit"] = np.random.normal(100, 100, len(anomalies))

    df = regional.regional_eq_sources(
        source_depth=100e3,
        anomalies=anomalies,
    )
    print(df)
    reg_range = np.max(df.reg) - np.min(df.reg)
    misfit_range = np.max(df.misfit) - np.min(df.misfit)
    print(reg_range, misfit_range)
    # test  whether regional field has been remove correctly
    # by whether the range of regional values are lower than the range of misfit values
    assert reg_range < misfit_range


# %%
