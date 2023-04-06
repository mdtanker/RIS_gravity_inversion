# %%
import numpy as np
import pandas as pd
import pytest
import verde as vd

from RIS_gravity_inversion import utils


def dummy_grid():
    (x, y, z) = vd.grid_coordinates(
        region=[000, 200, 200, 400],
        spacing=100,
        extra_coords=20,
    )

    # create topographic features
    data = y**2+ x**2

    grid = vd.make_xarray_grid(
        (x, y),
        (data, z),
        data_names=("scalars", "upward"),
        dims=("northing", "easting"),
    )

    return grid


def test_RMSE():
    """
    test the RMSE function
    """
    # create some dummy data
    data = np.array([1, 2, 3])
    # calculate the RMSE
    rmse = utils.RMSE(data)
    # test that the RMSE is 0
    assert rmse == 2.0


@pytest.mark.parametrize("test_input", ["pygmt", "rioxarray", "verde"])
def test_nearest_grid_fill(test_input):

    # make a grid with a hole in it
    grid = dummy_grid().scalars

    grid.loc[dict(easting=100, northing=200)]=np.nan
    grid.plot()
    print(grid)
    # check the grid has a hole
    assert grid.isnull().any() == True

    # fill the hole
    filled = utils.nearest_grid_fill(grid, method=test_input)

    # check that the hole has been filled
    assert filled.isnull().any() == False

    # check fill value is equal to one of the adjacent cells
    expected = [filled.loc[dict(easting=0, northing=200)],
                filled.loc[dict(easting=200, northing=200)],
                filled.loc[dict(easting=100, northing=300)],
                ]
    assert filled.loc[dict(easting=100, northing=200)] in expected


def test_filter_grid():
    """
    test the filter_grid function
    """
    # create some dummy data
    grid = dummy_grid().scalars

    # filter the grid
    filtered = utils.filter_grid(grid, 10000, filt_type="lowpass")

    # check that the grid has been low-pass filtered
    assert np.max(filtered) < np.mean(grid)
    # return filtered, grid


def test_dist_nearest_points():
    """
    test the dist_nearest_points function
    """
    # create some dummy data
    targets = pd.DataFrame({"x": [0, 2], "y": [0, -1]})

    df = pd.DataFrame(
        {"x": [-4, 4, 0, 4, -4], "y": [-4, 4, 0, -4, 4], "z": [0, 1, 2, 3, 4]}
    )

    da = df.set_index(["y", "x"]).to_xarray().z
    ds = df.set_index(["y", "x"]).to_xarray()

    # calculate the distance with a df
    dist_df = utils.dist_nearest_points(
        targets,
        df,
        coord_names=["x", "y"],
    )

    # calculate the distance with a da
    dist_da = utils.dist_nearest_points(
        targets,
        da,
        coord_names=["x", "y"],
    )

    # calculate the distance with a ds
    dist_ds = utils.dist_nearest_points(
        targets,
        ds,
        coord_names=["x", "y"],
    )

    ds_results = np.array(vd.grid_to_table(dist_ds).min_dist)
    da_results = np.array(vd.grid_to_table(dist_da).min_dist)
    df_results = np.array(dist_df.min_dist)

    # test that the results all match
    np.array_equal(ds_results, da_results)
    np.array_equal(da_results, df_results)

    # test that smallest min_dist and largest min_dist are correct
    assert np.min(df_results) == pytest.approx(0)
    assert np.max(df_results) == pytest.approx(5.656854)


def test_weight_grid():
    pass

def test_constraints_grid():
    pass

def test_prep_grav_data():
    pass

def test_block_reduce_gravity():
    pass

def test_normalize_xarray():
    pass

def test_grids_to_prisms():
    pass

def test_forward_grav_of_prismlayer():
    pass

def test_sample_bounding_surface():
    pass

def test_enforce_confining_surface():
    pass

def test_constrain_surface_correction():
    pass


