# %%
import pandas as pd
import pytest
import xarray as xr
import numpy as np
from RIS_gravity_inversion import utils
import verde as vd
import numpy as np

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

    #test that smallest min_dist and largest min_dist are correct
    assert np.min(df_results) == pytest.approx(0)
    assert np.max(df_results) == pytest.approx(5.656854)
