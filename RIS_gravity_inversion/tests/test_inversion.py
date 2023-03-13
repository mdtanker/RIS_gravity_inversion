#%%
import pytest

from RIS_gravity_inversion import inversion
import xarray as xr
import numpy as np
import pandas as pd
import verde as vd
import harmonica as hm

def dummy_df():

    data = {
        "northing": [200, 200, 400, 400,],
        "easting": [-100, 100, -100, 100,],
        "upward": [20, 20, 20, 20],
        "forward_grav": [12, 13, 14, 15],
        "observed": [113, 111, 115, 114],
    }
    # data = {
    #     "northing": [100, 100, 500, 500],
    #     "easting": [-200, 200, -200, 200],
    #     "upward": [20, 20, 20, 20],
    #     "forward_grav": [12, 13, 14, 15],
    #     "observed": [113, 111, 115, 114],
    # }
    df = pd.DataFrame(data)
    return df


def dummy_df_big():

    df = dummy_prism_layer().to_dataframe().reset_index().dropna().astype(float)
    df = df.drop(columns=["top", "bottom", "density"])
    df['upward'] = 20
    df['misfit'] = [-1,-1,-1, 0,0,0, 1,1,1]
    return df


def dummy_prism_layer():
    """
    Create a dummy prism layer
    """
    (easting, northing) = vd.grid_coordinates(region=[-200, 200, 100, 500], spacing=200)
    surface = [[0,0,0],[-30,-30,-30],[30,30,30]]
    density = 2670.0 * np.ones_like(surface)
    prism_layer = hm.prism_layer(
        coordinates=(easting[0, :], northing[:, 0]),
        surface=surface,
        reference=-100,
        properties={"density": density},
    )
    return prism_layer


def dummy_prism_layer_flat():
    """
    Create a dummy prism layer
    """
    (easting, northing) = vd.grid_coordinates(region=[-200, 200, 100, 500], spacing=200)
    surface = np.zeros_like(easting)
    density = 2670.0 * np.ones_like(surface)
    prism_layer = hm.prism_layer(
        coordinates=(easting[0, :], northing[:, 0]),
        surface=surface,
        reference=-100,
        properties={"density": density},
    )
    return prism_layer


def dummy_jacobian():
    """
    Create a under-determined jacobian with vertical derivative values
    """
    grav = dummy_df()

    prisms_layer = dummy_prism_layer_flat()

    prisms_properties = inversion.prism_properties(
            prisms_layer,
            method="itertools"
        )

    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )

    jac = inversion.jacobian_prism(
        prisms_properties,
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        .001,
        jac,
    )
    return jac


def dummy_jacobian_square():
    """
    Create a square jacobian with vertical derivative values
    """
    grav = dummy_df_big()

    prisms_layer = dummy_prism_layer_flat()

    prisms_properties = inversion.prism_properties(
            prisms_layer,
            method="itertools"
        )

    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )

    jac = inversion.jacobian_prism(
        prisms_properties,
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        .001,
        jac,
    )
    return jac


def test_misfit():
    """
    test the misfit function
    """

    # DC shift the observed and calculate the misfit
    misfit = inversion.misfit(
        dummy_df(),
        input_forward_column="forward_grav",
        input_grav_column="observed",
    )

    # test that the observed gravity has been shifted correctly
    assert (np.median(misfit.observed_shift)-np.median(misfit.forward_grav)) == pytest.approx(0, rel=1e-10)

    # test that the misfit has been calculated correctly
    expected_misfit = np.array([1, -2, 1, -1])
    np.array_equal(np.array(misfit.misfit), expected_misfit)


def test_grav_column_der():
    """
    test the grav_column_der function
    Below is a map view of a prism, with the locations of the various observation
    points a-h. The prism is 5x5, with a density of 2670kg/m^3.

    5  d---g---c
       |       |
       |   e   f      h
       |       |
    0  a-------b
       0       5
    """

    a = inversion.grav_column_der(
        grav_easting = 0,
        grav_northing = 0,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    b = inversion.grav_column_der(
        grav_easting = 5,
        grav_northing = 0,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    c = inversion.grav_column_der(
        grav_easting = 5,
        grav_northing = 5,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    d = inversion.grav_column_der(
        grav_easting = 0,
        grav_northing = 5,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    e = inversion.grav_column_der(
        grav_easting = 2.5,
        grav_northing = 2.5,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    f = inversion.grav_column_der(
        grav_easting = 5,
        grav_northing = 2.5,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    g = inversion.grav_column_der(
        grav_easting = 2.5,
        grav_northing = 5,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    h = inversion.grav_column_der(
        grav_easting = 10,
        grav_northing = 2.5,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )

    # test that derivative at all 4 corners of prism is same
    assert a == b == c == d

    # test that derivate on 2 prism edges are the same
    assert f == g

    # test that derivate within prism is same as on the edge
    assert e == f

    # test that derivative further away from prism is smaller
    assert h < a

    # expected result
    dg_z = inversion.grav_column_der(
        grav_easting = 20,
        grav_northing = 20,
        grav_upward = 100,
        prism_easting = np.array([2.5]),
        prism_northing = np.array([2.5]),
        prism_top = np.array([-10]),
        prism_spacing = 5,
        prism_density = np.array([2670]),
    )[0]
    # harmonica prism approximation
    step=.000001
    hm_dg_z = hm.prism_gravity((20,20,100), (0,5,0,5,-10,-10+step), 2670, field="g_z")/step

    # test that the derivative matches a small prism approximation from
    assert dg_z == pytest.approx(hm_dg_z, rel=1e-3)


def test_jacobian_annular():
    """
    test the jacobian_annular function
    """
    grav = dummy_df()

    prisms_layer = dummy_prism_layer()
    prisms_df = prisms_layer.to_dataframe().reset_index().dropna().astype(float)

    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )

    jac = inversion.jacobian_annular(
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        np.array(prisms_df.easting),
        np.array(prisms_df.northing),
        np.array(prisms_df.top),
        np.array(prisms_df.density),
        200,
        jac,
    )

    # test that prisms above observation point have negative vertical derivatives
    assert jac[:,-3:].max() < 0

    # test that prisms below observation point have positive vertical derivatives
    assert jac[:,0:-3].min() > 0


def test_prism_properties():
    """
    test the prism_properties function
    """

    prisms_layer = dummy_prism_layer()

    itertools_result = inversion.prism_properties(
        prisms_layer,
        method="itertools"
    )

    forloops_result = inversion.prism_properties(
        prisms_layer,
        method="forloops"
    )

    generator_result = inversion.prism_properties(
        prisms_layer,
        method="generator"
    )

    # test that the prism properties are the same with 3 methods
    np.array_equal(itertools_result, forloops_result)
    np.array_equal(itertools_result, generator_result)

    # test that the first prism's properties are correct
    np.array_equal(itertools_result[0], np.array([-300, -100, 0, 200, -100, 2670]))


def test_jacobian_prism():
    """
    test the jacobian_prism function
    """
    grav = dummy_df()

    prisms_layer = dummy_prism_layer()

    prisms_properties = inversion.prism_properties(
            prisms_layer,
            method="itertools"
        )

    jac = np.empty(
        (len(grav), prisms_layer.top.size),
        dtype=np.float64,
    )

    jac = inversion.jacobian_prism(
        prisms_properties,
        np.array(grav.easting),
        np.array(grav.northing),
        np.array(grav.upward),
        .001,
        jac,
    )

    # test that prisms above observation point have negative vertical derivatives
    assert jac[:,-3:].max() < 0

    # test that prisms below observation point have positive vertical derivatives
    assert jac[:,0:-3].min() > 0


solver_types=[
    "verde least squares",
    "scipy least squares",
    "scipy constrained",
    # "scipy nonlinear lsqr",
    # "CLR",
    "scipy conjugate",
    "numpy least squares",
    "steepest descent",
    "gauss newton",
]
@pytest.mark.parametrize("solver_type", solver_types)
def test_solver_square(solver_type):
    """
    test the solver function with equal number of prisms and misfit values
    """
    misfit = dummy_df_big().misfit.values
    jac = dummy_jacobian_square()
    correction = inversion.solver(jac, misfit, solver_type=solver_type)

    # test that correction is negative for negative misfits
    assert correction[0:3].max() < -9

    # test that correction is near 0 for misfits with values of 0
    np.testing.assert_allclose(correction[3:6], np.array([0,0,0]), atol=1e-8)

    # test that correction is positive for positive misfits
    assert correction[6:9].min() > 9

solver_types=[
    "verde least squares",
    "scipy least squares",
    "scipy constrained",
    # "scipy nonlinear lsqr",
    # "CLR",
    # "scipy conjugate",
    "numpy least squares",
    "steepest descent",
    # "gauss newton",
]
@pytest.mark.parametrize("solver_type", solver_types)
def test_solver_underdetermined(solver_type):
    """
    test the solver function
    flat prisms surface and base, all obs points above surface, consistent misfit
    values, should result if relatively uniform corrections
    """
    jac = dummy_jacobian()

    # test that correction is 0 if misfits are 0
    misfit = np.array([0,0,0,0])
    correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
    assert correction.all() == 0

    # test that all corrections are negative
    misfit = np.array([-10,-10,-10,-10])
    correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
    assert correction.max() < 0

    # test that all corrections are positive
    misfit = np.array([100,100,100,100])
    correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
    assert correction.min() > 0

    # test that mean correction is close to 0
    misfit = np.array([-100,-100,100,100])
    correction = inversion.solver(jac.copy(), misfit, solver_type=solver_type)
    assert correction.mean() == pytest.approx(0, abs=1e-5)


#%%
test_solver_square("scipy least squares")

# %%
test_solver_underdetermined("numpy least squares")
# %%
