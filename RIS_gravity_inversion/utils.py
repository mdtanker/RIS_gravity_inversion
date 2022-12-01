import matplotlib.pyplot as plt
import numpy as np
import verde as vd
import xarray as xr
import pandas as pd
import harmonica as hm
import pygmt
from antarctic_plots import fetch, maps, profile, utils
from scipy.sparse.linalg import lsqr
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_slice

import RIS_gravity_inversion.inversion as inv
import RIS_gravity_inversion.plotting as plots

def import_layers(
    layers_list: list,
    spacing_list: list,
    rho_list: list,
    fname_list: list,
    grav_spacing: float,
    active_layer: str,
    buffer_region: list,
    inversion_region: list,
    grav_file: str,
    registration="g",
    **kwargs,
):
    """
    Read zarrs and csvs, resample/rename to be uniform, and add to layers dictionary

    Parameters
    ----------
    layers_list : list
        list of names of layers
    spacing_list : list
        list of grid spacing of layers, in meters
    rho_list : list
        list of densities to use for layers, in kg/m3
    fname_list : list
        list of zarr file names for input grids
    grav_spacing : float
        spacing to resample gravity grid at
    active_layer : str
        layer which will be inverted for
    buffer_region : list
        inversion region with a buffer zone included
    inversion_region : list
       inversion region
    grav_file : str
        csv file with gravity point observation data
    registration: str
        choose grid registration type for input layers and constraints grid, by default
        "g"

    Other Parameters
    ----------------
    constraint_grid: str
        .nc file of a constraint grid, 0-1.
    constraint_points: str
        .csv file with position of constraints
    input_grav_name: str
        column name of observed gravity / freeair / disturbance
    input_obs_height_name: str
        column name of gravity station elevation
    block_reduction: str
        choose type of block reduction to apply to gravity data

    Returns
    -------
    tuple
        layers (dict),
        grav (pd.DataFrame),
        constraint_grid (xr.DataArray),
        constraint_points (pd.DataFrame),
        constraint_points_RIS (pd.DataFrame)
    """

    constraint_grid = kwargs.get("constraint_grid", None)
    constraint_points = kwargs.get("constraint_points", None)

    input_grav_name = kwargs.get("input_grav_name", "gravity_disturbance")
    input_obs_height_name = kwargs.get("input_obs_height_name", "ellipsoidal_elevation")

    layers_list = pd.Series(layers_list)
    spacing_list = pd.Series(spacing_list)
    rho_list = pd.Series(rho_list)
    fname_list = pd.Series(fname_list)

    # read gravity csv file
    df = pd.read_csv(
        grav_file,
        sep=",",
        header="infer",
        index_col=None,
        compression="gzip",
    )

    # remove other columns
    df = df[["x", "y", input_grav_name, input_obs_height_name]]

    # remove points outside of inversion region
    df = utils.points_inside_region(df, region=inversion_region)

    # get number of grav points before reduction
    prior_len = len(df[input_grav_name])

    # block reduce gravity data
    if kwargs.get("block_reduction", None) is None:
        grav = df.copy()
    elif kwargs.get("block_reduction", None) == "pygmt":
        grav = pygmt.blockmedian(
            df[["x", "y", input_grav_name]],
            spacing=grav_spacing,
            region=inversion_region,
            registration=registration,
        )

        grav[input_obs_height_name] = pygmt.blockmedian(
            df[["x", "y", input_obs_height_name]],
            spacing=grav_spacing,
            region=inversion_region,
            registration=registration,
        )[input_obs_height_name]

    elif kwargs.get("block_reduction", None) == "verde":
        reducer = vd.BlockReduce(
            reduction=np.median,
            spacing=grav_spacing,
            # center_coordinates=False,
            # adjust="region",
        )

        coordinates, data = reducer.filter(
            coordinates=(df.x, df.y),
            data=(
                df[input_grav_name],
                df[input_obs_height_name],
            ),
        )

        blocked = pd.DataFrame(
            data={
                "x": coordinates[0],
                "y": coordinates[1],
                input_grav_name: data[0],
                input_obs_height_name: data[1],
            },
        )
        grav = blocked[vd.inside((blocked.x, blocked.y), inversion_region)].copy()

    post_len = len(grav[input_grav_name])

    if kwargs.get("block_reduction", None) is None:
        print(f"{prior_len} gravity observation points")
    else:
        print(f"Block-reduced the gravity data at {int(grav_spacing)}m spacing")
        print(f"from {prior_len} points to {post_len} points")

    # center gravity around 0
    grav[input_grav_name] -= grav[input_grav_name].mean()

    # set standard names
    grav.rename(
        columns={input_grav_name: "Gobs", input_obs_height_name: "z"},
        inplace=True,
    )

    # make nested dictionary for layers and properties
    layers = {
        j: {"spacing": spacing_list[i], "fname": fname_list[i], "rho": rho_list[i]}
        for i, j in enumerate(layers_list)
    }

    # read and resample layer grids, convert to dataframes
    for k, v in layers.items():
        tmp_grid = xr.open_zarr(v["fname"])
        tmp_grid = tmp_grid[list(tmp_grid.keys())[0]].squeeze()
        print(f"\n{'':*<20}Resampling {k} layer {'':*>20}")
        v["grid"] = fetch.resample_grid(
            tmp_grid,
            spacing=v["spacing"],
            region=buffer_region,
            registration=registration,
            verbose="q",
        )

        # print spacing, region, max,min, and registration of layers
        print(f"{k} info: {utils.get_grid_info(v['grid'])}")

        v["df"] = v["grid"].to_dataframe().reset_index()
        v["df"]["rho"] = v["rho"]
        v["df"].dropna(how="any", inplace=True)
        v["len"] = len(v["df"].x)

    if constraint_grid is not None:
        # open zarr file
        tmp_grid = xr.open_zarr(constraint_grid).to_array().squeeze()
        # resample constraint grid
        print(f"\n{'':*<20}Resampling constraints grid {'':*>20}")
        constraint_grid = fetch.resample_grid(
            tmp_grid,
            spacing=layers[active_layer]["spacing"],
            region=buffer_region,
            registration=registration,
            verbose="q",
        )
        # print spacing, region, max,min, and registration of constraint grid
        print(f"constraint grid: {utils.get_grid_info(constraint_grid)}")

    if constraint_points is not None:
        # load constraint points into a dataframe, and create a subset within region
        constraint_points_all = pd.read_csv(
            constraint_points,
            sep=",",
            header="infer",
            index_col=None,
            compression="gzip",
        )

        constraint_points = utils.points_inside_region(
            constraint_points_all,
            inversion_region,
        )

        if kwargs.get("subset_constraints", True) is True:
            constraint_points_RIS = pygmt.select(
                data=constraint_points,
                F=kwargs.get("shapefile", "plotting/RIS_outline.shp"),
                coltypes="o",
            )

    print(f"gravity: {len(grav)} points")
    print(f"gravity avg. elevation: {int(np.nanmean(grav.z))}")

    if constraint_points is not None:
        print(f"bathymetry control points:{len(constraint_points)}")
        if kwargs.get("subset_constraints", True) is True:
            print(f"bathymetry control points within RIS:{len(constraint_points_RIS)}")

    grav.Gobs = grav.Gobs.astype(np.float64)
    grav.z = grav.z.astype(np.float64)
    grav.x = grav.x.astype(np.float64)
    grav.y = grav.y.astype(np.float64)

    outputs = [layers, grav, None, None, None]

    if constraint_grid is not None:
        outputs[2] = constraint_grid
    if constraint_points is not None:
        outputs[3] = constraint_points
        if kwargs.get("subset_constraints", True) is True:
            outputs[4] = constraint_points_RIS

    return outputs


class optuna_regional_RMSE:
    def __init__(self,
        regional_method,
        **kwargs,
        ):
        self.regional_method = regional_method
        self.true_regional = kwargs['true_regional']
        self.layers = kwargs['layers']
        self.input_grav = kwargs['input_grav']
        self.grav_spacing = kwargs['grav_spacing']
        self.inversion_region = kwargs['inversion_region']
        self.constraints = kwargs['constraints']

    def __call__(self, trial):
        if self.regional_method == 'filter':
            param = trial.suggest_int('param', 1e3, 1001e3, step=1e3)
        elif self.regional_method == 'trend':
            param = trial.suggest_int('param', 1, 20, step=1)
        elif self.regional_method == 'constraints':
            param = trial.suggest_int('param', 0, 10) # later divided by 10
        elif self.regional_method == 'eq_sources':
            param = trial.suggest_int('param', 10e3, 10010e3, step=10e3)

        rmse, df = regional_RMSE(
            true_regional=self.true_regional,
            layers=self.layers,
            input_grav=self.input_grav,
            grav_spacing=self.grav_spacing,
            inversion_region=self.inversion_region,
            constraints=self.constraints,
            regional_method=self.regional_method,
            param=param,
        )
        return rmse


def optimize_regional(
    n_trials,
    regional_method,
    plot_history=True,
    plot_results=True,
    **kwargs,
    ):
    # Create a new study
    study = optuna.create_study(
        direction='minimize',
        )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Invoke optimization of the objective function
    study.optimize(
        optuna_regional_RMSE(
            regional_method,
            **kwargs,
            ),
        n_trials=n_trials,
        )

    # Print the best result
    print(f"Method: {regional_method}, best result: {study.best_params}")

    if plot_history is True:
        plot_optimization_history(study).show()

    if plot_results is True:
        plot_slice(study).show()

    return study.trials_dataframe()


def optimize_regional_loop(
    n_trials,
    plot_history=True,
    plot_results=True,
    **kwargs,
    ):
    study_dfs = []
    study_names = []
    for i in ['filter', 'trend', 'constraints', 'eq_sources']:
        results = optimize_regional(n_trials, i, plot_history, plot_results, **kwargs)
        study_dfs.append(results)
        study_names.append(i)

    studies = dict(zip(study_names, study_dfs))
    return studies


class optuna_regional_RMSE_together:
    def __init__(self,
        **kwargs,
        ):
        self.true_regional = kwargs['true_regional']
        self.layers = kwargs['layers']
        self.input_grav = kwargs['input_grav']
        self.grav_spacing = kwargs['grav_spacing']
        self.inversion_region = kwargs['inversion_region']
        self.constraints = kwargs['constraints']

    def __call__(self, trial):

        regional_method = trial.suggest_categorical(
            'method',
            ['filter', 'trend', 'constraints', 'eq_sources'],
        )

        if regional_method == 'filter':
            param = trial.suggest_int('param', 1e3, 1001e3, step=1e3)
        elif regional_method == 'trend':
            param = trial.suggest_int('param', 1, 20, step=1)
        elif regional_method == 'constraints':
            param = trial.suggest_int('param', 0, 10) # later divided by 10
        elif regional_method == 'eq_sources':
            param = trial.suggest_int('param', 10e3, 10010e3, step=10e3)

        rmse, df = regional_RMSE(
            true_regional=self.true_regional,
            layers=self.layers,
            input_grav=self.input_grav,
            grav_spacing=self.grav_spacing,
            inversion_region=self.inversion_region,
            constraints=self.constraints,
            regional_method=regional_method,
            param=param,
        )
        return rmse


def optimize_regional_together(
    n_trials,
    plot_history=True,
    plot_results=True,
    **kwargs,
    ):
    # Create a new study
    study = optuna.create_study(
        direction='minimize',
        )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Invoke optimization of the objective function
    study.optimize(
        optuna_regional_RMSE_together(
            **kwargs,
            ),
        n_trials=n_trials,
        )

    # Print the best result
    print(f"Best result: {study.best_params}")

    if plot_history is True:
        plot_optimization_history(study).show()

    if plot_results is True:
        plot_slice(study).show()

    result = study.trials_dataframe().sort_values(by='value')

    return result, result.params_method.iloc[0]


def brute_optimize_regional(
    num,
    true_regional,
    layers,
    input_grav,
    grav_spacing,
    inversion_region,
    constraints,
    plot_best=True,
    plot_all=False,
    ):

    filters = np.linspace(1, 1000e3, num)
    trends = np.linspace(1, 20, num).astype(int)
    tensions = np.linspace(0.1, 1, num)
    depths = np.linspace(10e3, 10e6, num)

    data = {
        'filter': filters,
        'trend': trends,
        'constraints': tensions,
        'eq_sources': depths,}

    params = pd.DataFrame(data)

    for regional_method in params.columns:
        rms_values=[]
        print(regional_method)
        for i, j in enumerate(params[regional_method]):
            if regional_method == 'constraints':
                j=j/10
            rms, df_anomalies = regional_RMSE(
                true_regional=true_regional,
                layers=layers,
                input_grav=input_grav,
                grav_spacing=grav_spacing,
                inversion_region=inversion_region,
                constraints=constraints,
                regional_method=regional_method,
                param=j,
            )

            rms_values.append(rms)

            if plot_all is True:
                regional = pygmt.surface(
                    data=df_anomalies[["x", "y", "reg"]],
                    region=inversion_region,
                    spacing=grav_spacing,
                    T=0.25,
                    M="0c",
                    registration="g",
                )
                grids = utils.grd_compare(
                    true_regional,
                    regional,
                    plot=True,
                    plot_type='xarray',
                    grid1_name="true regional",
                    grid2_name=f"calculated regional: parameter={j}",
                    title=f"Method: {regional_method}, RMSE: {round(rms,2)}mGal",
                    points=constraints,
                    )

        params[f"{regional_method}_RMSE"]=rms_values

        if plot_best is True:

            best = params.sort_values(by=f"{regional_method}_RMSE", ascending=True)
            j = best[regional_method].iloc[0]
            if regional_method == 'constraints':
                j=j/10
            rms, df_anomalies = regional_RMSE(
                true_regional=true_regional,
                layers=layers,
                input_grav=input_grav,
                grav_spacing=grav_spacing,
                inversion_region=inversion_region,
                constraints=constraints,
                regional_method=regional_method,
                param=j,
            )

            rms_values.append(rms)

            regional = pygmt.surface(
                data=df_anomalies[["x", "y", "reg"]],
                region=inversion_region,
                spacing=grav_spacing,
                T=0.25,
                M="0c",
                registration="g",
            )
            grids = utils.grd_compare(
                true_regional,
                regional,
                plot=True,
                plot_type='xarray',
                grid1_name="true regional",
                grid2_name=f"calculated regional: parameter={j}",
                title=f"Method: {regional_method}, RMSE: {round(rms,2)}mGal",
                points=constraints,
                )
    return params

def regional_RMSE(
    true_regional,
    layers,
    input_grav,
    grav_spacing,
    inversion_region,
    constraints,
    regional_method,
    param
):

    df_anomalies = inv.anomalies(
        layers=layers,
        input_grav=input_grav,
        grav_spacing=grav_spacing,
        regional_method=regional_method,

        # KWARGS
        inversion_region=inversion_region,
        # filter kwargs
        filter=f"g{param}",
        # trend kwargs
        trend=param,
        fill_method="pygmt",
        # fill_method='rioxarray',
        # constraint kwargs
        constraints=constraints,
        tension_factor=param/10,
        # eq sources kwargs
        eq_sources=param,
        depth_type="relative",
        eq_damping=None,
        block_size=grav_spacing,
    )

    df = profile.sample_grids(df_anomalies, true_regional, "true_regional")
    rms = inv.RMSE(df.true_regional-df.reg)

    return rms, df_anomalies


def plot_best_param(study, regional_method, true_regional, **kwargs):

    print(f"\n{'':#<10} {regional_method} {'':#>10}")

    best = study.sort_values(by='value').iloc[0]
    print(best[['params_param', 'value']])

    df_anomalies = inv.anomalies(
        regional_method=regional_method,
        crs="3031",
        filter=f"g{best.params_param}",
        trend=best.params_param,
        fill_method="pygmt",
        tension_factor=best.params_param/10,
        eq_sources=best.params_param,
        depth_type="relative",
        eq_damping=None,
        block_size=kwargs['grav_spacing'],
        **kwargs,
    )
    if regional_method == 'filter':
        title= f"Method: {regional_method} (g{int(best.params_param/1e3)}km)"
    elif regional_method == 'trend':
        title= f"Method: {regional_method} (order={best.params_param})"
    elif regional_method == 'constraints':
        title= f"Method: {regional_method} (tension factor={best.params_param/10})"
    elif regional_method == 'eq_sources':
        title = f"Method: {regional_method} (Source depth={int(best.params_param/1e3)}km)"

    anom_grids = plots.anomalies_plotting(
        df_anomalies,
        region=kwargs['inversion_region'],
        grav_spacing=kwargs['grav_spacing'],
        title=title,
        constraints=kwargs['constraints'],
    )

    grids = utils.grd_compare(
        true_regional,
        anom_grids['reg'],
        plot=True,
        region=kwargs['inversion_region'],
        plot_type='xarray',
        cmap="RdBu_r",
        title=title,
        grid1_name="True regional misfit",
        grid2_name="best regional misfit",
    )


def plot_best_params_per_method(studies, **kwargs):
    for k, v in studies.items():
        plot_best_param(studies[k], k, **kwargs)