import numpy as np
import copy
import optuna
import pandas as pd
import pygmt
import verde as vd
import xarray as xr
import matplotlib.pyplot as plt
import harmonica as hm
from antarctic_plots import fetch, profile, utils, maps
from optuna.visualization import plot_optimization_history, plot_slice
import seaborn as sns
from typing import TYPE_CHECKING, Union

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


def normalize_xarray(da, low=0, high=1):

    # min_val = da.values.min()
    # max_val = da.values.max()

    min_val = da.quantile(0)
    max_val = da.quantile(1)

    return (high - low) * (((da - min_val) / (max_val - min_val)).clip(0, 1)) + low


def grids_to_prisms(
    top: xr.DataArray,
    bottom: xr.DataArray,
    density: Union[float, int, xr.DataArray],
    **kwargs,
):
    # if density provided as a single number, use it for all prisms
    if isinstance(density, (float, int)):
        dens = density * np.ones_like(top)
    # if density provided as a dataarray, map each density to the correct prisms
    elif isinstance(density, xr.DataArray):
        dens = density
    else:
        raise ValueError("invalid density type, should be a number or DataArray")

    # create layer of prisms based off input dataarrays
    prisms = hm.prism_layer(
        coordinates=(top.x.values, top.y.values),
        surface=top,
        reference=bottom,
        properties={
            "density": dens,
            "thickness": top - bottom,
        },
    )

    return prisms

def forward_grav_of_prismlayer(
    prisms: list,
    observation_points: pd.DataFrame,
    names: list,
    remove_mean=False,
    progressbar=False,
    plot: bool = True,
    **kwargs,
):
    df = copy.deepcopy(observation_points)

    for i, p in enumerate(prisms):
        grav = p.prism_layer.gravity(
            coordinates=(df.x, df.y, df.z),
            field="g_z",
            progressbar=progressbar,
        )

        # center around 0
        if remove_mean is True:
            grav -= grav.mean()

        # add to dataframe
        df[names[i]]=grav

    # add all together
    df["forward_total"] = df[names].sum(axis=1, skipna=True)

    if remove_mean is True:
        df.forward_total -= df.forward_total.mean()

    # grid each column into a common dataset
    grids = df.set_index(["y", "x"]).to_xarray()

    if plot is True:
        for i, n in enumerate(names+["forward_total"]):
            if i == 0:
                fig = maps.plot_grd(
                    grids[n],
                    cmap="vik+h0",
                    cbar_label="mGal",
                    title=n,
                    coast=False,
                )
            else:
                fig = maps.plot_grd(
                    grids[n],
                    cmap="vik+h0",
                    cbar_label="mGal",
                    title=n,
                    coast=False,
                    fig=fig,
                    origin_shift="xshift",
                )
        fig.show()

    return grids, df


def make_surface(
    spacing,
    region,
    top,
    checkerboard=True,
    amplitude=100,
    wavelength=10e3,
    plot=True,
):
    """
    Function to create either a flat or checkboard surface.

    Parameters
    ----------
    spacing : float
        grid spacing
    region : list
        region string [e,w,n,s]
    top : float
        top for flat surface, or baselevel for checkboard
    checkerboard : bool, optional
        choose whether to return a checkboard or flat surface, by default True
    amplitude : int, optional
        amplitude of checkboard, by default 100
    wavelength : float, optional
        checkboard wavelength in same units as grid, by default 10,000
    plot : bool, optional
        plot the results, by default True

    Returns
    -------
    xarray.DataArray
        the generated surface
    """
    # create a surface with repeating checkerboard pattern
    if checkerboard is True:
        synth = vd.synthetic.CheckerBoard(
            amplitude= amplitude,
            region=region,
            w_east=wavelength,
            w_north=wavelength,
        )

        surface = synth.grid(
            spacing=spacing,
            data_names="z",
            dims=('y', 'x')).z

        surface += top

    # create grid of coordinates
    else:
        coords = vd.grid_coordinates(
            region=region,
            spacing=spacing,
        )

        # create xarray dataarray from coordinates with a constant value as defined by 'top'
        surface = vd.make_xarray_grid(
            coords, np.ones_like(coords[0]) * top, data_names="z", dims=("y", "x")
        ).z

    if plot is True:
        # plot gravity and percentage contours
        fig, ax = plt.subplots()
        surface.plot(ax=ax, robust=True)
        ax.set_aspect("equal")

    return surface


def gravity_decay_buffer(
    buffer_perc,
    spacing = 1e3,
    interest_region = [-5e3, 5e3, -10e3, 15e3],
    top = 2e3,
    checkerboard=False,
    density_contrast=False,
    reference = -4e3,
    obs_height = 1200,
    density = 2300,
    plot = False,
    percentages = [.99, .95, .90],
    **kwargs
):
    """
    For a given buffer zone width (as percentage of x or y range) and domain parameters,
    calculate the max percent decay of the gravity anomaly within the region of interest.
    Decay is defined as the (max-min)/max.
    """
    # get x and y range of interest region
    x_diff = np.abs(interest_region[0]-interest_region[1])
    y_diff = np.abs(interest_region[2]-interest_region[3])

    # pick the bigger range
    max_diff = max(x_diff, y_diff)

    # calc buffer as percentage of width
    buffer_width = max_diff * (buffer_perc/100)

    # round to nearest multiple of spacing
    def round_to_input(num, multiple):
        return round(num / multiple) * multiple

    # round buffer width to nearest spacing interval
    buffer_width = round_to_input(buffer_width, spacing)

    # define buffer region
    buffer_region = utils.alter_region(interest_region, buffer=buffer_width)[1]

    # calculate buffer width in terms of number of cells
    buffer_cells = buffer_width / spacing

    # create surface
    surface = make_surface(
        spacing=spacing,
        region = buffer_region,
        top=top,
        checkerboard=checkerboard,
        amplitude=kwargs.get('amplitude', 100),
        wavelength=kwargs.get('wavelength', 10e3),
        plot=False,
    )

    # define what the reference is
    if density_contrast is True:
        # prism reference is mean value of surface
        reference = surface.values.mean()

        # positive densities above, negative below
        dens = surface.copy()
        dens.values[:] = (density)
        dens = dens.where(surface >= reference, -density)
    else:
        dens=density

    # create prism layer
    flat_prisms = grids_to_prisms(
        surface,
        reference,
        density=dens,
    )

    # create set of observation points
    data = vd.grid_coordinates(
        interest_region,
        spacing=spacing,
        extra_coords=obs_height,
    )

    observation_points = pd.DataFrame({
        'x': data[0].ravel(),
        'y': data[1].ravel(),
        'z': data[2].ravel()})

    # calculate forward gravity of prism layer
    forward_grid, forward_df = forward_grav_of_prismlayer(
        [flat_prisms],
        observation_points,
        plot=False,
        names=["Flat prisms"],
    )

    grav = forward_grid['forward_total']

    # get max decay value inside the region of interest
    if density_contrast is True:
        max_decay = (abs(grav.values.min())-grav.values.max())/(grav.values.max()*2)
    else:
        max_decay = (grav.values.max()-grav.values.min())/grav.values.max()

    if plot is True:

        results = f"maximum decay: {int(max_decay*100)}% \n" \
            f"buffer: {buffer_perc}% / {buffer_width}m / {int(buffer_cells)} cells"

        print(results)

        # plot diagonal profile
        if kwargs.get('plot_profile', False) is True:
            data_dict = profile.make_data_dict(
                ["Forward gravity"],
                [grav],
                ["black"],
            )
            # profile.plot_data(
            #     "points",
            #     start=(interest_region[0], interest_region[2]),
            #     stop=(interest_region[1], interest_region[3]),
            #     data_dict=data_dict,
            # )
            layers_dict = profile.make_data_dict(
                ["Surface"],
                [surface],
                ["black"],
            )
            profile.plot_profile(
                "points",
                start=(
                    interest_region[0],
                    (interest_region[3]-interest_region[2])/2),
                stop=(
                    interest_region[1],
                    (interest_region[3]-interest_region[2])/2),
                layers_dict = layers_dict,
                data_dict = data_dict,
                add_map = True,
                map_background = surface,
                inset=False,
                gridlines=False,
            )

        # plot histogram of gravity decay values
        sns.displot(forward_df.forward_total, kde=True)

        # add lines for various decay percentiles
        col=['r', 'cyan', 'k']
        for i, p in enumerate(percentages):
            plt.axvline(
                forward_df.forward_total.max()*p,
                color=col[i],
                label=f"{p*100}%",
                )
        plt.xlabel('grav')
        plt.ylabel('count')
        plt.title('Gravity decay within region of interest')
        plt.legend()

        # plot gravity and percentage contours
        fig, ax = plt.subplots()
        grav.plot(ax=ax, robust=True)
        ax.set_aspect("equal")

        grav.plot.contour(
            levels=[
                forward_df.forward_total.max()*percentages[0],
                forward_df.forward_total.max()*percentages[1],
                forward_df.forward_total.max()*percentages[2],
            ],
            colors=['k', 'cyan', 'r'],
            )

        ax.set_title('Forward gravity with decay contours')

    return max_decay, buffer_width, buffer_cells, grav


def optimal_buffer(
    top,
    reference,
    spacing,
    interest_region,
    density,
    obs_height,
    target,
    buffer_perc_range=[1,50],
    n_trials=25,
    density_contrast=False,
    checkerboard=False,
    amplitude=None,
    wavelength=None,
    full_search=False,
    ):
    def objective(
        trial,
        top,
        reference,
        spacing,
        interest_region,
        density,
        target,
        obs_height,
        checkerboard,
        density_contrast,
        amplitude,
        wavelength,
        ):
        """
        Find buffer percentage which gives a max decay within the region of interest closest
        to target percentage (i.e., target=5 is 5% decay).
        """
        buffer_perc = trial.suggest_int("buffer_perc", buffer_perc_range[0], buffer_perc_range[1])

        max_decay = gravity_decay_buffer(
            buffer_perc = buffer_perc,
            top = top,
            reference = reference,
            spacing = spacing,
            interest_region = interest_region,
            density = density,
            obs_height=obs_height,
            checkerboard=checkerboard,
            density_contrast=density_contrast,
            amplitude=amplitude,
            wavelength=wavelength,
            )[0]

        return np.abs((target/100)-max_decay)

    # Create a new study
    if full_search is True:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.GridSampler(
                search_space={"buffer_perc":list(range(buffer_perc_range[0],buffer_perc_range[1]))})
            )
    else:
        study = optuna.create_study(
            direction="minimize",
            )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # add custom logging function
    def logging_callback(study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
                )
            )

    # run optimization
    study.optimize(lambda trial:
        objective(
            trial,
            top,
            reference,
            spacing,
            interest_region,
            density,
            target,
            obs_height,
            checkerboard,
            density_contrast,
            amplitude,
            wavelength,
        ),
        n_trials=n_trials,
        callbacks=[logging_callback],
    )

    print(study.best_params)

    plot = plot_optimization_history(study)
    plot2 = plot_slice(study)

    plot.show()
    plot2.show()

    return study, study.trials_dataframe().sort_values(by="value")


def regional_seperation_quality(
    method,
    input_grav,
    input_forward_column,
    input_grav_column,
    true_regional,
    grav_spacing,
    inversion_region,
    constraints,
    regional_method,
    param,
):
    """
    Evaluate the effectiveness of the regional-residual seperation of the gravity
    misfit. 2 methods of evaluations, by comparing to the know regional field, or by
    minimizing the misfit at the constraint points.

    The returned RMSE value is used in the below optimization function as the object to
    minimize.

    Returns
    -------
    rms, df_anomalies
        returns a single RMSE value, and a dataframe
    """

    df_anomalies = inv.regional_seperation(
        input_grav=input_grav,
        input_forward_column = input_forward_column,
        input_grav_column = input_grav_column,
        grav_spacing=grav_spacing,
        regional_method=regional_method,
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

    if method == "regional comparison":
        # compare the true regional gravity with the calculated regional
        df = profile.sample_grids(df_anomalies, true_regional, "true_regional")
        rms = inv.RMSE(df.true_regional - df.reg)
    elif method == "minimize constraints":
        # grid the residuls
        residuals = pygmt.xyz2grd(
            data=df_anomalies[["x", "y", "res"]],
            region=inversion_region,
            spacing=grav_spacing,
            registration="g",
            verbose="q",
        )
        # sample the residuals at the constraint points
        df = profile.sample_grids(constraints, residuals, "residuals")
        rms = inv.RMSE(df.residuals)

    return rms, df_anomalies


class optuna_regional_RMSE_together:
    def __init__(
        self,
        **kwargs,
    ):
        self.method = kwargs["method"]
        self.true_regional = kwargs["true_regional"]
        self.input_grav = kwargs["input_grav"]
        self.input_forward_column = kwargs["input_forward_column"]
        self.input_grav_column = kwargs["input_grav_column"]
        self.grav_spacing = kwargs["grav_spacing"]
        self.inversion_region = kwargs["inversion_region"]
        self.constraints = kwargs["constraints"]

    def __call__(self, trial):

        regional_method = trial.suggest_categorical(
            "method",
            ["filter", "trend", "constraints", "eq_sources"],
        )

        if regional_method == "filter":
            param = trial.suggest_int("param", 1e3, 1001e3, step=1e3)
        elif regional_method == "trend":
            param = trial.suggest_int("param", 1, 20, step=1)
        elif regional_method == "constraints":
            param = trial.suggest_int("param", 0, 10) # later divided by 10
        elif regional_method == "eq_sources":
            param = trial.suggest_int("param", 10e3, 10010e3, step=10e3)

        rmse, df = regional_seperation_quality(
            method=self.method,
            true_regional=self.true_regional,
            input_grav=self.input_grav,
            input_forward_column=self.input_forward_column,
            input_grav_column=self.input_grav_column,
            grav_spacing=self.grav_spacing,
            inversion_region=self.inversion_region,
            constraints=self.constraints,
            regional_method=regional_method,
            param=param,
        )
        return rmse


class optuna_regional_RMSE:
    def __init__(
        self,
        regional_method,
        **kwargs,
    ):
        self.method = kwargs["method"]
        self.regional_method = regional_method
        self.true_regional = kwargs["true_regional"]
        self.input_grav = kwargs["input_grav"]
        self.input_forward_column = kwargs["input_forward_column"]
        self.input_grav_column = kwargs["input_grav_column"]
        self.grav_spacing = kwargs["grav_spacing"]
        self.inversion_region = kwargs["inversion_region"]
        self.constraints = kwargs["constraints"]

    def __call__(self, trial):
        if self.regional_method == "filter":
            param = trial.suggest_int("param", 1e3, 1001e3, step=1e3)
        elif self.regional_method == "trend":
            param = trial.suggest_int("param", 1, 20, step=1)
        elif self.regional_method == "constraints":
            param = trial.suggest_int("param", 0, 10)  # later divided by 10
        elif self.regional_method == "eq_sources":
            param = trial.suggest_int("param", 10e3, 10010e3, step=10e3)

        rmse, df = regional_seperation_quality(
            method=self.method,
            true_regional=self.true_regional,
            input_grav=self.input_grav,
            input_forward_column=self.input_forward_column,
            input_grav_column=self.input_grav_column,
            grav_spacing=self.grav_spacing,
            inversion_region=self.inversion_region,
            constraints=self.constraints,
            regional_method=self.regional_method,
            param=param,
        )
        return rmse


def logging_callback(study, frozen_trial):
        previous_best_value = study.user_attrs.get("previous_best_value", None)
        if previous_best_value != study.best_value:
            study.set_user_attr("previous_best_value", study.best_value)
            print(
                "Trial {} finished with best value: {} and parameters: {}. ".format(
                frozen_trial.number,
                frozen_trial.value,
                frozen_trial.params,
                )
            )


def optimize_regional_together(
    n_trials,
    plot_history=True,
    plot_results=True,
    **kwargs,
):
    # Create a new study
    study = optuna.create_study(
        direction="minimize",
    )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # Invoke optimization of the objective function
    study.optimize(
        optuna_regional_RMSE_together(
            **kwargs,
        ),
        n_trials=n_trials,
        callbacks=[logging_callback],
    )

    # Print the best result
    print(f"Best result: {study.best_params}")

    if plot_history is True:
        plot_optimization_history(study).show()

    if plot_results is True:
        plot_slice(study).show()

    result = study.trials_dataframe().sort_values(by="value")

    return result, result.params_method.iloc[0]


def optimize_regional(
    n_trials,
    regional_method,
    plot_history=True,
    plot_results=True,
    **kwargs,
):
    # Create a new study
    study = optuna.create_study(
        direction="minimize",
    )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # Invoke optimization of the objective function
    study.optimize(
        optuna_regional_RMSE(
            regional_method,
            **kwargs,
        ),
        n_trials=n_trials,
        callbacks=[logging_callback],
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
    for i in ["filter", "trend", "constraints", "eq_sources"]:
        results = optimize_regional(n_trials, i, plot_history, plot_results, **kwargs)
        study_dfs.append(results.sort_values(by="value"))
        study_names.append(i)

    studies = dict(zip(study_names, study_dfs))

    return studies


def plot_best_param(study, regional_method=None, **kwargs):

    best = study.sort_values(by="value").iloc[0]

    if regional_method is None:
        regional_method = best.params_method

    print(f"\n{'':#<10} {regional_method} {'':#>10}")
    print(best[["params_param", "value"]])

    rmse, df_anomalies = regional_seperation_quality(
        regional_method=regional_method,
        param=7,
        **kwargs,
    )

    if regional_method == "filter":
        title = f"Method: {regional_method} (g{int(best.params_param/1e3)}km)"
    elif regional_method == "trend":
        title = f"Method: {regional_method} (order={best.params_param})"
    elif regional_method == "constraints":
        title = f"Method: {regional_method} (tension factor={best.params_param/10})"
    elif regional_method == "eq_sources":
        title = (
            f"Method: {regional_method} (Source depth={int(best.params_param/1e3)}km)"
        )

    score = best.value

    anom_grids = plots.anomalies_plotting(
        df_anomalies,
        region=kwargs["inversion_region"],
        grav_spacing=kwargs["grav_spacing"],
        title=title + f" Optimization score: {round(score,2)} mGal",
        constraints=kwargs["constraints"],
        input_forward_column=kwargs.get('input_forward_column', 'forward') ,
        input_grav_column=kwargs.get('input_grav_column', 'grav'),
    )

    if kwargs["method"] == "regional comparison":
        _ = utils.grd_compare(
            kwargs["true_regional"],
            anom_grids["reg"],
            plot=True,
            region=kwargs["inversion_region"],
            plot_type="xarray",
            cmap="RdBu_r",
            title=title,
            grid1_name="True regional misfit",
            grid2_name="best regional misfit",
        )

    return df_anomalies

def plot_best_params_per_method(studies, **kwargs):
    for k, v in studies.items():
        plot_best_param(studies[k], k, **kwargs)
