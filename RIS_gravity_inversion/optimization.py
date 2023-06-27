import math
import pathlib
import pickle
import warnings

import joblib
import numpy as np
import optuna
import pandas as pd
import psutil
import xarray as xr
from tqdm_joblib import tqdm_joblib

import RIS_gravity_inversion.inversion as inv
import RIS_gravity_inversion.utils as inv_utils


def logging_callback(study, frozen_trial):
    """
    custom optuna callback, only print trial if it's the best value yet.
    """
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


def optuna_parallel(
    study_name,
    study_storage,
    objective,
    n_trials=100,
    maximize_cpus=True,
    parallel=True,
):
    """
    Run optuna optimization in parallel. Pre-define the study, storage, and objective
    function and input them here.
    """
    # load study metadata from storage
    study = optuna.load_study(storage=study_storage, study_name=study_name)

    # set up parallel processing and run optimization
    if parallel is True:

        @inv_utils.supress_stdout
        def optimize_study(study_name, storage, objective, n_trials):
            study = optuna.load_study(study_name=study_name, storage=storage)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(
                objective,
                n_trials=n_trials,
            )

        if maximize_cpus is True:
            optuna_max_cores(
                n_trials, optimize_study, study_name, study_storage, objective
            )
        elif maximize_cpus is False:
            optuna_1job_per_core(
                n_trials, optimize_study, study_name, study_storage, objective
            )

    # run in normal, non-parallel mode
    elif parallel is False:
        study = optuna.load_study(
            study_name=study_name,
            storage=study_storage,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Progress bar is experimental")
            study.optimize(
                objective,
                n_trials=n_trials,
                show_progress_bar=True,
            )

    # reload the study
    study = optuna.load_study(
        study_name=study_name,
        storage=study_storage,
    )

    # get dataframe from study and sort by objective value
    study_df = study.trials_dataframe()

    return study, study_df


def optuna_max_cores(n_trials, optimize_study, study_name, study_storage, objective):
    """
    Set up optuna optimization in parallel splitting up the number of trials over all
    available cores.
    """
    # get available cores (UNIX and Windows)
    num_cores = len(psutil.Process().cpu_affinity())

    # set trials per job
    trials_per_job = math.ceil(n_trials / num_cores)

    # set number of jobs
    if n_trials >= num_cores:
        n_jobs = num_cores
    else:
        n_jobs = n_trials

    with tqdm_joblib(desc="Optimizing", total=n_trials) as progress_bar:  # noqa F841
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(optimize_study)(
                study_name,
                study_storage,
                objective,
                n_trials=trials_per_job,
            )
            for i in range(n_trials)
        )


def optuna_1job_per_core(
    n_trials,
    optimize_study,
    study_name,
    study_storage,
    objective,
):
    """
    Set up optuna optimization in parallel giving each available core 1 trial.
    """
    trials_per_job = 1
    with tqdm_joblib(desc="Optimizing", total=n_trials) as progress_bar:  # noqa F841
        joblib.Parallel(n_jobs=int(n_trials / trials_per_job))(
            joblib.delayed(optimize_study)(
                study_name,
                study_storage,
                objective,
                n_trials=trials_per_job,
            )
            for i in range(int(n_trials / trials_per_job))
        )


def optimal_buffer(
    top,
    reference,
    spacing,
    interest_region,
    density,
    obs_height,
    target,
    buffer_perc_range=[1, 50],
    n_trials=25,
    density_contrast=False,
    checkerboard=False,
    amplitude=None,
    wavelength=None,
    full_search=False,
):
    """
    Optuna optimization to find best buffer zone width.
    """

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
        Find buffer percentage which gives a max decay within the region of interest
        closest
        to target percentage (i.e., target=5 is 5% decay).
        """
        buffer_perc = trial.suggest_int(
            "buffer_perc", buffer_perc_range[0], buffer_perc_range[1]
        )

        max_decay = inv_utils.gravity_decay_buffer(
            buffer_perc=buffer_perc,
            top=top,
            reference=reference,
            spacing=spacing,
            interest_region=interest_region,
            density=density,
            obs_height=obs_height,
            checkerboard=checkerboard,
            density_contrast=density_contrast,
            amplitude=amplitude,
            wavelength=wavelength,
        )[0]

        return np.abs((target / 100) - max_decay)

    # Create a new study
    if full_search is True:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.GridSampler(
                search_space={
                    "buffer_perc": list(
                        range(buffer_perc_range[0], buffer_perc_range[1])
                    )
                }
            ),
        )
    else:
        study = optuna.create_study(
            direction="minimize",
        )

    # Disable logging
    optuna.logging.set_verbosity(optuna.logging.WARN)

    # run optimization
    study.optimize(
        lambda trial: objective(
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

    plot = optuna.visualization.plot_optimization_history(study)
    plot2 = optuna.visualization.plot_slice(study)

    plot.show()
    plot2.show()

    return study, study.trials_dataframe().sort_values(by="value")


class optimal_regional_params:
    def __init__(
        self,
        comparison_method,
        filter_limits,
        trend_limits,
        constraints_limits,
        eq_sources_limits,
        regional_method_options=None,
        regional_method=None,
        **kwargs,
    ):
        self.comparison_method = comparison_method
        self.regional_method_options = regional_method_options
        self.regional_method = regional_method
        self.filter_limits = filter_limits
        self.trend_limits = trend_limits
        self.constraints_limits = constraints_limits
        self.eq_sources_limits = eq_sources_limits
        self.kwargs = kwargs

    def __call__(self, trial):
        # define parameter space
        # use either single regional method, or include regional method as a parameter
        # to optimize on
        if self.regional_method_options is not None:
            regional_method = trial.suggest_categorical(
                "regional_method", self.regional_method_options
            )
        elif self.regional_method is not None:
            regional_method = self.regional_method
        else:
            raise ValueError(
                "Need to supply either `regional_method` or `regional_method_options`"
            )

        if regional_method == "filter":
            param = trial.suggest_int(
                "filter",
                self.filter_limits[0],
                self.filter_limits[1],
                step=self.filter_limits[2],
            )
        elif regional_method == "trend":
            param = trial.suggest_int(
                "trend",
                self.trend_limits[0],
                self.trend_limits[1],
                step=self.trend_limits[2],
            )
        elif regional_method == "constraints":
            param = trial.suggest_float(
                "constraints",
                self.constraints_limits[0],
                self.constraints_limits[1],
                step=self.constraints_limits[2],
            )
        elif regional_method == "eq_sources":
            param = trial.suggest_int(
                "eq_sources",
                self.eq_sources_limits[0],
                self.eq_sources_limits[1],
                step=self.eq_sources_limits[2],
            )
        else:
            raise ValueError("invalid string for region_method")

        # run regional seperations and return RMSE based on comparison method.
        with inv_utils.HiddenPrints():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Engine 'cfgrib'")
                rmse, df = inv_utils.regional_seperation_quality(
                    regional_method=regional_method,
                    comparison_method=self.comparison_method,
                    param=param,
                    **self.kwargs,
                )

        return rmse


class optimal_regional_eq_sources_params:
    def __init__(
        self,
        damping_limits,
        depth_limits,
        **kwargs,
    ):
        self.damping_limits = damping_limits
        self.depth_limits = depth_limits
        self.kwargs = kwargs

    def __call__(self, trial):
        # define parameter space
        damping = trial.suggest_float(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
        )
        depth = trial.suggest_float(
            "depth",
            self.depth_limits[0],
            self.depth_limits[1],
        )
        # run regional seperations and return RMSE based on comparison method.
        with inv_utils.HiddenPrints():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Engine 'cfgrib'")
                rmse, df = inv_utils.regional_seperation_quality(
                    param=depth,
                    eq_damping=damping,
                    regional_method="eq_sources",
                    comparison_method="regional_comparison",
                    **self.kwargs,
                )

        return rmse


class optimal_eq_source_params:
    def __init__(
        self,
        coordinates,
        data,
        damping_limits,
        depth_limits,
        **kwargs,
    ):
        self.coordinates = coordinates
        self.data = data
        self.damping_limits = damping_limits
        self.depth_limits = depth_limits
        self.kwargs = kwargs

    def __call__(self, trial):
        # define parameter space
        damping = trial.suggest_float(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
        )
        depth = trial.suggest_float(
            "depth",
            self.depth_limits[0],
            self.depth_limits[1],
        )

        score = inv_utils.eq_sources_score(
            params={"damping": damping, "depth": depth},
            coordinates=self.coordinates,
            data=self.data,
            **self.kwargs,
        )

        return score


class CV_inversion:
    def __init__(
        self,
        training_data,
        testing_data,
        fname,
        true_surface=None,
        l2_norm_limits=None,
        l2_norm_step=0.01,
        weights_exponent_limits=None,
        weights_exponent_step=None,
        starting_prisms=None,
        constraints=None,
        inversion_region=None,
        layer_spacing=None,
        damping_limits=None,  # will be 10^lim, with lim as integers
        damping_step=0.1,
        weights_grid_kwargs=None,
        inversion_kwargs=None,
    ):
        self.fname = fname
        self.true_surface = true_surface
        self.training_data = training_data
        self.testing_data = testing_data
        self.damping_limits = damping_limits
        self.damping_step = damping_step
        self.weights_exponent_limits = weights_exponent_limits
        self.starting_prisms = starting_prisms
        self.constraints = constraints
        self.weights_exponent_step = weights_exponent_step
        self.layer_spacing = layer_spacing
        self.l2_norm_limits = l2_norm_limits
        self.l2_norm_step = l2_norm_step
        self.weights_grid_kwargs = weights_grid_kwargs
        self.inversion_kwargs = inversion_kwargs

    def __call__(self, trial):
        # define parameter space

        if self.damping_limits is None:
            solver_damping = self.inversion_kwargs.get("solver_damping")
        else:
            exp = trial.suggest_float(
                "damping",
                self.damping_limits[0],
                self.damping_limits[1],
                step=self.damping_step,
            )
            solver_damping = 10**exp

        if self.weights_exponent_limits is None:
            assert self.starting_prisms is not None
        else:
            weights_exponent = trial.suggest_float(
                "weights_exponent",
                self.weights_exponent_limits[0],
                self.weights_exponent_limits[1],
                step=self.weights_exponent_step,
            )
            # re-create the weights grid
            weights = inv_utils.normalized_mindist(
                self.constraints,
                self.starting_prisms.drop("weights"),
                mindist=self.layer_spacing / np.sqrt(2),
                low=0,
                high=1,
                **self.weights_grid_kwargs,
            )
            weights = weights**weights_exponent

            self.starting_prisms["weights"] = weights

        if self.l2_norm_limits is None:
            l2_norm_tolerance = self.inversion_kwargs.get("l2_norm_tolerance")
        else:
            l2_norm_tolerance = trial.suggest_float(
                "l2_norm_tolerance",
                self.l2_norm_limits[0],
                self.l2_norm_limits[1],
                step=self.l2_norm_step,
            )

        # run inversion and return RMSE between true and starting layer
        with inv_utils.HiddenPrints():
            (
                rmse,
                prism_results,
                grav_results,
                params,
                elapsed_time,
                constraints_rmse,
            ) = inv.inversion_RMSE(
                self.true_surface,
                input_grav=self.training_data,
                prism_layer=self.starting_prisms,
                plot=False,
                solver_damping=solver_damping,
                l2_norm_tolerance=l2_norm_tolerance,
                **{
                    k: v
                    for k, v in self.inversion_kwargs.items()
                    if k not in ["l2_norm_tolerance"]
                },
            )

        results = dict(
            rmse=rmse,
            prism_results=prism_results,
            grav_results=grav_results,
            params=params,
            elapsed_time=elapsed_time,
            constraints_rmse=constraints_rmse,
        )

        # save results into each trial
        trial.set_user_attr("true_surface_RMSE", rmse)
        trial.set_user_attr("elapsed_time", elapsed_time)
        trial.set_user_attr("constraints_rmse", constraints_rmse)

        # remove if exists
        pathlib.Path(f"{self.fname}_trial_{trial.number}_results.pickle").unlink(
            missing_ok=True
        )

        # save results to pickle dataframes
        with open(f"{self.fname}_trial_{trial.number}_results.pickle", "wb") as fout:
            pickle.dump(results, fout)

        # grid resulting prisms dataframe
        prism_ds = prism_results.set_index(["northing", "easting"]).to_xarray()

        # get last iteration's layer result
        cols = [s for s in prism_results.columns.to_list() if "_layer" in s]
        final_surface = prism_ds[cols[-1]]

        # set reference as un-inverted surface mean
        zref = prism_ds.starting_bed.values.mean()
        # zref = prisms_ds.top.values.min()

        # set density contrast
        density_contrast = prism_ds.density.values.max()

        assert zref == prism_ds.top.values.min()
        assert zref == prism_ds.bottom.values.max()
        assert prism_ds.density.values.max() == -prism_ds.density.values.min()

        # create new prism layer
        prism_layer = inv_utils.grids_to_prisms(
            surface=final_surface,
            reference=zref,
            density=xr.where(
                final_surface >= zref, density_contrast, -density_contrast
            ),
        )

        # calculate forward gravity of prisms on testing points
        grav_grid, grav_df = inv_utils.forward_grav_of_prismlayer(
            [prism_layer],
            self.testing_data,
            names=["test_point_grav"],
            remove_median=False,
            progressbar=False,
            plot=False,
        )

        # compare new forward with observed
        observed = (
            self.testing_data[self.inversion_kwargs.get("input_grav_column")]
            - self.testing_data.reg
        )
        predicted = grav_df.test_point_grav

        dif = predicted - observed

        score = inv_utils.RMSE(dif)

        return score


class optimal_inversion_damping_and_weights:
    def __init__(
        self,
        true_surface,
        weights_inner_limits,
        weights_outer_limits,
        weights_step,
        damping_limits,  # will be 10^lim, with lim as integers, [-2, 0] = [.01, .1, 1]
        starting_prisms,
        objectives=["RMSE"],  # can include: 'RMSE', 'duration', 'constraints'
        constraints=None,
        **kwargs,
    ):
        self.true_surface = true_surface
        self.damping_limits = damping_limits
        self.weights_inner_limits = weights_inner_limits
        self.weights_outer_limits = weights_outer_limits
        self.starting_prisms = starting_prisms
        self.objectives = objectives
        self.constraints = constraints
        self.weights_step = weights_step
        self.kwargs = kwargs

    def __call__(self, trial):
        # define parameter space
        exp = trial.suggest_int(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
        )
        solver_damping = 10**exp
        if len(self.weights_inner_limits) == 1:
            weights_inner = self.weights_inner_limits[0]
            weights_outer = trial.suggest_int(
                "weights_outer",
                self.weights_outer_limits[0],
                self.weights_outer_limits[1],
                self.weights_step,
            )
        else:
            weights_inner = trial.suggest_int(
                "weights_inner",
                self.weights_inner_limits[0],
                self.weights_inner_limits[1],
                self.weights_step,
            )
            weights_outer = trial.suggest_int(
                "weights_outer",
                self.weights_outer_limits[0],
                self.weights_outer_limits[1],
                self.weights_step,
            )

        # re-create the weights grid
        weights, _ = inv_utils.constraints_grid(
            self.constraints,
            self.starting_prisms.drop("weights"),
            inner_bound=weights_inner,
            outer_bound=weights_outer,
            low=0,
            high=1,
            region=self.kwargs.get("inversion_region"),
            interp_type="spline",
        )

        self.starting_prisms["weights"] = weights

        # run inversion and return RMSE between true and starting layer
        (
            rmse,
            prism_results,
            grav_results,
            params,
            elapsed_time,
            constraints_RMSE,
        ) = inv.inversion_RMSE(
            self.true_surface,
            self.constraints,
            prism_layer=self.starting_prisms,
            plot=False,
            solver_damping=solver_damping,
            **self.kwargs,
        )

        objective_values = []
        if "RMSE" in self.objectives:
            objective_values.append(rmse)

        if "duration" in self.objectives:
            objective_values.append(elapsed_time)

        if "constraints" in self.objectives:
            objective_values.append(constraints_RMSE)
            print(f"RMSE between surfaces at constraints: {constraints_RMSE} m")

        return tuple(objective_values)


class optimal_inversion_damping:
    def __init__(
        self,
        true_surface,
        damping_limits,  # will be 10^lim, with lim as integers, so [-2, 0]=[.01, .1, 1]
        objectives=["RMSE"],  # can include: 'RMSE', 'duration', 'constraints'
        constraints=None,
        **kwargs,
    ):
        self.true_surface = true_surface
        self.damping_limits = damping_limits
        self.objectives = objectives
        self.constraints = constraints
        self.kwargs = kwargs

    def __call__(self, trial):
        # define parameter space
        exp = trial.suggest_int(
            "damping",
            self.damping_limits[0],
            self.damping_limits[1],
        )
        solver_damping = 10**exp

        # run inversion and return RMSE between true and starting layer
        (
            rmse,
            prism_results,
            grav_results,
            params,
            elapsed_time,
            constraints_RMSE,
        ) = inv.inversion_RMSE(
            self.true_surface,
            self.constraints,
            plot=False,
            solver_damping=solver_damping,
            **self.kwargs,
        )

        objective_values = []
        if "RMSE" in self.objectives:
            objective_values.append(rmse)

        if "duration" in self.objectives:
            objective_values.append(elapsed_time)

        if "constraints" in self.objectives:
            objective_values.append(constraints_RMSE)
            print(f"RMSE between surfaces at constraints: {constraints_RMSE} m")

        return tuple(objective_values)


class optimal_inversion_params:
    def __init__(
        self,
        true_surface,
        verde_damping_limits,
        scipy_damping_limits,
        objectives=["RMSE"],  # can include: 'RMSE', 'duration', 'constraints'
        constraints=None,
        **kwargs,
    ):
        self.true_surface = true_surface
        self.verde_damping_limits = verde_damping_limits
        self.scipy_damping_limits = scipy_damping_limits
        self.objectives = objectives
        self.constraints = constraints
        self.kwargs = kwargs

    def __call__(self, trial):
        # define parameter space
        deriv_type = trial.suggest_categorical("deriv_type", ["annulus", "prisms"])
        solver_type = trial.suggest_categorical(
            "solver_type", ["verde least squares", "scipy least squares"]
        )
        if solver_type == "verde least squares":
            exp = trial.suggest_int(
                "verde_damping",
                self.verde_damping_limits[0],
                self.verde_damping_limits[1],
            )
        elif solver_type == "scipy least squares":
            exp = trial.suggest_int(
                "scipy_damping",
                self.scipy_damping_limits[0],
                self.scipy_damping_limits[1],
            )
        solver_damping = 10**exp

        # run inversion and return RMSE between true and starting layer
        (
            rmse,
            prism_results,
            grav_results,
            params,
            elapsed_time,
            constraints_RMSE,
        ) = inv.inversion_RMSE(
            self.true_surface,
            self.constraints,
            plot=False,
            deriv_type=deriv_type,
            solver_type=solver_type,
            solver_damping=solver_damping,
            **self.kwargs,
        )

        objective_values = []
        if "RMSE" in self.objectives:
            objective_values.append(rmse)

        if "duration" in self.objectives:
            objective_values.append(elapsed_time)

        if "constraints" in self.objectives:
            objective_values.append(constraints_RMSE)
            print(f"RMSE between surfaces at constraints: {constraints_RMSE} m")

        return tuple(objective_values)


def get_best_of_each_param(study, objectives):
    if len(study.directions) > 1:
        best_trials = {}
        for i, j in enumerate(study.best_trials):
            best_trials[j.number] = {}
            best_trials[j.number]["number"] = j.number
            for k, l in enumerate(objectives):
                best_trials[j.number][l] = j.values[k]
            if "duration" not in objectives:
                seconds = (j.datetime_complete - j.datetime_start).total_seconds()
                best_trials[j.number]["duration"] = seconds
            best_trials[j.number]["params_deriv_type"] = j.params["deriv_type"]
            best_trials[j.number]["params_solver_type"] = j.params["solver_type"]
            best_trials[j.number]["params_scipy_damping"] = 10 ** j.params.get(
                "scipy_damping", np.nan
            )
            best_trials[j.number]["params_verde_damping"] = 10 ** j.params.get(
                "verde_damping", np.nan
            )
        df = pd.DataFrame(best_trials.values())
        df.set_index("number", inplace=True, drop=False)

    elif len(study.directions) == 1:
        df = (
            study.trials_dataframe()
            .sort_values(by="value")
            .rename(columns={"value": objectives[0]})
        )

    try:
        best_verde = df[df.params_solver_type == "verde least squares"].iloc[0].number
        print(f"best trial w/ solver = verde: index {best_verde}")
    except:  # noqa: E722
        best_verde = None

    try:
        best_scipy = df[df.params_solver_type == "scipy least squares"].iloc[0].number
        print(f"best trial w/ solver = scipy: index {best_scipy}")
    except:  # noqa: E722
        best_scipy = None

    try:
        best_prisms = df[df.params_deriv_type == "prisms"].iloc[0].number
        print(f"best trial w/ deriv = prisms: index {best_prisms}")
    except:  # noqa: E722
        best_prisms = None

    try:
        best_annulus = df[df.params_deriv_type == "annulus"].iloc[0].number
        print(f"best trial w/ deriv = annulus: index {best_annulus}")
    except:  # noqa: E722
        best_annulus = None

    return df, best_verde, best_scipy, best_prisms, best_annulus


def get_best_params_from_study(study):
    # get results from best trial
    # for single objective
    try:
        if study.best_params["solver_type"] == "scipy least squares":
            solver_damping = 10 ** study.best_params["scipy_damping"]
        elif study.best_params["solver_type"] == "verde least squares":
            solver_damping = 10 ** study.best_params["verde_damping"]
        max_layer_change_per_iter = (
            None  # study.best_params['max_layer_change_per_iter']
        )
        deriv_type = study.best_params["deriv_type"]
        solver_type = study.best_params["solver_type"]

    # for multiobjective
    except RuntimeError:
        best = min(study.best_trials, key=lambda t: t.values[0])
        if best.params["solver_type"] == "scipy least squares":
            solver_damping = 10 ** best.params["scipy_damping"]
        elif best.params["solver_type"] == "verde least squares":
            solver_damping = 10 ** best.params["verde_damping"]
        max_layer_change_per_iter = None  # best.params['max_layer_change_per_iter']
        deriv_type = best.params["deriv_type"]
        solver_type = best.params["solver_type"]

    params_dict = {
        "deriv_type": deriv_type,
        "max_layer_change_per_iter": max_layer_change_per_iter,
        "solver_damping": solver_damping,
        "solver_type": solver_type,
    }
    return params_dict
