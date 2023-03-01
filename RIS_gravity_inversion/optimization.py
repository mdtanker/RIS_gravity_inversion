import optuna
from optuna.storages import JournalStorage, JournalFileStorage
import numpy as np
import copy

import pandas as pd
import pygmt
import verde as vd
import xarray as xr
import matplotlib.pyplot as plt
import harmonica as hm
from antarctic_plots import fetch, profile, utils, maps
import seaborn as sns
import plotly
from typing import TYPE_CHECKING, Union
import os, sys
import warnings
import time
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError
import joblib
from tqdm_joblib import tqdm_joblib

import RIS_gravity_inversion.inversion as inv
import RIS_gravity_inversion.plotting as plots
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
    Run optuna optimization in parallel. Pre-define the study, storage, and objective function and input them here.
    """
    # load study metadata from storage
    study = optuna.load_study(storage=study_storage, study_name=study_name)
    
    # set up parallel processing and run optimization
    if parallel is True:
        @inv_utils.supress_stdout
        def optimize_study(study_name, storage, objective, n_trials):
            study = optuna.load_study(study_name=study_name, storage=storage)
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=n_trials,)
        if maximize_cpus is True:
            optuna_max_cores(n_trials, optimize_study, study_name, study_storage, objective)
        elif maximize_cpus is False:
            optuna_1job_per_core(n_trials, optimize_study, study_name, study_storage, objective)
    
    # run in normal, non-parallel mode
    elif parallel is False:
        study = optuna.load_study(
            study_name=study_name,
            storage=study_storage,
            )
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
    Set up optuna optimization in parallel splitting up the number of trials over all available cores.
    """
    # get available cores (UNIX and Windows)
    import psutil
    num_cores = len(psutil.Process().cpu_affinity())

    # set trials per job
    import math
    trials_per_job = math.ceil(n_trials/num_cores)

    # set number of jobs
    if n_trials >= num_cores:
        n_jobs = num_cores
    else:
        n_jobs = n_trials

    with tqdm_joblib(desc="Optimizing", total=n_trials) as progress_bar:
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(optimize_study)(
                study_name,
                study_storage,
                objective,
                n_trials=trials_per_job,
                )
            for i in range(n_trials))
        
        
def optuna_1job_per_core(n_trials, optimize_study, study_name, study_storage, objective, ):
    """
    Set up optuna optimization in parallel giving each available core 1 trial.
    """
    trials_per_job = 1
    with tqdm_joblib(desc="Optimizing", total=n_trials) as progress_bar:
      joblib.Parallel(n_jobs=int(n_trials/trials_per_job))(
            joblib.delayed(optimize_study)(
                study_name,
                study_storage,
                objective,
                n_trials=trials_per_job,
                )
            for i in range(int(n_trials/trials_per_job)))
    
    
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
        Find buffer percentage which gives a max decay within the region of interest closest
        to target percentage (i.e., target=5 is 5% decay).
        """
        buffer_perc = trial.suggest_int("buffer_perc", buffer_perc_range[0], buffer_perc_range[1])

        max_decay = inv_utils.gravity_decay_buffer(
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

    plot = optuna.visualization.plot_optimization_history(study)
    plot2 = optuna.visualization.plot_slice(study)

    plot.show()
    plot2.show()

    return study, study.trials_dataframe().sort_values(by="value")


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
        Find buffer percentage which gives a max decay within the region of interest closest
        to target percentage (i.e., target=5 is 5% decay).
        """
        buffer_perc = trial.suggest_int("buffer_perc", buffer_perc_range[0], buffer_perc_range[1])

        max_decay = inv_utils.gravity_decay_buffer(
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
        **kwargs):

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
        # use either single regional method, or include regional method as a parameter to optimize on
        if self.regional_method_options is not None:
            regional_method = trial.suggest_categorical("regional_method", self.regional_method_options)
        elif self.regional_method is not None:
            regional_method = self.regional_method
        else:
            raise ValueError("Need to supply either `regional_method` or `regional_method_options`")
            
        if regional_method == "filter":
            param = trial.suggest_int("filter", 
                self.filter_limits[0], self.filter_limits[1], step=self.filter_limits[2]) 
        elif regional_method == "trend":
            param = trial.suggest_int("trend", 
                self.trend_limits[0], self.trend_limits[1], step=self.trend_limits[2])
        elif regional_method == "constraints":
            param = trial.suggest_float("constraints", 
                self.constraints_limits[0], self.constraints_limits[1], step=self.constraints_limits[2])
        elif regional_method == "eq_sources":
            param = trial.suggest_int("eq_sources", 
                self.eq_sources_limits[0], self.eq_sources_limits[1], step=self.eq_sources_limits[2])
        else:
            raise ValueError("invalid string for region_method")
        
        # run regional seperations and return RMSE based on comparison method. 
        rmse, df = inv_utils.regional_seperation_quality(
            regional_method = regional_method,
            comparison_method = self.comparison_method,
            param = param, 
            **self.kwargs,
            )

        return rmse
    
 
class optimal_inversion_params:
    def __init__(
        self, 
        true_surface,  
        verde_damping_limits, 
        scipy_damping_limits,
        objectives = ['RMSE'], # can include: 'RMSE', 'duration', 'constraints'
        constraints = None,
        **kwargs):
        self.true_surface = true_surface
        self.verde_damping_limits = verde_damping_limits
        self.scipy_damping_limits = scipy_damping_limits
        self.objectives = objectives
        self.constraints = constraints
        self.kwargs = kwargs
           
    def __call__(self, trial):
        # define parameter space
        deriv_type = trial.suggest_categorical("deriv_type", ["annulus", "prisms"])
        solver_type = trial.suggest_categorical("solver_type", ["verde least squares","scipy least squares"])
        if solver_type == "verde least squares":
            solver_damping = trial.suggest_float("verde_damping", self.verde_damping_limits[0], self.verde_damping_limits[1])
        elif solver_type == "scipy least squares":
            solver_damping = trial.suggest_float("scipy_damping", self.scipy_damping_limits[0], self.scipy_damping_limits[1])

        # run inversion and return RMSE between true and starting layer
        rmse, prism_results, grav_results, params, elapsed_time, constraints_RMSE = inv.inversion_RMSE(
            self.true_surface,
            self.constraints,
            plot = False,
            deriv_type = deriv_type,
            solver_type = solver_type,
            solver_damping = solver_damping,
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
    if len(study.directions)>1:
        best_trials = {}
        for i,j in enumerate(study.best_trials):
            best_trials[j.number] = {}
            best_trials[j.number]['number'] = j.number
            for k, l in enumerate(objectives):
                best_trials[j.number][l]=j.values[k]
            if "duration" not in objectives:
                time = (j.datetime_complete - j.datetime_start).total_seconds()
                best_trials[j.number]["duration"]=time
            best_trials[j.number]["params_deriv_type"]=j.params["deriv_type"]
            best_trials[j.number]["params_solver_type"]=j.params["solver_type"]
            best_trials[j.number]["params_scipy_damping"]=j.params.get("scipy_damping", np.nan)
            best_trials[j.number]["params_verde_damping"]=j.params.get("verde_damping", np.nan)  
        df = pd.DataFrame(best_trials.values())
        df.set_index('number', inplace=True, drop=False)

    elif len(study.directions)==1:
        df = study.trials_dataframe().sort_values(by="value")
        
    try:
        best_verde = df[df.params_solver_type == 'verde least squares'].iloc[0].number
        print(f"best trial w/ solver = verde: index {best_verde}")
    except:
        best_verde = None
    
    try:
        best_scipy = df[df.params_solver_type == 'scipy least squares'].iloc[0].number
        print(f"best trial w/ solver = scipy: index {best_scipy}")
    except:
        best_scipy = None
    
    try:
        best_prisms = df[df.params_deriv_type == 'prisms'].iloc[0].number
        print(f"best trial w/ deriv = prisms: index {best_prisms}")
    except:
        best_prisms = None
    
    try:
        best_annulus = df[df.params_deriv_type == 'annulus'].iloc[0].number
        print(f"best trial w/ deriv = annulus: index {best_annulus}")
    except:
        best_annulus = None
        
    return df, best_verde, best_scipy, best_prisms, best_annulus


def get_best_params_from_study(study):
    # get results from best trial
    # for single objective
    try:
        if study.best_params['solver_type'] == 'scipy least squares':
                solver_damping = study.best_params['scipy_damping']
        elif study.best_params['solver_type'] == 'verde least squares':
            solver_damping = study.best_params['verde_damping']
        max_layer_change_per_iter = None #study.best_params['max_layer_change_per_iter']
        deriv_type = study.best_params['deriv_type']
        solver_type = study.best_params['solver_type']
        
    # for multiobjective
    except RuntimeError:
        best = min(study.best_trials, key=lambda t: t.values[0])
        if best.params['solver_type'] == 'scipy least squares':
            solver_damping = best.params['scipy_damping']
        elif best.params['solver_type'] == 'verde least squares':
            solver_damping = best.params['verde_damping']
        max_layer_change_per_iter = None #best.params['max_layer_change_per_iter']
        deriv_type = best.params['deriv_type']
        solver_type = best.params['solver_type']

    params_dict = {
        'deriv_type': deriv_type,
        'max_layer_change_per_iter': max_layer_change_per_iter,
        'solver_damping': solver_damping,
        'solver_type': solver_type,
    }
    return params_dict


# def optimal_inversion_param_old(
#     true_surface,
#     inversion_region,
#     study_name,
#     study_storage,
#     n_trials=100,
#     timeout=500,
#     plot_optimization=True,
#     plot_results=True,
#     grav_spacing=None,
#     **kwargs
# ):
#     # Define objective function
#     @timeout_decorator.timeout(timeout, use_signals=True)
#     @inv_utils.supress_stdout
#     def objective(trial):
#         """
#         Calculate the RMSE of the inversion results compared with the true starting
#         topography.
#         """
#         nonlocal true_surface, inversion_region, plot_results, kwargs

#         deriv_type = trial.suggest_categorical("deriv_type",
#             ["annulus", "prisms"])
#         # max_layer_change_per_iter = trial.suggest_float("max_layer_change_per_iter",
#         #     0, 1000, step=10)
#         solver_type = trial.suggest_categorical("solver_type",
#             [
#                 "verde least squares",
#                 "scipy least squares",
#             ])
#         if solver_type == "verde least squares":
#             solver_damping = trial.suggest_float("verde_damping",
#                 0, 1000)
#             # exp = trial.suggest_float("verde_damping",
#             #     1, 9, step=.1) # 10 - 1e10
#             # solver_damping = 10 ** exp
#         elif solver_type == "scipy least squares":
#             solver_damping = trial.suggest_float("scipy_damping",
#                 0, .1)
#             # exp = trial.suggest_float("scipy_damping",
#             #     -6, 2, step=.1) # 1e-6 - 100
#             # solver_damping = 10 ** exp

#         with warnings.catch_warnings():
#             warnings.filterwarnings("once",
#                 message="Under-determined problem detected")
#             warnings.filterwarnings("once",
#                 message="DataFrame is highly fragmented.")
#             rmse, prism_results, grav_results, params, elapsed_time = inv.inversion_RMSE(
#                 true_surface,
#                 plot=False,
#                 inversion_region=inversion_region,
#                 deriv_type=deriv_type,
#                 solver_type=solver_type,
#                 solver_damping=solver_damping,
#                 # max_layer_change_per_iter=max_layer_change_per_iter,
#                 **kwargs,
#                 )
#         # save results for plotting
#         if plot_results is True:
#             try:
#                 trial.set_user_attr("prism_results", prism_results)
#                 trial.set_user_attr("grav_results", grav_results)
#                 trial.set_user_attr("params", params)
#             except TypeError:
#                 pass

#         return rmse

#     # load study metadata from storage
#     study = optuna.load_study(storage=study_storage, study_name=study_name)

#     #####
#     #####
#     # standard
#     #####
#     #####
#     # # Invoke optimization of the objective function
#     # optuna.logging.set_verbosity(optuna.logging.WARNING)
#     # study.optimize(
#     #     objective,
#     #     n_trials=n_trials,
#     #     # n_jobs=10, # -1 sets to number of CPUs
#     #     show_progress_bar=True,
#     #     catch=(TimeoutError,),
#     #     callbacks=[
#     #         optuna.study.MaxTrialsCallback(n_trials, states=(optuna.trial.TrialState.COMPLETE,)),
#     #         logging_callback,
#     #     ],
#     # )

#     #####
#     #####
#     # with joblib package
#     #####
#     #####
#     @inv_utils.supress_stdout
#     def optimize_study(study_name, storage, objective, n_trials):
#         study = optuna.load_study(
#             study_name=study_name,
#             storage=storage,
#             )
#         optuna.logging.set_verbosity(optuna.logging.WARNING)
#         study.optimize(
#             objective,
#             n_trials=n_trials,
#             catch=(TimeoutError,),
#             # callbacks=[
#             # logging_callback,
#             # ],
#         )

#     #####
#     # with max available CPUs
#     #####
#     # get available cores (UNIX and Windows)
#     import psutil
#     num_cores = len(psutil.Process().cpu_affinity())

#     # set trials per job
#     import math
#     trials_per_job = math.ceil(n_trials/num_cores)

#     # set number of jobs
#     if n_trials >= num_cores:
#         n_jobs = num_cores
#     else:
#         n_jobs = n_trials

#     with tqdm_joblib(desc="Optimizing", total=n_trials) as progress_bar:
#         joblib.Parallel(n_jobs=n_jobs)(
#             joblib.delayed(optimize_study)(
#                 study_name,
#                 study_storage,
#                 objective,
#                 n_trials=trials_per_job,
#                 )
#             for i in range(n_trials))

#     #####
#     # with 1 job per CPU
#     #####
#     # trials_per_job = 1
#     # with tqdm_joblib(desc="Optimizing", total=n_trials) as progress_bar:
#     #   optuna.logging.set_verbosity(optuna.logging.WARNING)
#     #   joblib.Parallel(n_jobs=int(n_trials/trials_per_job))(
#     #         joblib.delayed(optimize_study)(
#     #             study_name,
#     #             study_storage,
#     #             objective,
#     #             # n_trials=n_trials,
#     #             n_trials=trials_per_job,
#     #             )
#     #         # for i in range(n_trials))
#     #         for i in range(int(n_trials/trials_per_job)))


#     # reload the study
#     study = optuna.load_study(
#         study_name=study_name,
#         storage=study_storage,
#         )

#     # Print the best result
#     print(study.best_params)

#     # get dataframe from study
#     study_df = study.trials_dataframe()

#     # sort by rmse
#     study_df = study_df.sort_values(by="value")

#     if plot_optimization is True:
#         plots.plot_optuna_inversion_figures_single(study)

#     if plot_results is True:
#         plots.plot_best_inversion(
#             true_surface,
#             inversion_region,
#             study=study,
#             grav_spacing=grav_spacing,
#             **kwargs
#         )

#     return study, study_df