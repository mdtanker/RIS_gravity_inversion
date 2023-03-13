# %%
import tempfile

import optuna
import pandas as pd
import pytest

from RIS_gravity_inversion import optimization


def test_optuna_parallel():
    """
    test that the optuna parallel optimization works
    Just tests that functions runs, doesnt' test that it's properly running in parallel.
    """
    with tempfile.NamedTemporaryFile() as file:
        file_storage = optuna.storages.JournalFileStorage(file.name)
        storage = optuna.storages.JournalStorage(file_storage)
        study_name = file.name

        # create study
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            sampler=optuna.samplers.TPESampler(n_startup_trials=5),
            direction="minimize",
        )

        # create a dummy objective function
        def objective(trial):
            x = trial.suggest_int("x", 0, 10)
            return (x) ** 2

        # run the optimization
        study, study_df = optimization.optuna_parallel(
            study_name=study_name,
            study_storage=storage,
            objective=objective,
            n_trials=10,
            parallel=True,
            maximize_cpus=True,
        )

        pd.testing.assert_frame_equal(study_df, study.trials_dataframe())

        assert study.best_value < 5
